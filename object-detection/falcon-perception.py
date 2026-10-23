#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "falcon-perception>=1.0.0",
#   "datasets>=4.5.0",
#   "huggingface-hub>=1.12.0",
#   "pillow",
# ]
# ///
"""Zero-shot object detection + instance segmentation -> a YOLO detection dataset.

Falcon-Perception finds every instance of a class you name, with no training and
no label set. Output is a detection dataset in `yolo` format, so it feeds the
other recipes in this directory directly:

    validate-hf-dataset.py you/first-pass --bbox-format yolo
    stats-hf-dataset.py    you/first-pass --bbox-format yolo
    convert-hf-dataset.py  you/first-pass you/for-review --from yolo --to label_studio
    #  ... a human corrects the first pass in Label Studio ...
    diff-hf-datasets.py    you/first-pass you/corrected      # IoU -> zero-shot accuracy

RUNS ON YOUR LAPTOP TOO. Unusually for this repo no CUDA GPU is required: on
Apple Silicon it selects the MLX backend automatically. Slower (~6 s/img vs
~0.4 on an A10G), which is fine for the step that matters locally -- checking
your class name works on your images before spending GPU hours on the corpus.

    # 1. does the model do the thing?
    uv run falcon-perception.py --image page.jpg --query illustration --preview

    # 2. does it work on MY data? (first rows of the real corpus)
    uv run falcon-perception.py --dataset biglam/british-library-book-images \
        --config plates --limit 3 --preview

    # 3. the whole corpus, on a GPU
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN \
        https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py \
        --dataset biglam/british-library-book-images --config plates \
        --id-col fname --query illustration --out you/plates-illustrations

Output goes wherever --out points:

    --out you/plates-illustrations   a Hub dataset (yolo format, feeds the scripts above)
    --out results.json               a local JSON file  -- no Hub push
    --out results.jsonl              a local JSONL file -- one record per line
    --out results.parquet            a local parquet file
    --json                           also print the records on stdout, for piping
    (omit --out)                     print a summary and, with --preview, annotated JPEGs

For images in a bucket rather than a dataset, see falcon-perception-bucket.py.

MEASURED LIMITS -- not guesses; each one cost a failed run:

  * --query takes a CLASS NAME, never an instruction. "illustration" works;
    "the illustration, excluding captions" returns nothing at all.
  * ONE class per run. A combined query ("illustration, map, portrait") returned
    6 instances where three single-class passes found 24, and emitted <|absence|>
    on the richest image. The output vocabulary has no class token either, so
    instances could not be attributed even if the counts held. N classes = N runs.
  * NO confidence scores -- the model has no score token. Two triage proxies are
    emitted instead: `rectangularity` (mask area / bbox area; measured 0.34-1.00,
    low = irregular, 1.00 = clean rectangular plate) and `area`. Sort review by
    rectangularity ascending and apply an area floor; the smallest box seen was
    941 px^2 and was spurious.
  * torch.compile is OFF. Per-image dynamic shapes break Inductor
    ("ValueError: Exponent must be non-negative" after symbolic-shape recursion).
  * CUDA graphs are OFF by default. engine_config_for_gpu() sizes itself from the
    GPU and ignores host RAM; on the 15 GB-host-RAM flavors (t4-small, a10g-small
    -- both measured) the container is OOMKilled (exit 137) before one image is
    processed. Pick a flavor with >15 GB `ram` from `hf jobs hardware --json`,
    or pass --cudagraph knowingly.
"""

import argparse
import glob as globlib
import hashlib
import io
import itertools
import json
import os
import pathlib
import platform
import sys
import time


def stable_id(key):
    """Deterministic int64 image_id from the source key.

    COCO-style consumers (transformers RT-DETR / D-FINE annotation prep) require an
    INTEGER image_id -- a string crashes them with "ValueError: too many dimensions
    'str'". A content hash (not a running index) keeps ids identical across separate
    runs over the same source, so per-class runs merge on image_id cleanly.
    """
    return int.from_bytes(hashlib.blake2b(str(key).encode(), digest_size=8).digest(), "big") >> 1

# ── backend / engine selection ──────────────────────────────────────────────
# The MLX and torch APIs match parameter-for-parameter, but are NOT drop-in:
# torch also needs setup_torch_config(), a compile= kwarg, and every batch tensor
# moved with .to(device). Omitting the last fails deep inside
# flex_attention.create_block_mask, nowhere near the actual cause.


def pick_backend(requested):
    if requested != "auto":
        return requested
    return "mlx" if (sys.platform == "darwin" and platform.machine() == "arm64") else "torch"


def guard_mlx_memory(frac=0.55):
    """MLX allocates from unified memory with NO default cap.

    An oversized image through the AnyUp upsampler exhausts system RAM and hangs
    the whole machine -- the process is never OOM-killed, because there is no
    separate GPU pool for the kernel to reclaim. Measured: 0.22 MP ran fine;
    5.4 MP took down a 32 GiB Mac whose MLX default ceiling was 30.4 GiB.
    """
    try:
        import mlx.core as mx

        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        mx.set_memory_limit(int(total * frac))
        print(f"mlx memory capped at {total * frac / 2**30:.1f} GiB", flush=True)
    except Exception as e:
        print(f"WARNING: could not cap MLX memory ({e}) -- a large image may hang this machine", flush=True)


# ── sources: every source yields (key, PIL image) ───────────────────────────


def src_images(spec):
    from falcon_perception.data import load_image
    from PIL import Image

    if spec.startswith(("http://", "https://")):
        from urllib.parse import unquote

        yield unquote(spec.rsplit("/", 1)[-1])[:120], load_image(spec).convert("RGB")
        return
    paths = sorted(globlib.glob(spec)) if any(c in spec for c in "*?[") else [spec]
    if not paths:
        raise SystemExit(f"no files matched {spec!r}")
    for p in paths:
        yield os.path.basename(p), Image.open(p).convert("RGB")


def src_dataset(repo, config, split, image_col, id_col):
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset(repo, config, split=split, streaming=True)
    for idx, row in enumerate(ds):
        im = row[image_col]
        if isinstance(im, dict) and "bytes" in im:
            im = Image.open(io.BytesIO(im["bytes"]))
        yield (str(row.get(id_col)) if id_col else str(idx)), im.convert("RGB")


# ── helpers ─────────────────────────────────────────────────────────────────


def pair_bboxes(raw):
    """[{x,y}, {h,w}, ...] -> [{x,y,h,w}, ...]. xy is the normalised CENTRE.

    Centre-not-corner is why the output is natively `yolo` -- and why a corner
    reading would put every box out of bounds.
    """
    boxes, cur = [], {}
    for e in raw:
        if not isinstance(e, dict):
            continue
        cur.update(e)
        if all(k in cur for k in ("x", "y", "h", "w")):
            boxes.append(dict(cur))
            cur = {}
    return boxes


def fit(im, max_dim, backend):
    """Downscale BEFORE the preprocessor sees it -- on MLX the full-size
    intermediate is what exhausts memory."""
    budget = max_dim if backend == "mlx" else max_dim * 2
    if max(im.size) > budget:
        im = im.copy()
        im.thumbnail((budget, budget))
    return im


def save_preview(key, im, boxes, rles, out_dir):
    import numpy as np
    from PIL import Image, ImageDraw
    from pycocotools import mask as mask_utils

    os.makedirs(out_dir, exist_ok=True)
    W, H = im.size
    canvas = np.array(im.convert("RGB"), dtype=np.float32)
    for i, rle in enumerate(rles):
        m = rle if isinstance(rle.get("counts"), bytes) else {**rle, "counts": str(rle["counts"]).encode()}
        try:
            dec = mask_utils.decode(m).astype("uint8")
        except Exception:
            continue
        if dec.shape != (H, W):  # mask is at model resolution -- NEAREST only
            dec = np.array(Image.fromarray(dec).resize((W, H), Image.NEAREST))
        col = np.array([(255, 60, 60), (60, 160, 255), (80, 200, 120)][i % 3], dtype=np.float32)
        sel = dec > 0
        canvas[sel] = canvas[sel] * 0.65 + col * 0.35
    out = Image.fromarray(canvas.clip(0, 255).astype("uint8"))
    pen = ImageDraw.Draw(out)
    for b in boxes:
        cx, cy, bw, bh = b["x"] * W, b["y"] * H, b["w"] * W, b["h"] * H
        pen.rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], outline=(255, 220, 0), width=3)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)[:80]
    path = os.path.join(out_dir, f"{safe}.jpg")
    out.save(path)
    return path


def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


SCRIPT_URL = "https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py"


def jobs_flavor() -> str:
    """Return the HF Jobs hardware flavor this run is on, or "" when it cannot be told.

    HF Jobs sets ACCELERATOR to "cpu" or "gpu", not the flavor name, so the flavor is looked up
    in the public Jobs hardware list: first by GPU model and count, then by the CPU_CORES and
    MEMORY values that Jobs sets in the container (to tell a10g-small from a10g-large, or
    cpu-basic from cpu-upgrade). Any failure (no network, older huggingface_hub) returns "".
    """
    hardware = os.environ.get("ACCELERATOR") or ""
    if "-" in hardware or any(character.isdigit() for character in hardware):
        return hardware  # already a flavor name
    try:
        from huggingface_hub import HfApi

        gpu_name, gpu_count = "", 0
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
        except ImportError:
            # Torch-free drivers (e.g. the -server recipes): ask nvidia-smi instead.
            import subprocess

            try:
                listed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10, check=True,
                ).stdout
                names = [line.strip() for line in listed.splitlines() if line.strip()]
                if names:
                    gpu_name, gpu_count = names[0], len(names)
            except (OSError, subprocess.SubprocessError):
                pass
        if gpu_count == 0 and hardware == "gpu":
            return ""  # a GPU run whose GPU cannot be identified: do not guess a CPU flavor

        # Whole-word match, so "L4" does not match "NVIDIA L40S" ("NVIDIA A100-SXM4-80GB" -> A100).
        gpu_words = gpu_name.upper().replace("-", " ").split()
        candidates = []
        for flavor in HfApi(token=False).list_jobs_hardware():
            accelerator = flavor.accelerator
            if gpu_count == 0 and accelerator is None:
                candidates.append(flavor)
            elif (
                gpu_count > 0
                and accelerator is not None
                and accelerator.model.upper() in gpu_words
                and int(accelerator.quantity) == gpu_count
            ):
                candidates.append(flavor)
        if not candidates:
            return ""

        cores = float(os.environ.get("CPU_CORES") or 0)
        memory_gb = float((os.environ.get("MEMORY") or "0").upper().rstrip("GIB ") or 0)

        def distance(flavor) -> float:
            # Relative gap to the listed vCPUs and RAM; Jobs reports slightly different numbers.
            listed_cores = float(flavor.cpu.split()[0])
            listed_memory = float(flavor.ram.split()[0])
            return abs(listed_cores - cores) / listed_cores + abs(listed_memory - memory_gb) / listed_memory

        return min(candidates, key=distance).name
    except Exception:
        return ""


def push_card(repo_id, query, counters):
    """Dataset card with the repo's canonical provenance stamp (see AGENTS.md)."""
    from huggingface_hub import DatasetCard

    on_jobs = os.environ.get("JOB_ID") is not None  # set by HF Jobs in-container
    hw = jobs_flavor()  # flavor name, e.g. "a10g-small"; "" on CPU
    origin = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        + (f" (`{hw}`)" if hw else "")
    ) if on_jobs else "Generated"
    tags = "\n".join(f"- {t}" for t in (["uv-script", "hf-jobs"] if on_jobs else ["uv-script"]))
    args_summary = " ".join(sys.argv[1:])
    card = DatasetCard(f"""---
tags:
{tags}
---

# Zero-shot detection: `{query}`

{counters["images"]} images, {counters["instances"]} instances. Labels are **zero-shot weak
labels** from [Falcon-Perception](https://huggingface.co/tiiuae/Falcon-Perception) -- no human
annotated anything, and recall against human truth is unmeasured. `objects.bbox` is `yolo`
format (normalised centre x, y, w, h); `objects.category` is a `ClassLabel` named `{query}`;
`objects.rectangularity` (mask area / box area) is the triage proxy -- the model emits no
confidence scores.

## Reproduction

{origin} with the [`falcon-perception.py`]({SCRIPT_URL}) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN \\
    {SCRIPT_URL} \\
    {args_summary}
```
""")
    try:
        card.push_to_hub(repo_id)
    except Exception as e:
        print(f"WARNING: could not push dataset card ({e})", flush=True)


# ── the two generation paths ────────────────────────────────────────────────


def run_paged(model, tokenizer, items, prompt, args):
    """CUDA: TII's continuous-batching engine. ~0.4 s/img on an A10G."""
    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_inference import (
        PagedInferenceEngine,
        SamplingParams,
        Sequence,
        engine_config_for_gpu,
    )

    cfg = engine_config_for_gpu(max_image_size=args.max_dim, dtype=model.dtype)
    print(f"paged config: {cfg}", flush=True)
    engine = PagedInferenceEngine(
        model, tokenizer, ImageProcessor(patch_size=16, merge_size=1),
        max_seq_length=8192, capture_cudagraph=args.cudagraph, **cfg,
    )
    sp = SamplingParams(
        args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.end_of_query_token_id],
        coord_dedup_threshold=0.01,
    )
    for chunk in batched(items, args.chunk):
        chunk = [(k, fit(im, args.max_dim, "torch")) for k, im in chunk]
        seqs = [
            Sequence(text=prompt, image=im, min_image_size=256,
                     max_image_size=args.max_dim, request_idx=i, task=args.task)
            for i, (_, im) in enumerate(chunk)
        ]
        t0 = time.perf_counter()
        engine.generate(seqs, sampling_params=sp)
        dt = (time.perf_counter() - t0) / len(seqs)
        for (k, im), s in zip(chunk, seqs):
            yield k, im, s.output_aux, dt


def run_batch(model, tokenizer, items, prompt, args, backend, max_seq_len):
    """MLX (and a torch fallback): the readable reference engine. ~6 s/img on an M1 Pro."""
    if backend == "mlx":
        from falcon_perception.mlx.batch_inference import BatchInferenceEngine, process_batch_and_generate
    else:
        from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate

    engine = BatchInferenceEngine(model, tokenizer)
    for chunk in batched(items, 1 if backend == "mlx" else args.chunk):
        chunk = [(k, fit(im, args.max_dim, backend)) for k, im in chunk]
        b = process_batch_and_generate(
            tokenizer, [(im, prompt) for _, im in chunk],
            max_length=max_seq_len, min_dimension=256, max_dimension=args.max_dim,
        )
        if backend != "mlx":  # torch needs every tensor on the model's device
            import torch

            b = {k2: (v.to(model.device) if torch.is_tensor(v) else v) for k2, v in b.items()}
        t0 = time.perf_counter()
        _, auxes = engine.generate(
            tokens=b["tokens"], pos_t=b["pos_t"], pos_hw=b["pos_hw"],
            pixel_values=b["pixel_values"], pixel_mask=b["pixel_mask"],
            max_new_tokens=args.max_new_tokens, temperature=0.0, task=args.task,
        )
        dt = (time.perf_counter() - t0) / len(chunk)
        for (k, im), aux in zip(chunk, auxes):
            yield k, im, aux, dt


# ── main ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_mutually_exclusive_group(required=True)
    s.add_argument("--image", help="path, URL, or glob ('scans/*.jpg')")
    s.add_argument("--dataset", help="Hub dataset repo id (streamed)")
    p.add_argument("--config")
    p.add_argument("--split", default="train")
    p.add_argument("--image-col", default="image")
    p.add_argument("--id-col", default=None, help="stable id column; falls back to row index")
    p.add_argument("--query", required=True, help="a CLASS NAME, not an instruction")
    p.add_argument("--task", default="segmentation", choices=["segmentation", "detection"])
    p.add_argument("--out", default=None,
                   help="where results go. A path ending .json/.jsonl/.parquet writes that file "
                        "locally; anything else is treated as a Hub dataset repo id. Omit for "
                        "stdout + previews only.")
    p.add_argument("--json", action="store_true",
                   help="also print the records as JSON on stdout (for piping / agents)")
    p.add_argument("--private", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="3 for a sense check")
    p.add_argument("--preview", action="store_true", help="save annotated JPEGs")
    p.add_argument("--preview-dir", default="./falcon-preview")
    p.add_argument("--max-dim", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])
    p.add_argument("--engine", default="auto", choices=["auto", "batch", "paged"])
    p.add_argument("--cudagraph", action="store_true", help="opt IN -- can OOM the host on small flavors")
    p.add_argument("--mlx-mem-fraction", type=float, default=0.55)
    args = p.parse_args()

    backend = pick_backend(args.backend)
    if backend == "mlx":
        guard_mlx_memory(args.mlx_mem_fraction)
    use_paged = args.engine == "paged" or (args.engine == "auto" and backend == "torch")
    if use_paged and backend == "mlx":
        print("paged engine is CUDA-only -- using batch", flush=True)
        use_paged = False
    print(f"backend={backend} engine={'paged' if use_paged else 'batch'} query={args.query!r}", flush=True)

    from falcon_perception import PERCEPTION_MODEL_ID, build_prompt_for_task, load_and_prepare_model
    from pycocotools import mask as mask_utils

    kw = {}
    if backend == "torch":
        from falcon_perception import setup_torch_config

        setup_torch_config()
        kw = {"compile": False}  # dynamic image shapes break Inductor
    t = time.perf_counter()
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID,
        dtype="float16" if backend == "mlx" else "bfloat16",
        backend=backend, **kw,
    )
    print(f"model loaded in {time.perf_counter() - t:.1f}s", flush=True)
    prompt = build_prompt_for_task(args.query, args.task)

    items = src_images(args.image) if args.image else src_dataset(
        args.dataset, args.config, args.split, args.image_col, args.id_col)
    if args.limit:
        # islice STOPS the iterator; a filter would keep streaming the whole corpus.
        items = itertools.islice(items, args.limit)

    gen = (run_paged(model, tokenizer, items, prompt, args) if use_paged
           else run_batch(model, tokenizer, items, prompt, args, backend, model_args.max_seq_len))

    # Records are STREAMED, never accumulated: a whole-corpus run used to hold every
    # decoded PIL image in a list until the final push (~3-12 MB each -> tens of GB
    # RSS -> OOM-killed before anything was pushed).
    counters = {"images": 0, "instances": 0}
    json_rows = []  # populated only when --json; records here are image-free
    t0 = time.perf_counter()

    def iter_records():
        for key, im, aux, dt in gen:
            W, H = im.size
            boxes = pair_bboxes(aux.bboxes_raw)
            rles = list(aux.masks_rle)
            bbox, area, rect = [], [], []
            for i, b in enumerate(boxes):
                bbox.append([b["x"], b["y"], b["w"], b["h"]])  # yolo: cx, cy, w, h normalised
                a = b["w"] * b["h"]
                area.append(a)
                r = 0.0
                if i < len(rles):  # rectangularity -- the only triage signal available
                    try:
                        m = rles[i]
                        if isinstance(m.get("counts"), str):
                            m = {**m, "counts": m["counts"].encode()}
                        # measure box area in the MASK's own frame (rle size), not the
                        # fitted image's -- the two never match, and mixing frames skews r
                        mh, mw = (m.get("size") or [H, W])[:2]
                        r = min(float(mask_utils.area(m)) / max(a * mw * mh, 1.0), 1.0)
                    except Exception:
                        r = 0.0
                rect.append(r)
            counters["images"] += 1
            counters["instances"] += len(bbox)
            print(f"[{counters['images']}] {key[:55]:55s} {len(bbox):2d} inst  {dt:.2f}s", flush=True)
            if args.preview:
                print(f"     -> {save_preview(key, im, boxes, rles, args.preview_dir)}", flush=True)
            rec = {
                "image": im, "image_id": stable_id(key), "source_id": key,
                "width": W, "height": H,
                "objects": {"bbox": bbox, "category": [0] * len(bbox),
                            "area": area, "rectangularity": rect},
                "n_instances": len(bbox),
                "masks_rle": json.dumps([
                    {**m, "counts": m["counts"].decode() if isinstance(m.get("counts"), bytes) else m.get("counts")}
                    for m in rles
                ]),
            }
            if args.json:
                json_rows.append(plain(rec))
            yield rec

    def plain(r):  # everything except the PIL image, which is not serialisable
        return {k: v for k, v in r.items() if k != "image"}

    hub_out = args.out and not args.out.endswith((".json", ".jsonl", ".parquet"))

    if hub_out:
        from datasets import ClassLabel, Dataset, Features, Image as ImageFeat, Sequence as SeqFeat, Value

        feats = Features({
            "image": ImageFeat(), "image_id": Value("int64"), "source_id": Value("string"),
            "width": Value("int32"), "height": Value("int32"),
            # category is a ClassLabel named after the query, so the class name travels
            # with the dataset (viewer, trainers, id2label) instead of a bare 0.
            "objects": {"bbox": SeqFeat(SeqFeat(Value("float32"))),
                        "category": SeqFeat(ClassLabel(names=[args.query])),
                        "area": SeqFeat(Value("float32")),
                        "rectangularity": SeqFeat(Value("float32"))},
            "n_instances": Value("int32"), "masks_rle": Value("string"),
        })
        # Stream through an ArrowWriter: accumulating records in RAM OOMs whole-corpus
        # runs, and the from_generator APIs pickle their callable, which this closure
        # (live generator, loaded model) cannot survive. The writer flushes to disk per
        # batch; from_file memory-maps the result back for the push.
        import tempfile

        from datasets.arrow_writer import ArrowWriter

        arrow_path = os.path.join(tempfile.mkdtemp(prefix="falcon-out-"), "data.arrow")
        with ArrowWriter(features=feats, path=arrow_path, writer_batch_size=100) as writer:
            for rec in iter_records():
                writer.write(feats.encode_example(rec))
            writer.finalize()
        ds = Dataset.from_file(arrow_path)
        ds.push_to_hub(args.out, private=args.private)
        push_card(args.out, args.query, counters)
        if args.json:
            print(json.dumps(json_rows, indent=2), flush=True)
    else:
        rows = [plain(r) for r in iter_records()]
        if args.json:
            print(json.dumps(rows, indent=2), flush=True)
        if args.out and args.out.endswith(".json"):
            pathlib.Path(args.out).write_text(json.dumps(rows, indent=2))
        elif args.out and args.out.endswith(".jsonl"):
            pathlib.Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        elif args.out:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(rows), args.out, compression="zstd")

    wall = time.perf_counter() - t0
    n = counters["images"]
    print(f"\n{n} images in {wall:.1f}s ({n / max(wall, 1e-9):.2f} img/s)", flush=True)
    if args.out:
        print(f"{counters['instances']} instances -> {args.out}", flush=True)
    if hub_out:
        print(f"\nNEXT: validate-hf-dataset.py {args.out} --bbox-format yolo", flush=True)


main()
