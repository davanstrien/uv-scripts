# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "vllm",
#     "huggingface-hub",
# ]
# ///
"""
Superseded by generate-embeddings-saturate.py, which runs a vLLM server on a pinned image.

High-throughput embedding generation with vLLM pooling mode — the "scale" variant of
generate-embeddings.py, for large *decoder* embedding models (e.g. Qwen3-Embedding). On
Qwen3-Embedding-0.6B this was ~2x the sentence-transformers throughput on the same GPU.

Prefer the plain sentence-transformers `generate-embeddings.py` unless you specifically need
vLLM throughput: this variant has a heavier cold-start and two footguns handled below
(the embedding-mode kwarg drifted across vLLM versions; vLLM does not auto-truncate).

Runs on the BARE uv image (vLLM ships the CUDA toolkit + flashinfer as wheels).

    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings-vllm.py \\
        stanfordnlp/imdb your-name/imdb-embeddings --column text --model Qwen/Qwen3-Embedding-0.6B --private
"""
import argparse
import logging
import os
import time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate-embeddings-vllm")


def build_llm(LLM, model, max_model_len, gpu_mem_util):
    """vLLM's embedding-mode selector drifted: modern uses runner='pooling', old used
    task='embed'. A wrong kwarg raises TypeError at init (cheap) → fall through."""
    base = dict(enforce_eager=True, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem_util)
    for label, extra in [("runner", {"runner": "pooling"}), ("task", {"task": "embed"}), ("auto", {})]:
        try:
            llm = LLM(model=model, **base, **extra)
            log.info(f"engine init via '{label}'")
            return llm
        except TypeError as te:
            log.warning(f"ctor '{label}' rejected: {te}")
    raise RuntimeError("no vLLM constructor form accepted")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dataset")
    ap.add_argument("output_dataset")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--column", default="text")
    ap.add_argument("--output-column", default="embeddings")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--config", default=None, help="dataset config name (e.g. wikipedia needs one)")
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from huggingface_hub import DatasetCard, login
    from vllm import LLM
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU available — vLLM needs one. Run with a GPU flavor, e.g. "
                         "`hf jobs uv run --flavor l4x1 ...` (or use generate-embeddings.py on CPU).")
    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    ds = (load_dataset(args.input_dataset, args.config, split=args.split) if args.config
          else load_dataset(args.input_dataset, split=args.split))
    if args.output_column in ds.column_names:
        raise SystemExit(f"Output column {args.output_column!r} already exists — pick another.")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    texts = [t if isinstance(t, str) and t.strip() else " " for t in ds[args.column]]
    n = len(texts)

    llm = build_llm(LLM, args.model, args.max_model_len, args.gpu_mem_util)
    embed_fn = getattr(llm, "embed", None) or getattr(llm, "encode")

    # vLLM raises on inputs > max_model_len (no silent truncation) — pre-truncate at the tokenizer.
    # Tokenize each text once (not twice) — this pass is CPU-bound on large datasets.
    tk = llm.get_tokenizer()
    cap = max(8, args.max_model_len - 16)
    def _truncate(t):
        ids = tk.encode(t)
        return tk.decode(ids[:cap]) if len(ids) > cap else t
    texts = [_truncate(t) for t in texts]

    t0 = time.perf_counter()
    outs = embed_fn(texts)
    log.info(f"embedded {n} rows in {time.perf_counter()-t0:.1f}s")

    def vec(o):
        e = o.outputs
        e = getattr(e, "embedding", None) or getattr(e, "data", e)
        return list(e)
    ds = ds.add_column(args.output_column, [vec(o) for o in outs])
    dim = len(ds[0][args.output_column])

    card = DatasetCard(
        f"# {args.output_dataset}\n\nEmbeddings of `{args.input_dataset}` column `{args.column}` "
        f"with [`{args.model}`](https://huggingface.co/{args.model}) (dim {dim}, vLLM pooling).\n\n"
        f"Produced on Hugging Face Jobs with `uv-scripts/embeddings/generate-embeddings-vllm.py`.\n")
    # Retry the push with an XET-disable fallback — a transient failure would lose the paid run.
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                log.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            ds.push_to_hub(args.output_dataset, private=args.private)
            break
        except Exception as e:
            log.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                log.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                log.error("All upload attempts failed. Results are lost.")
                raise SystemExit(1)
    try:
        card.push_to_hub(args.output_dataset, repo_type="dataset")
    except Exception as e:
        log.warning(f"card push skipped: {e}")
    log.info(f"✅ https://huggingface.co/datasets/{args.output_dataset}")


if __name__ == "__main__":
    main()
