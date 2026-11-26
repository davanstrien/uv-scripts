# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "sentence-transformers>=5.0.0",
#     "torch",
#     "numpy",
#     "einops",
#     "pyarrow",
#     "pylance",
#     "huggingface-hub",
# ]
# ///
"""
Embed a Hugging Face dataset and push it back as a Lance vector index — a Hub dataset that
IS a searchable vector database. Anyone you share it with can vector-search it over `hf://`
without downloading it:

    import lance
    ds = lance.dataset("hf://datasets/your-name/my-vecdb/vecdb.lance")   # opens fast, no download
    hits = ds.to_table(nearest={"column": "vector", "q": query_vector, "k": 5})

Best for share-and-search over a corpus; for high-QPS serving, pull the dataset local first.

PROMPTS: documents are embedded with the model's known DOCUMENT convention (e5 → "passage: ",
nomic → "search_document: "; bge-en/bge-m3 → none). At SEARCH time, embed your query with the
matching QUERY prefix (printed at the end of the run) or retrieval quality silently drops.
Override the document prefix with --prompt '<prefix>' (or --prompt '' for none).

    hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/embed-to-lance.py \\
        stanfordnlp/imdb your-name/imdb-vecdb --column text --model BAAI/bge-base-en-v1.5 --private
"""
import argparse
import logging
import os
import re
import shutil
import sys
import time
import numpy as np
import pyarrow as pa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed-to-lance")


def known_convention(model_id):
    """(query_prefix, doc_prefix) for common families (documented in model cards, not registered
    in sentence-transformers config). Same table as generate-embeddings.py; None = unknown."""
    m = model_id.lower()
    if "instruct" in m:
        return None
    if "nomic-embed-text" in m:
        return ("search_query: ", "search_document: ")
    if "bge-m3" in m:
        return ("", "")
    if re.search(r"(^|[/_-])e5([_-]|$)", m):
        return ("query: ", "passage: ")
    if "bge" in m and "-en" in m:
        return ("Represent this sentence for searching relevant passages: ", "")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dataset", help="Input dataset repo id")
    ap.add_argument("output_repo", help="Output dataset repo id; the index is written to vecdb.lance in it (created if missing)")
    ap.add_argument("--column", default="text", help="Text column to embed (default: text)")
    ap.add_argument("--config", default=None, help="dataset config name (e.g. wikipedia needs one)")
    ap.add_argument("--split", default="train", help="Dataset split to read (default: train)")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5",
                    help="Sentence-transformers model repo id (default: BAAI/bge-base-en-v1.5)")
    ap.add_argument("--max-samples", type=int, default=None, help="Embed only the first N rows (for testing)")
    ap.add_argument("--batch-size", type=int, default=64, help="Encoding batch size (default: 64)")
    ap.add_argument("--max-seq-len", type=int, default=512,
                    help="Truncate texts to this many tokens; the model's own limit wins if lower (default: 512)")
    ap.add_argument("--prompt", default=None,
                    help="Document prefix to prepend (default: auto from the known-family table; "
                         "pass '' to force none)")
    ap.add_argument("--private", action="store_true",
                    help="Create the output repo as private (no effect on a repo that already exists)")
    args = ap.parse_args()

    import torch
    import lance
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
    from sentence_transformers import SentenceTransformer

    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    t_all = time.perf_counter()
    ds = load_dataset(args.input_dataset, args.config, split=args.split) if args.config \
        else load_dataset(args.input_dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    texts = [t if isinstance(t, str) and t.strip() else " " for t in ds[args.column]]
    n = len(texts)

    t_load = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device, trust_remote_code=True)
    if getattr(model, "max_seq_length", None):
        model.max_seq_length = min(model.max_seq_length, args.max_seq_len)
    dim = model.get_sentence_embedding_dimension()

    # Document-side prompt: explicit --prompt wins (incl. '' for none), else the known-family
    # table; else None → encode_document() natively selects any REGISTERED document prompt
    # (and routes Router models by task).
    registered = {k: v for k, v in (getattr(model, "prompts", {}) or {}).items() if v}
    kc = known_convention(args.model)
    doc_prompt = args.prompt if args.prompt is not None else (kc[1] if kc else None)
    query_prompt = kc[0] if kc else registered.get("query", "")
    log.info(f"document prompt: {doc_prompt!r}" if doc_prompt
             else ("document prompt: native (registered)" if registered.get("document")
                   else "document prompt: (none)"))

    t0 = time.perf_counter()
    encode_kwargs = {"prompt": doc_prompt} if doc_prompt is not None else {}
    emb = model.encode_document(texts, batch_size=args.batch_size, show_progress_bar=True,
                                convert_to_numpy=True, normalize_embeddings=True,
                                **encode_kwargs).astype(np.float32)
    log.info(f"embedded {n} rows in {time.perf_counter()-t0:.1f}s, dim={dim}")

    tbl = pa.table({
        "id": pa.array(range(n), pa.int64()),
        "text": pa.array([t[:2000] for t in texts]),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1), pa.float32()), dim),
    })
    local = "vecdb.lance"
    if os.path.exists(local):
        shutil.rmtree(local)
    lds = lance.write_dataset(tbl, local, mode="overwrite")
    try:
        parts = max(1, min(256, int(np.sqrt(n))))
        lds.create_index("vector", index_type="IVF_PQ", num_partitions=parts,
                         num_sub_vectors=max(1, dim // 16))
        log.info(f"built IVF_PQ index (partitions={parts})")
    except Exception as e:
        log.warning(f"index build skipped ({repr(e)[:120]}); flat search still works over hf://")

    # Retry the upload with an XET-disable fallback — a transient failure here would lose the
    # whole (paid) embedding run.
    api = HfApi()
    api.create_repo(args.output_repo, repo_type="dataset", private=args.private, exist_ok=True)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                log.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            api.upload_folder(folder_path=local, path_in_repo="vecdb.lance",
                              repo_id=args.output_repo, repo_type="dataset")
            break
        except Exception as e:
            log.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                log.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                log.error("All upload attempts failed. Results are lost.")
                sys.exit(1)
    total_s = time.perf_counter() - t_all
    import json as _json
    log.info("ROUNDTRIP " + _json.dumps({
        "input": args.input_dataset, "n": n, "dim": dim, "model": args.model,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "batch_size": args.batch_size, "load_s": round(t_load - t_all, 1),
        "total_roundtrip_s": round(total_s, 1), "rows_per_s_end_to_end": round(n / total_s, 1),
        "hf_path": f"hf://datasets/{args.output_repo}/vecdb.lance"}))
    log.info(f"✅ {n} rows → searchable vector DB in {total_s/60:.1f} min "
             f"(load→embed→index→push). hf://datasets/{args.output_repo}/vecdb.lance")
    if query_prompt or registered.get("query"):
        log.info("⚠️ At search time, embed queries with the QUERY convention — mismatched prompts "
                 "degrade retrieval. Easiest: model.encode_query([your_query])"
                 + (f", or explicitly: model.encode([{query_prompt!r} + your_query])" if query_prompt else "."))


if __name__ == "__main__":
    main()
