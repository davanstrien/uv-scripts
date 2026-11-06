---
name: maintain-recipes
description: "Internal, dev-only: keep this repo's recipes working on HF Jobs. Use to refresh active recipes (newer image, smoke test), freeze a superseded recipe (digest, exclude-newer, model revision), triage a recipe that fails on Jobs, check what an engine image ships, or run the monthly/quarterly maintenance pass. Reads tiers from each script's [tool.uv-scripts] header and records smoke tests in maintenance/verifications.jsonl. Proposes diffs; never merges."
---

# maintain-recipes

The rules are in [MAINTAINING.md](../../../MAINTAINING.md): two tiers (active, frozen), the header table,
the freeze steps, and why image-mode recipes break. Read it first. This skill is the procedure.

## Before you start

- `hf auth whoami` works, and the namespace has Jobs credit. Every smoke test is a paid Job.
- **Spend:** a 3-page smoke test on `a10g-small` or `l4x1` costs about $0.05 to $0.15. Ask the
  maintainer before a pass that would cost more than $10 in total. Run at most 8 Jobs at once.
- **Input:** `davanstrien/ufo-ColPali` (public, `image` column). Use `--max-samples 3 --shuffle`.
- **Output:** a private dataset named `<you>/uvs-smoke-<recipe>-<YYYYMMDD>`, with `--private`. Delete
  it after you have checked it (`hf repos delete <id> --repo-type dataset --yes`). Delete only the
  repos you created in this pass.
- Work on a branch in a separate `git worktree`. Never push to the Hub: GitHub is the source of truth.

## Find the work

```bash
python3 tools/recipe-status.py            # tier, image, cutoff and last smoke test per recipe
python3 tools/recipe-status.py --check    # rule breaks for recipes that have a tier
```

Rows marked `⚠️ due` are past their smoke-test interval (active 45 days, frozen 120 days).

## Smoke test (one recipe)

**OCR and other image-to-text recipes** (`INPUT_DATASET OUTPUT_DATASET`, an `image` column):

```bash
hf jobs uv run --detach --timeout 45m <folder>/<recipe>.py \
  davanstrien/ufo-ColPali <you>/uvs-smoke-<recipe>-<YYYYMMDD> --max-samples 3 --shuffle --private
```

The header supplies the image, flavor, Python, `PYTHONPATH` and secret. Use `--dry-run` first to see
the resolved Job.

**Every other recipe:** do not reuse the OCR command. Read the script's docstring and `add_argument`
calls first (named arguments, a text column, labels, mounts for bucket variants), and pick a small public
input of the right shape. Copy the docstring's example, cut it to a few rows, and check the output
column that recipe writes. A wrong command still starts a paid Job.

- **Job ID:** take it from the `hf jobs logs -f <namespace>/<id>` hint. Do not grep for any 24-hex
  string: image digests contain them too.
- **Wait** with `hf jobs wait <namespace>/<id>`, then `hf jobs inspect` and `hf jobs logs`.
- **Stuck in SCHEDULING for more than 10 minutes:** cancel it and relaunch on another flavor that fits
  (`--flavor` before the script overrides the header).

**A pass needs all of these**, not only a completed Job:

1. The Job stage is `COMPLETED` and the output dataset has the expected number of rows.
2. Read the output column. No error sentinels (`[OCR ERROR]` and similar), no repetition loops,
   and no empty rows for pages that have text. A blank page can give empty output for every model; check
   the image before you call that a failure.
3. Note the model revision and the image as run (the log prints both for frozen recipes).

Then append one record to `maintenance/verifications.jsonl` (fields in MAINTAINING.md), and run
`python3 tools/recipe-status.py --write`.

## Refresh an active recipe

1. Smoke-test the current header. If it fails, triage it (below) before you change the image.
2. Check what the newer image ships (next section). Compare with the recipe's dependencies: the
   image's copy of a shared package wins.
3. Change the image tag, smoke-test again, and compare the output with step 1 on the same pages.
4. Record both tests. Propose the diff in a PR; the maintainer merges.

## Freeze a recipe

Follow the steps in MAINTAINING.md ("Freeze a recipe"). Then:

- Check the cutoff resolves what you expect: `UV_NO_CONFIG=1 uv tree --script <recipe>.py --depth 1`
  (copy the script to a scratch folder first; this needs no GPU).
- Run `hf jobs uv run --dry-run <recipe>.py in out` and confirm the image shows the digest.
- Smoke-test, record, and run `python3 tools/recipe-status.py --check`.

## Check what an image ships

A CPU Job, no GPU needed (it pulls the full image, so scheduling can take a few minutes):

```bash
hf jobs run --detach --flavor cpu-basic --timeout 30m <image> python3 -c '
import importlib.metadata as m
for p in ["huggingface-hub","transformers","tokenizers","datasets","pyarrow","numpy","pillow","fsspec","torch","vllm","sglang"]:
    try: print(p, m.version(p))
    except m.PackageNotFoundError: print(p, "-")'
```

Then compare with what the recipe's dependencies need, for example
`curl -s https://pypi.org/pypi/datasets/<version>/json | jq '.info.requires_dist'`.

## Triage a failure

Match the first error in the log:

| Error | Likely cause | Fix |
|---|---|---|
| `ImportError` / `cannot import name ...` from a package the image also ships | shadowing: uv picked a version that needs a newer copy than the image's | cap the dependency with the image limit as the comment (MAINTAINING.md); for frozen, check the cutoff |
| `This modeling file requires the following packages ...` | the model repo changed its remote code (`trust_remote_code`) | `HfApi().list_repo_commits(model)`; pin the last working revision, or add the packages |
| `KeyError: 'accessToken'` in `login()` | old `huggingface_hub` on the image vs the current Jobs token | set `os.environ["HF_TOKEN"]` instead of calling `login()` |
| `ImportError` from `vllm` itself | vLLM API drift (see vLLM facts) | update the call, or keep the old image |
| `nvcc` / FlashInfer at engine start | the CUDA toolkit is missing | run on the `vllm/vllm-openai` image |
| `found version 12090` / driver too old | CUDA-13 build on an older driver (`a100-large`) | use a CUDA-12 image, or test on `l4x1` |
| `No module named 'vllm'` | wrong `PYTHONPATH` for this image (`site-packages` vs `dist-packages`) | match the image's Python path (see the image table) |
| Completed, but output empty or looping | prompt, resize or token limits, or a bad page | read 3 rows and the images; compare with the last passing record |

If the cause is outside the repo (Hub, Jobs, an upstream model), record a `fail` with the reason in
`notes` and tell the maintainer. Do not guess a fix.

## vLLM facts (verified 2026-06)

- **Structured outputs:** use `StructuredOutputsParams` (`structured_outputs=`), floor `vllm>=0.11.0`.
  `GuidedDecodingParams` / `guided_decoding=` was removed in 0.12.0 (ImportError on 0.12.0 and later).
- **Pin the image tag**, not `:latest`: it fixes vLLM, torch, the CUDA toolkit and FlashInfer together.
  Keep a `vllm==` or `>=` floor in the docstring for the local path.
- **Capture, then pin.** Pin to a tested version, never a guessed one. vLLM logs its version at engine
  start; read it from the passing run.

## Output

A short report per recipe: tier, what was tested, pass or fail with the Job link, and the proposed
diff. Change files only on a branch, and open a PR only when the maintainer asks.
