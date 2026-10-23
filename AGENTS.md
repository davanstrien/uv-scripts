# AGENTS.md — operating this repo

This is a GitHub-authoritative cookbook of **UV scripts you run on Hugging Face Jobs in one command**
(`hf jobs uv run <url>`). Each top-level folder **mirrors to a Hugging Face dataset repo** under
[`uv-scripts`](https://huggingface.co/uv-scripts) via the [`huggingface/hub-sync`](https://github.com/huggingface/hub-sync)
Action on push to `main`. GitHub is the source of truth; the Hub repos are mirrors. (Claude Code: this
file is the runbook — `CLAUDE.md` only points here.)

## ⛔ The one rule: never delete a file from the Hub

The sync runs `hf upload <repo> <folder> --delete="*"` on every push. **Any file on the Hub repo that is
absent from its GitHub folder is permanently deleted.** Therefore:

- A folder gets a sync workflow **only after** it is a *superset* of that repo's current Hub contents.
- **Seed from the Hub, never from old local clones:**
  `hf download uv-scripts/<repo> --repo-type dataset --local-dir <repo>/ && rm -rf <repo>/.cache <repo>/.gitattributes`.
  Dropping `.gitattributes` is mandatory: it carries `*.gif`/`*.png`/`*.pdf filter=lfs` rules, and with
  git-lfs installed `git add` would turn binaries into pointers that the workflow (no `lfs: true` checkout)
  uploads as text — **corrupting the Hub asset**. The action excludes `.gitattributes` anyway, so removing
  it is lossless.
- The formerly-private `embeddings` repo was **adopted as a public recipe family** (2026-07, PR #73):
  `embeddings/` in the tree maps to `uv-scripts/embeddings` like any other folder. Its pre-adoption
  content was one stale script, overwritten by the seed sync. (The old rule — never pull that private
  repo's content into the tree — is moot now; nothing was ever imported from it.)
- Edits to an existing repo's folder are **additive** — add files freely; do not remove anything that's on
  the Hub. (Retiring a Hub file is a separate, deliberate, confirmed action — never a seeding side effect.)
- The superset gate runs **both locally** (`tools/verify-superset.sh <folder> uv-scripts/<repo>` — respect a
  BLOCK) **and inside every sync workflow** as a hard pre-step, so it guards re-triggers too, not just first
  flips.

## Layout — one folder per Hub repo, named identically

`ocr/ → uv-scripts/ocr`, `sam3/ → uv-scripts/sam3`, … `subdirectory == folder == repo name`, so the mapping
can't be mis-typed. Domain grouping for readers lives in the root `README.md`, not in folder nesting. Don't
mirror the org's *demo* Spaces. (One Space *is*
mirrored — the org profile card: `org-card/` → `uv-scripts/README`, a static Space, via the reusable
workflow's optional `repo_id` / `repo_type` inputs.)

## Script conventions

- Self-contained **[PEP 723](https://peps.python.org/pep-0723/) / [UV script](https://docs.astral.sh/uv/guides/scripts/)** — deps inline, no `requirements.txt`, no shared importable lib:
  ```python
  # /// script
  # requires-python = ">=3.10"
  # dependencies = ["datasets", "transformers", "torch", "huggingface-hub[hf_transfer]"]
  # ///
  ```
- Each repo folder has a `README.md` with frontmatter `viewer: false` + `tags: [uv-script, …]` (these are
  dataset repos hosting code).
- Every script's docstring shows a runnable **`hf jobs uv run`** example. README examples use the **HF raw
  URL** `https://huggingface.co/datasets/uv-scripts/<repo>/raw/main/<script>.py` (the trackable invocation).
- **Provenance stamp (canonical — copy this, don't invent variants).** Every output dataset card a recipe
  pushes ends with a `## Reproduction` section in exactly this shape (inline it in the card f-string;
  no shared helper):

  ```python
  on_jobs = os.environ.get("JOB_ID") is not None          # set by HF Jobs in-container
  hw = jobs_flavor()                                      # e.g. "a10g-small"; "" on CPU
  origin = (
      f"Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
      + (f" (`{hw}`)" if hw else "")
  ) if on_jobs else "Generated"
  ```

  ```markdown
  ## Reproduction

  {origin} with the [`<script>.py`](<script-raw-url>) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

  ```bash
  hf jobs uv run <script-raw-url> <args>
  ```
  ```

  **Do not stamp the raw `ACCELERATOR` value.** On Jobs it holds only `cpu` / `gpu` (verified on t4-small,
  a10g-small, l4x1, a10g-largex2 and l4x4, 2026-09/10), not the flavor the hub docs describe, so a raw stamp
  prints "(`gpu`)" and a reproduce command built from it says `--flavor gpu`. Copy the `jobs_flavor()`
  helper from `ocr/glm-ocr.py`: it looks the flavor up in the public Jobs hardware list
  (`HfApi().list_jobs_hardware()`, no token needed) by GPU model and count, then by the `CPU_CORES` /
  `MEMORY` values Jobs sets, so it also tells `a10g-small` from `a10g-large` and names CPU flavors. It
  returns "" on any failure. Use `jobs_flavor() or "<default>"` for the reproduce command's `--flavor`.

  Rules: the reproduce command is **`hf jobs uv run`** (never bare `uv run` — the card is the Jobs
  advertisement); the "Produced on HF Jobs" claim is **gated on `JOB_ID`** (a local run must not claim it);
  credit goes to the **org**, never a personal name; no badges. Card frontmatter tags: always `uv-script`,
  plus **`hf-jobs` when `JOB_ID` is set** — these are the adoption meters
  (`list_datasets(tags="uv-script")` / `tags="hf-jobs"` count stamped public outputs).
- GPU scripts check `torch.cuda.is_available()` and exit with a clear message. Write outputs to the Hub
  (`push_to_hub`) or a bucket (`-v hf://…`), never bare local paths (Jobs disk is ephemeral; local *input*
  dirs can be mounted with `-v ./dir:/mnt`, which syncs them to a bucket). Name by task, not tool.

## The mirror — how it works

- `.github/workflows/sync-<repo>.yml`, one **thin caller** per folder (triggered on `push` to `main` filtered
  to `['<repo>/**', '.github/workflows/sync-<repo>.yml']`), which calls the shared reusable workflow
  `.github/workflows/_sync-folder.yml` — the single place the gate + LFS guard + `hub-sync` logic lives, so an
  action-version bump or gate change is a one-file edit.
- It uploads file **bytes** over HTTP, creates the repo if missing (`--exist-ok`), makes **one squashed
  commit**, and **excludes `.git*` and `.github*`** (so `.gitattributes` is not synced). No git history carried.
- **`repo_type: dataset` is required** (the action defaults to `space`). Pin `huggingface/hub-sync@v0.1.0`.
- **Commit binaries as plain blobs — do NOT use Git LFS here.** Files are well under GitHub's 100 MB limit; HF
  stores large files as LFS/xet automatically on upload. Because every binary repo's `.gitattributes` has
  `*.gif`/`*.png`/`*.pdf filter=lfs`, **delete `.gitattributes` on seed** (above) so `git add` can't make a
  pointer; the workflow's LFS-pointer guard is the backstop. (If Git LFS is ever deliberately added, the
  checkout MUST use `lfs: true`, or the action uploads pointer text and corrupts the Hub file.)

## Add a repo to the mirror (checklist)

1. `uv-scripts/<repo>` exists? **Yes** → `hf download` it into `<repo>/` (+ `rm -rf <repo>/.cache
   <repo>/.gitattributes`), then `git add <repo>/` and run `tools/verify-superset.sh <repo> uv-scripts/<repo>`
   (must pass). **No** → create `<repo>/` with scripts + a `viewer:false` README; the action births it.
2. Add `.github/workflows/sync-<repo>.yml` — a 13-line caller from the template below (just set the workflow
   `name`, the `paths` filter, and `folder: <repo>`). All sync logic is inherited from `_sync-folder.yml`.
3. Make sure `HF_TOKEN` (repo secret, fine-grained Write on the `uv-scripts` org) covers `<repo>`.
4. Commit + push to `main`. Only this repo syncs. Then confirm on the Hub: file count ≥ before, named assets present.
5. Add the recipe to the root `README.md` under its domain group.

## Token

GitHub repo secret **`HF_TOKEN`** = a fine-grained HF token with **Write** on the `uv-scripts` org (Write covers
repo creation; it does NOT grant whole-repo deletion). Re-create/rotate rather than widening scope.

## Workflow template

Each folder gets a thin **caller**; the gate + LFS guard + `hub-sync` logic lives once in the reusable
`.github/workflows/_sync-folder.yml` (so an action-version bump or a gate change is a single edit, not one per
repo). To add a repo, copy this caller and change the two `<repo>` occurrences + `folder:`:

```yaml
# .github/workflows/sync-<repo>.yml
name: Sync <repo> → HF
on:
  push:
    branches: [main]
    paths: ['<repo>/**', '.github/workflows/sync-<repo>.yml']
jobs:
  sync:
    uses: ./.github/workflows/_sync-folder.yml
    with:
      folder: <repo>
    secrets:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
```
