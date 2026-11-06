# Maintaining recipes

This file is for maintainers and their agents. It covers how much effort each recipe gets, how to
freeze one, and why recipes on pinned images break. For the Hub mirror and the never-delete rule, read
[AGENTS.md](AGENTS.md). For running recipes, read each folder's `README.md` (and `ocr/AGENTS.md`).

The procedures live in the [`maintain-recipes`](.claude/skills/maintain-recipes/SKILL.md) skill.

## Two maintenance tiers

Every recipe with a `[tool.hf-jobs]` header is either **active** or **frozen**. The tier says how much
maintenance effort the recipe gets. It does not say whether the recipe works today (see the smoke-test
records below and, for OCR, `support` in `ocr/models.json`), and it does not promise a reproducible run.

|  | Active | Frozen |
|---|---|---|
| Which recipes | the newest model in a family, tools, recipes people use a lot | superseded or little-used models |
| Image | newest tested tag | pinned by digest, with the tag in a comment |
| Packages | resolve fresh on each run, within the recipe's floors and caps | `[tool.uv] exclude-newer` at a date when the resolution worked |
| Model | main branch; pin a revision if upstream changes the repo in place | revision pinned (model, code and tokenizer) |
| Smoke test | monthly, and after a vLLM release the recipe could use | quarterly, and after a Jobs or Hub incident |
| Changes | upgrades and features | fixes only when it breaks |

Effort goes to active recipes. A frozen recipe should keep working for a long time without anyone
looking at it. It can still break from things no pin controls: Hub API and auth changes, the Jobs
runtime, the `hf` CLI, GPU drivers and flavors, files removed upstream, and packages a script installs
at run time.

### The header table

The tier lives in the recipe's own PEP 723 header, so it travels with the script:

```toml
# [tool.uv-scripts]
# maintenance = "frozen"             # "active" or "frozen"
# frozen_on = "2026-11-06"           # frozen only
# superseded_by = "lighton-ocr3.py"  # optional; a script in the same folder
```

uv and the `hf` CLI ignore this table. Keep it out of `[tool.hf-jobs]`. Recipes without the table have
no tier yet.

### Freeze a recipe

Freeze a recipe when a newer model replaces it, or when it gets little use. The PR that adds a
successor freezes the recipe it replaces.

1. **Choose the cutoff.** Use a date when this recipe's packages resolved to a working set. This is not
   always today: if a package release broke the recipe, the cutoff must be before that release. Check
   what a date gives with `UV_NO_CONFIG=1 uv tree --script <recipe>.py --depth 1`.
2. **Pin the image by digest.** Get it with
   `curl -s https://hub.docker.com/v2/repositories/<namespace>/<repo>/tags/<tag> | jq -r .digest`
   and write `image = "<namespace>/<repo>@sha256:<digest>"  # <tag>`.
3. **Pin the model.** Pass `revision`, `code_revision` and `tokenizer_revision` (vLLM `LLM(...)`), or
   `revision` (`from_pretrained`), for the default model. Pin any other download the same way.
4. **Add the header table** and a docstring line that names the successor.
5. **Smoke-test it** with the skill, and append a record to `maintenance/verifications.jsonl`.
6. For OCR, update `support_note` in `ocr/models.json` to name the successor.

[`ocr/lighton-ocr2.py`](ocr/lighton-ocr2.py) is the worked example.

### When a frozen recipe breaks

- Make the smallest fix that keeps it frozen, smoke-test it, and record the result.
- If the fix is not worth it, mark it `unsupported` (for OCR, in `ocr/models.json`) and say why.
- Removing a script deletes it from the Hub. Do that only as a deliberate, confirmed step (see
  [AGENTS.md](AGENTS.md)), never as a side effect of a cleanup.
- To make a frozen recipe active again: set `maintenance = "active"`, remove `frozen_on`,
  `superseded_by` and `exclude-newer`, move it to a current image tag, smoke-test, and record the result.

## Image-mode recipes break when PyPI moves

Most GPU recipes run on a pinned engine image, for example `vllm/vllm-openai:v0.29.0`:

```toml
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.29.0"
# python = "/usr/bin/python3"
# env = { PYTHONPATH = "/usr/local/lib/python3.12/dist-packages" }
```

The Job runs `uv run --python /usr/bin/python3`. uv builds a fresh environment with the header's
dependencies, and `PYTHONPATH` adds the image's packages **in front of it**. When both have a package,
the image's copy wins, but uv resolved the environment as if its own copy would be used. So uv can pick
a version of one package that needs a newer copy of something the image ships.

Example, 2026-11-06: `datasets` 5.1.0 (released 2026-10-05) needs `huggingface_hub>=1.31`. The vLLM
images v0.20.1 to v0.29.0 ship hub 1.13 to 1.30. Every recipe on those images that depends on
`datasets` failed at import with `cannot import name 'httpx' from 'huggingface_hub.utils'`.

Rules:

- Before you add a dependency or move a recipe to another image, check what the image ships (the skill
  has a one-line CPU Job for this).
- A cap that encodes an image limit is correct. Write the reason next to it, for example
  `"datasets>=4.0.0,<5.1",  # 5.1 needs huggingface_hub>=1.31; the image ships an older 1.x ...`.
  This is not the stale-cap problem that `ocr/CLAUDE.md` warns about.
- A floor such as `huggingface-hub>=1.31` in the header does not help: the image's copy still wins.

Packages in each image, checked 2026-11-06 (none of these images ships `datasets` or `pyarrow`, except
the SGLang one):

| Image | huggingface_hub | transformers | fsspec | Python |
|---|---|---|---|---|
| `vllm/vllm-openai:v0.20.1` | 1.13.0 | 5.7.0 | 2026.4.0 | 3.12.13 (`/usr/local/bin`) |
| `vllm/vllm-openai:v0.22.1` | 1.17.0 | 5.10.2 | 2026.4.0 | 3.12.13 |
| `vllm/vllm-openai:v0.24.0` | 1.21.0 | 5.12.1 | 2026.6.0 | 3.12.13 |
| `vllm/vllm-openai:unlimited-ocr` | 1.21.0 | 5.12.1 | 2026.6.0 | 3.12.13 |
| `vllm/vllm-openai:v0.29.0` | 1.30.0 | 5.16.1 | 2026.7.0 | 3.12.3 |
| `vllm/vllm-openai:v0.30.0` | 1.32.0 | 5.17.0 | 2026.9.0 | 3.12.3 |
| `vllm/vllm-openai:v0.32.0` | 2.2.0 | 5.19.0 | 2026.9.0 | 3.12.3 |
| `lmsysorg/sglang:v0.5.20-cu130` | 1.32.0 | 5.12.1 | 2026.6.0 | 3.12.3 (also `datasets` 5.0.1, `pyarrow` 25.0.1) |

## Smoke-test records and the status table

`maintenance/verifications.jsonl` is append-only, with one JSON object per smoke test:

```json
{"date": "2026-11-06", "script": "ocr/lighton-ocr2.py", "commit": "<git sha of the tested script>",
 "image": "<image as run>", "exclude_newer": "2026-10-01T00:00:00Z", "model_revision": "<sha>",
 "flavor": "a10g-small", "input": "<dataset>", "rows": 3, "outcome": "pass",
 "job_url": "https://huggingface.co/jobs/<namespace>/<id>", "notes": "<what was checked>"}
```

`outcome` is `pass` or `fail`. A pass means the Job completed, the output has the expected rows, and
the text was read: no error sentinels, no empty pages that have text, no repetition loops.

`maintenance/STATUS.md` is generated from the headers and the records:

```bash
python3 tools/recipe-status.py --write   # regenerate maintenance/STATUS.md
python3 tools/recipe-status.py --check   # rule breaks for recipes that have a tier
```

`--check` also fails on a header that does not parse and on a malformed record. For a frozen recipe,
the latest record must be a pass on or after `frozen_on`, with the header's image and cutoff and a
40-character model revision. The tool cannot see the revision in the script's code: check that by
hand when you freeze a recipe or change its model loading.

`maintenance/` is not mirrored to the Hub.

## Skills

- [`maintain-recipes`](.claude/skills/maintain-recipes/SKILL.md): refresh active recipes, freeze
  recipes, triage failures, run smoke tests.
- [`refresh-uv-recipes`](.claude/skills/refresh-uv-recipes/SKILL.md): check the shippable
  `skills/uv-recipes` skill against the live Hub and CLI.
