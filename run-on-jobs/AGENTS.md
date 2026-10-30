# Run a model on Hugging Face Jobs

Each recipe is one script, run from its URL with `hf jobs uv run`. There is nothing to clone or install except the `hf` CLI.

## Set up

- Check `hf version` (needs 1.32+). If it's missing: `curl -LsSf https://hf.co/cli/install.sh | bash`, then `hf auth login`.
- Jobs need pay-as-you-go credit, and each job costs money. Confirm with the user before starting one.
- The CLI is the current reference: `hf --help`, `hf jobs uv run --help`, `hf jobs hardware`. To install the Hugging Face CLI skill, run `hf skills add`.

## Find a recipe

- `hf download uv-scripts/run-on-jobs recipes.json --repo-type space`, then look for the model in `recipes[].models`.
- `kind`: `run` applies a model to data; `train` fine-tunes one.
- `source`: `header` carries its launch settings; `documented` is an example from the script's docs; `flags-needed` has no command, so read the script first.
- Read a script's docstring and options: `curl -s <url> | head -100`. Running it locally may fail on GPU-only dependencies.
- Each task's README explains its recipes, for example https://huggingface.co/datasets/uv-scripts/ocr. OCR also has `models.json`, with language coverage and support levels.
- Fuller guides: [uv-recipes](https://github.com/davanstrien/uv-scripts/tree/main/skills/uv-recipes) and, for few-shot fine-tuning, [setfit-on-jobs](https://github.com/davanstrien/uv-scripts/tree/main/skills/setfit-on-jobs).

## Run it

Start from the recipe's `command` field. Most recipes go from dataset to dataset:

    hf jobs uv run <url> <input-dataset> <output-dataset> --max-samples 10

Recipes that read files use mounts. A local folder also works as a source:

    hf jobs uv run -v hf://buckets/<user>/<input>:/input:ro -v hf://buckets/<user>/<output>:/output <url> /input /output

- Preview for free with `--dry-run`, and check the flavor: without a header, a GPU recipe can resolve to `cpu-basic`.
- Flags before the URL configure the job and override the header, such as `--flavor` and `--timeout`. Flags after it go to the script. If a script flag has the same name as a job flag, put `--` before the URL.
- Start with a small `--max-samples` if the script has it, and add `--private` if the output shouldn't be public.

## Check it

- Use `--detach` for long runs, then `hf jobs logs -f <id>`, `hf jobs wait <id>` or `hf jobs inspect <id>`. `hf jobs cancel <id>` stops one.
- Look at the output rows. A completed job doesn't prove they're correct.
