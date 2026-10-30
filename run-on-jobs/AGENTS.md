# Run a model on Hugging Face Jobs

`recipes.json` lists one-command batch recipes. Find the model in `recipes[].models`.

    hf download uv-scripts/run-on-jobs recipes.json --repo-type space

Each recipe has:

- `kind`: `run` applies a model to data; `train` fine-tunes one.
- `command`: the command to start from. `INPUT` and `OUTPUT` are usually dataset IDs; read the script to confirm.
- `source`: how far to trust the command.
  - `header`: the script carries its launch settings.
  - `documented`: an example from the script's docs. It may set its own hardware flags.
  - `flags-needed`: `command` is null. Read the script and choose the options yourself.

Before you start a job:

1. Read the options: open the recipe's `url`.
2. Preview without spending: `hf jobs uv run --dry-run <url> <args>`. Check the image and flavor; without a header a GPU recipe can resolve to `cpu-basic`.
3. Start small: pass `--max-samples` if the script has it.
4. Monitor: `hf jobs ps`, `hf jobs logs <id>`.
5. Check the output. A completed job is not proof that its rows are correct.
