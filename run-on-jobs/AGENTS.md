# Run a model on Hugging Face Jobs

`recipes.json` lists one-command batch and fine-tuning recipes.

## Set up

Check for the `hf` CLI (1.32 or newer) with `hf version`. If it's missing, install it, then ask the user to sign in:

    curl -LsSf https://hf.co/cli/install.sh | bash
    hf auth login

Jobs need pay-as-you-go credit on the user's account, and each job costs money, so confirm with the user before starting one.

Learn the commands from the CLI, because it stays current:

    hf --help
    hf jobs uv run --help
    hf jobs hardware

For a persistent reference, install the `hf-cli` skill with `hf skills add`. The installer above adds it by default.

## Find a recipe

    hf download uv-scripts/run-on-jobs recipes.json --repo-type space

Find the model in `recipes[].models`.

Each recipe has:

- `kind`: `run` applies a model to data; `train` fine-tunes one.
- `command`: the command to start from. `INPUT` and `OUTPUT` are usually dataset IDs; read the script to confirm.
- `source`: how far to trust the command.
  - `header`: the script carries its launch settings.
  - `documented`: an example from the script's docs. It may set its own hardware flags.
  - `flags-needed`: `command` is null. Read the script and choose the options yourself.

## Run it

1. Read the options: open the recipe's `url`.
2. Preview without spending: `hf jobs uv run --dry-run <url> <args>`. Check the image and flavor; without a header a GPU recipe can resolve to `cpu-basic`.
3. Start small: pass `--max-samples` if the script has it.
4. Monitor: `hf jobs ps`, `hf jobs logs <id>`.
5. Check the output. A completed job is not proof that its rows are correct.
