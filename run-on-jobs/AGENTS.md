# Run a model on Hugging Face Jobs

- **Set up:** `hf version` (needs 1.32+). If missing: `curl -LsSf https://hf.co/cli/install.sh | bash`, then `hf auth login`. Jobs need pay-as-you-go credit.
- **Learn the CLI:** `hf --help`, `hf jobs uv run --help`, `hf jobs hardware`. Install the skill with `hf skills add`.
- **Find a recipe:** `hf download uv-scripts/run-on-jobs recipes.json --repo-type space`, then look for the model in `recipes[].models`.
- **Trust the command by `source`:** `header` carries its launch settings; `documented` is an example from the script's docs; `flags-needed` has no command, so read the script.
- **Preview for free:** `hf jobs uv run --dry-run <url> <args>`. Check the flavor: without a header, a GPU recipe can resolve to `cpu-basic`.
- **Spend carefully:** confirm with the user first, and start small with `--max-samples` if the script has it.
- **Check the result:** `hf jobs ps`, `hf jobs logs <id>`, then look at the output rows. A completed job doesn't prove they're correct.
