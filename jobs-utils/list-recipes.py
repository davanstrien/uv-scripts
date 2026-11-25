# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub",
# ]
# ///
"""
List every recipe (UV script) in the `uv-scripts` Hugging Face org.

The zero-setup way to see what's available: this script asks the Hub for every
dataset repo under the org, finds the `.py` scripts in each, and prints a runnable
URL for every one. No GPU, no token, no account needed — it only reads public repos.

Run it locally (nothing to install but uv itself):

    uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/list-recipes.py

Add --describe to also print the first line of each script's docstring (slower —
it fetches each file):

    uv run https://huggingface.co/.../list-recipes.py --describe
"""

import argparse
import sys

from huggingface_hub import HfApi

ORG = "uv-scripts"


def first_docstring_line(api: HfApi, repo_id: str, filename: str) -> str:
    """Best-effort: read a script's module docstring's first non-empty line."""
    try:
        path = api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return ""
    # Find the first triple-quoted string and return its first real line.
    for quote in ('"""', "'''"):
        if quote in text:
            body = text.split(quote, 2)
            if len(body) >= 3:
                for line in body[1].splitlines():
                    line = line.strip()
                    if line:
                        return line
    return ""


def main(describe: bool = False) -> None:
    api = HfApi()
    repos = sorted(api.list_datasets(author=ORG), key=lambda d: d.id)

    total = 0
    for repo in repos:
        repo_id = repo.id
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset")
        except Exception as e:
            print(f"  (skipped {repo_id}: {e})", file=sys.stderr)
            continue
        scripts = sorted(f for f in files if f.endswith(".py") and "/" not in f)
        if not scripts:
            continue

        print(f"\n## {repo_id}   →   https://huggingface.co/datasets/{repo_id}")
        for script in scripts:
            url = f"https://huggingface.co/datasets/{repo_id}/raw/main/{script}"
            if describe:
                desc = first_docstring_line(api, repo_id, script)
                print(f"  {script:<28} {url}")
                if desc:
                    print(f"  {'':<28} {desc}")
            else:
                print(f"  {script:<28} {url}")
            total += 1

    print(f"\n{total} recipes across {ORG}. Read a recipe before you run it:  curl -fsSL <url>")
    print("Its docstring has an example command and its argument definitions list every flag.")
    print("Run it with:  uv run <url>  (or  hf jobs uv run <url>  for a GPU)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"List all UV-script recipes in the {ORG} org.")
    parser.add_argument(
        "--describe", action="store_true",
        help="Also print each script's docstring summary (slower — fetches each file)",
    )
    args = parser.parse_args()
    main(describe=args.describe)
