#!/usr/bin/env python3
"""Generate the Run on Jobs Space from the recipes in this repo.

    python3 tools/build-catalog.py                   # write run-on-jobs/
    python3 tools/build-catalog.py --verify-models   # also check each declared model on the Hub

The sync workflow runs this on merge, so run-on-jobs/ is not committed.

Scripts are only parsed, never imported or run. Standard library only.
"""

import argparse
import ast
import html
import json
import re
import shutil
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "run-on-jobs"
HUB_BASE = "https://huggingface.co/datasets/uv-scripts"

TASKS = {
    "ocr": "OCR",
    "transcription": "Transcription",
    "sam3": "Segmentation",
    "vlm-object-detection": "Object detection",
    "video": "Video",
    "classification": "Classification",
    "gliner": "Entities",
    "embeddings": "Embeddings",
    "vllm": "Bulk inference",
}
TRAINING_TASK = "Fine-tuning"
TASK_ORDER = [*TASKS.values(), TRAINING_TASK]

DATA_TOOLS = {
    "object-detection": "Check and convert detection datasets",
    "deduplication": "Remove near-duplicate text",
    "build-atlas": "Build an embedding atlas",
    "iiif-tiles": "Make IIIF deep-zoom tiles",
    "data-processing": "Convert uploads to optimized Parquet",
}

# Recipes without a [tool.hf-jobs] header whose documented command passed a smoke Job with checked output.
# Only these and header recipes are declared on model pages.
SMOKE_TESTED_EXAMPLES = {
    "embeddings/generate-embeddings.py",
    "sam3/detect-objects.py",
    "sam3/segment-objects.py",
}

# Left out of the Space until they run again.
BROKEN = {
    "gliner/extract-entities.py",  # the default model's tokenizer needs tiktoken, which the script doesn't declare
}

# Listed on the page, but not declared on model pages until the plain command works for the model.
NOT_DECLARED = {
    "vllm/generate-responses.py",  # the default model needs --max-model-len on the header's a100-large
}

# Recipe metadata that names a family rather than a Hub repo.
NOT_HUB_MODELS = {"PaddlePaddle/PP-OCRv6"}

OCR_SUPPORT_LEVELS = {"core", "experimental"}
SKIP_SCRIPT = re.compile(r"(^_|^test_|conftest|setup|benchmark|judge|compare)", re.IGNORECASE)
ALTERNATIVE_SCRIPT = re.compile(r"saturate|live|server|fleet|fewshot", re.IGNORECASE)
PATH_ARGUMENT = re.compile(r"^(input|output)$|_dir$|_path$|_folder$")
MODEL_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.+-]+$")
PLACEHOLDER_OWNERS = {"org", "datasets", "models", "hf", "bucket", "buckets", "input", "output", "user", "your", "username", "path", "dir", "my-org", "uv-scripts"}

STATUS = {"header": "command ready", "documented": "example command", "flags-needed": "read the script first"}
ROW_STATUS = {"header": "command ready", "documented": "example command", "flags-needed": "no command yet"}


def read_header(text):
    match = re.search(r"(?m)^# /// script\n((?:#.*\n)*?)# ///", text)
    if not match:
        return {}
    lines = [line[2:] if line.startswith("# ") else line[1:] for line in match.group(1).splitlines()]
    try:
        return tomllib.loads("\n".join(lines)).get("tool", {}).get("hf-jobs", {})
    except tomllib.TOMLDecodeError:
        return {}


def read_arguments(tree):
    """Return module string constants and the add_argument calls, using only the syntax tree."""
    constants = {}
    for node in tree.body:
        is_single_assign = isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        if is_single_assign and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            constants[node.targets[0].id] = node.value.value

    arguments = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        names = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
        if not names:
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        default = keywords.get("default")
        if isinstance(default, ast.Constant):
            default = default.value
        elif isinstance(default, ast.Name):
            default = constants.get(default.id)
        else:
            default = None
        required = isinstance(keywords.get("required"), ast.Constant) and keywords["required"].value is True
        arguments.append({"name": names[0], "default": default, "positional": not names[0].startswith("-"), "required": required})
    return constants, arguments


def find_models(constants, arguments):
    candidates = []
    for argument in arguments:
        if isinstance(argument["default"], str) and "model" in argument["name"].lower() and MODEL_ID.match(argument["default"]):
            candidates.append(argument["default"])
    for name, value in constants.items():
        if "MODEL" in name.upper() and MODEL_ID.match(value):
            candidates.append(value)
    unique = dict.fromkeys(candidates)
    return [model for model in unique if model.split("/")[0].lower() not in PLACEHOLDER_OWNERS]


def first_documented_command(text, script_name, url):
    """Return the first documented example that launches this script, as (job_flags, script_arguments)."""
    for match in re.finditer(r"hf jobs uv run(?:[^\n]*\\\n)*[^\n]*", text):
        if script_name not in match.group(0):
            continue
        lines = [re.sub(r"^\s*(#\s*)?", "", line).rstrip(" \\") for line in match.group(0).split("\n")]
        command = " \\\n    ".join(line for line in lines if line)
        if url not in command:
            command = re.sub(r"(?<![\w/.-])" + re.escape(script_name), url, command)
        if url not in command:
            continue
        job_flags, script_arguments = command.split(url, 1)
        return job_flags, script_arguments
    return None


def format_command(job_flags, url, script_arguments):
    command = job_flags.replace("-s HF_TOKEN=HF_TOKEN", "-s HF_TOKEN").rstrip(" \\\n") + f" \\\n    {url}"
    script_arguments = script_arguments.strip(" \\\n")
    if script_arguments:
        command += f" \\\n    {script_arguments}"
    return normalize_placeholders(command)


def normalize_placeholders(command):
    command = re.sub(r"\b(your-name|yourname|my-username|username|user)(?=/)", "your-username", command)
    command = re.sub(r"(?<![-\w])input-dataset", "INPUT_DATASET", command)
    return re.sub(r"(?<![-\w])output-dataset", "OUTPUT_DATASET", command)


def header_command(url, arguments, documented):
    """Header recipes need no hardware flags, so keep only mounts from the docs, and the script's own arguments."""
    if documented:
        job_flags, script_arguments = documented
        mounts = re.findall(r"(?:-v|--volume)\s+\S+", job_flags)
        return format_command(" \\\n    ".join(["hf jobs uv run", *mounts]), url, script_arguments)

    mounts = []
    parts = []
    for argument in arguments:
        name = argument["name"].replace("-", "_")
        if argument["positional"] and PATH_ARGUMENT.search(name):
            mount_point = "/output" if "output" in name else "/input"
            access = "" if "output" in name else ":ro"
            mounts.append(f"-v hf://buckets/your-username/{mount_point[1:]}:{mount_point}{access}")
            parts.append(mount_point)
        elif argument["positional"]:
            parts.append(name.upper())
        elif argument["required"]:
            parts.append(f"{argument['name']} {name.lstrip('_').upper()}")
    return format_command(" \\\n    ".join(["hf jobs uv run", *mounts]), url, " ".join(parts))


def summarize(tree):
    docstring = ast.get_docstring(tree) or ""
    paragraph = " ".join(docstring.strip().split("\n\n")[0].split())
    sentence = re.split(r"(?<!e\.g)(?<!i\.e)(?<=[.!?])\s", paragraph)[0]
    sentence = re.sub(r"^(EXPERIMENTAL:|\[[^\]]*\])\s*", "", sentence).replace("**", "").replace("`", "").replace("*", "")
    if len(sentence) <= 150:
        return sentence
    return sentence[:150].rsplit(" ", 1)[0] + "…"


def load_recipes():
    ocr_catalog = json.loads((ROOT / "ocr" / "models.json").read_text())
    recipes = []
    for folder, task in TASKS.items():
        for path in sorted((ROOT / folder).glob("*.py")):
            if SKIP_SCRIPT.search(path.name) or f"{folder}/{path.name}" in BROKEN:
                continue
            text = path.read_text()
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            constants, arguments = read_arguments(tree)
            models = find_models(constants, arguments)
            support = None
            if folder == "ocr":
                entry = ocr_catalog.get(path.name)
                if not entry or entry.get("support") not in OCR_SUPPORT_LEVELS or not entry.get("model_id"):
                    continue
                support = entry["support"]
                models = [entry["model_id"]]
            if not models:
                continue

            url = f"{HUB_BASE}/{folder}/raw/main/{path.name}"
            jobs_header = read_header(text)
            documented = first_documented_command(text, path.name, url)
            if jobs_header:
                command, source = header_command(url, arguments, documented), "header"
            elif documented:
                command, source = format_command(documented[0], url, documented[1]), "documented"
            else:
                command, source = None, "flags-needed"

            script = f"{folder}/{path.name}"
            is_training = path.name.startswith(("train-", "finetune"))
            is_tested = source == "header" or script in SMOKE_TESTED_EXAMPLES
            declared = is_tested and not is_training and support != "experimental" and script not in NOT_DECLARED
            models = [model for model in models if model not in NOT_HUB_MODELS]
            if not models:
                continue
            recipes.append({
                "kind": "train" if is_training else "run",
                "task": TRAINING_TASK if is_training else task,
                "script": script,
                "repo": f"uv-scripts/{folder}",
                "page": f"{HUB_BASE}/{folder}/blob/main/{path.name}",
                "url": url,
                "models": models,
                "support": support,
                "source": source,
                "flavor": jobs_header.get("flavor"),
                "timeout": jobs_header.get("timeout"),
                "command": command,
                "summary": summarize(tree),
                "declared": declared,
            })
    return recipes


def declared_models(recipes):
    return sorted({model for recipe in recipes if recipe["declared"] for model in recipe["models"]}, key=str.lower)


def rank(recipe):
    """Lead with the most dependable recipe for a model."""
    return (
        recipe["support"] == "experimental",
        recipe["source"] != "header",
        recipe["source"] == "flags-needed",
        bool(ALTERNATIVE_SCRIPT.search(recipe["script"])),
        recipe["script"],
    )


def command_block(command):
    return f'<div class="cmd"><pre>{html.escape(command)}</pre><button class="copy">Copy</button></div>'


def recipe_html(recipe):
    status = STATUS[recipe["source"]]
    if recipe["support"] == "experimental":
        status += ", experimental"
    status_class = "warn" if recipe["source"] == "flags-needed" else "muted"
    flavor = f" · {html.escape(recipe['flavor'])}" if recipe["flavor"] else ""
    summary = f'<p class="muted">{html.escape(recipe["summary"])}</p>' if recipe["summary"] else ""
    if recipe["source"] == "flags-needed":
        body = f'<p class="small"><a href="{html.escape(recipe["page"])}">Read the script</a> for its options. There is no ready-made command yet.</p>'
    else:
        body = command_block(recipe["command"])
    name = recipe["script"].split("/")[-1]
    return (
        f'<div class="recipe"><p><a href="{html.escape(recipe["page"])}"><code>{html.escape(name)}</code></a> '
        f'<span class="{status_class} small">{status}{flavor}</span></p>{summary}{body}'
        f'<p class="small muted">From <a href="https://huggingface.co/datasets/{html.escape(recipe["repo"])}">{html.escape(recipe["repo"])}</a>, with more options and examples.</p></div>'
    )


def model_row_html(task, model, recipes):
    ranked = sorted(recipes, key=rank)
    best = ranked[0]
    row_status = ROW_STATUS[best["source"]] + (f" · {best['flavor']}" if best["flavor"] else "")
    row_class = "warn" if best["source"] == "flags-needed" else "muted"
    search_text = " ".join([model, task] + [f"{recipe['script']} {recipe['summary']}" for recipe in ranked]).lower()
    others = ranked[1:]
    more = ""
    if others:
        label = f"{len(others)} more recipe{'s' if len(others) > 1 else ''}"
        more = f'<details class="more"><summary>{label}</summary>' + "".join(recipe_html(recipe) for recipe in others) + "</details>"
    return (
        f'<li data-task="{html.escape(task)}" data-text="{html.escape(search_text)}"><details>'
        f'<summary><span class="name">{html.escape(model)}</span><span class="state {row_class}">{html.escape(row_status)}</span></summary>'
        + f'<p class="model-link small muted"><a href="https://huggingface.co/{html.escape(model)}">Open the model page</a></p>'
        + recipe_html(best) + more + "</details></li>"
    )


def render_index(recipes):
    template = (ROOT / "tools" / "run-on-jobs-template.html").read_text()
    entries = {}
    for recipe in recipes:
        for model in recipe["models"]:
            entries.setdefault((recipe["task"], model), []).append(recipe)

    groups = []
    used_tasks = []
    for task in TASK_ORDER:
        models = sorted((model for entry_task, model in entries if entry_task == task), key=str.lower)
        if not models:
            continue
        used_tasks.append(task)
        rows = "".join(model_row_html(task, model, entries[(task, model)]) for model in models)
        groups.append(f'<section data-group><h2>{html.escape(task)}</h2><ul class="models">{rows}</ul></section>')

    filters = '<button data-task="All" aria-pressed="true">All</button>' + "".join(
        f'<button data-task="{html.escape(task)}" aria-pressed="false">{html.escape(task)}</button>' for task in used_tasks
    )
    tools = "".join(
        f'<li><a href="{HUB_BASE}/{folder}">{html.escape(label)}</a></li>' for folder, label in DATA_TOOLS.items()
    )
    return template.replace("{{FILTERS}}", filters).replace("{{GROUPS}}", "".join(groups)).replace("{{TOOLS}}", tools)


def render_readme(recipes):
    model_lines = "\n".join(f"  - {model}" for model in declared_models(recipes))
    return f"""---
title: Run on Jobs
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
short_description: Batch-run this model on your own data with HF Jobs
models:
{model_lines}
---

# Run on Jobs

A catalogue of [uv-scripts](https://huggingface.co/uv-scripts) recipes: one command runs a model over your own data on [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs).

This Space is generated from [davanstrien/uv-scripts](https://github.com/davanstrien/uv-scripts). To change it, edit the recipes or `tools/build-catalog.py` there. Agents should read [AGENTS.md](./AGENTS.md).
"""


def render_agents():
    return """# Run a model on Hugging Face Jobs

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
"""


def generate():
    recipes = load_recipes()
    catalog = {
        "recipes": sorted(recipes, key=lambda recipe: recipe["script"]),
        "data_tools": [{"repo": f"uv-scripts/{folder}", "description": label} for folder, label in DATA_TOOLS.items()],
    }
    return {
        "index.html": render_index(recipes),
        "README.md": render_readme(recipes),
        "AGENTS.md": render_agents(),
        "recipes.json": json.dumps(catalog, indent=1, ensure_ascii=False) + "\n",
    }


def verify_models(recipes):
    missing = []
    for model in declared_models(recipes):
        try:
            urllib.request.urlopen(f"https://huggingface.co/api/models/{model}", timeout=30)
        except urllib.error.HTTPError as error:
            missing.append(f"{model} (HTTP {error.code})")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-models", action="store_true", help="Check that declared models exist on the Hub")
    args = parser.parse_args()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(ROOT / "tools" / "run-on-jobs-assets", OUTPUT_DIR / "assets")
    for name, content in generate().items():
        (OUTPUT_DIR / name).write_text(content)

    recipes = load_recipes()
    print(f"{len(recipes)} recipes, {len(declared_models(recipes))} declared models")
    if args.verify_models:
        missing = verify_models(recipes)
        if missing:
            print("Declared models not found:", ", ".join(missing))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
