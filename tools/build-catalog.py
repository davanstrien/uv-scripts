#!/usr/bin/env python3
"""Generate the Run on Jobs Space from the recipes in this repo.

    python3 tools/build-catalog.py                   # write run-on-jobs/
    python3 tools/build-catalog.py --verify-models   # also check each declared model on the Hub

The sync workflow runs this on merge, so run-on-jobs/ is not committed.

Scripts are only parsed, never imported or run. Standard library only.

Within each task, models are listed newest first, by the date the model was created on the Hub. Without
network access the dates are missing, and the models are listed alphabetically.
"""

import argparse
import ast
import concurrent.futures
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
SPACE_URL = "https://huggingface.co/spaces/uv-scripts/run-on-jobs"

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
    "iiif-tiles": "Publish images as a IIIF collection",
    "data-processing": "Convert uploads to optimized Parquet",
}

# Recipes without a [tool.hf-jobs] header whose documented command passed a smoke Job with checked output.
# Only these and header recipes are declared on model pages.
SMOKE_TESTED_EXAMPLES = {
    "gliner/extract-entities.py",
    "sam3/detect-objects.py",
    "sam3/segment-objects.py",
}

# Models other than a script's default that ran the recipe in a smoke Job with checked output.
# Each gets its own entry on the page and on its model page, with `--model <id>` in the command.
TESTED_MODELS = {
    "embeddings/generate-embeddings.py": [
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-m3",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-large",
        "Qwen/Qwen3-Embedding-0.6B",
        "google/embeddinggemma-2",
    ],
}

# Left out of the Space until they run again.
BROKEN = set()

# Listed on the page, but not declared on model pages until the plain command works for the model.
NOT_DECLARED = {
    "vllm/generate-responses.py",  # the default model needs --max-model-len on the header's a100-large
}

# Recipe metadata that names a family rather than a Hub repo.
NOT_HUB_MODELS = {"PaddlePaddle/PP-OCRv6"}

# Added to a command when the script has --max-samples and its documented example does not set one.
FIRST_RUN_SAMPLES = 10

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


def lines_without_examples(tree):
    """Return the line numbers where an `hf jobs uv run` is not a documented example.

    An f-string is a template the script fills in when it runs, such as the card it writes for its output.
    A one-line string holds only part of a command, such as one line of a printed example.
    """
    skipped = set()
    for node in ast.walk(tree):
        is_f_string = isinstance(node, ast.JoinedStr)
        is_one_line_string = isinstance(node, ast.Constant) and isinstance(node.value, str) and node.lineno == node.end_lineno
        if is_f_string or is_one_line_string:
            skipped.update(range(node.lineno, node.end_lineno + 1))
    return skipped


def first_documented_command(text, tree, script_name, url):
    """Return the first documented example that launches this script, as (job_flags, script_arguments)."""
    skipped_lines = lines_without_examples(tree)
    for match in re.finditer(r"hf jobs uv run(?:[^\n]*\\\n)*[^\n]*", text):
        if script_name not in match.group(0):
            continue
        if text.count("\n", 0, match.start()) + 1 in skipped_lines:
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
    """Rewrite the stand-ins that script docs use, so the page has two: `your-username` and capitalised names."""
    command = re.sub(r"<(user|you|username|your-username)>(?=/)", "your-username", command)
    command = re.sub(r"(?<![-\w])(your-name|yourname|my-username|username|user)(?=/)", "your-username", command)
    command = re.sub(r"(?<![-\w])(your-input-dataset|input-dataset|my-dataset)(?![-\w])", "INPUT_DATASET", command)
    command = re.sub(r"(?<![-\w])(your-output-dataset|output-dataset|my-output)(?![-\w])", "OUTPUT_DATASET", command)
    command = re.sub(r"(?<![-\w/.:])input output(?![-\w/])", "INPUT_DATASET OUTPUT_DATASET", command)
    command = re.sub(r"your-username/(INPUT_DATASET|OUTPUT_DATASET)", r"\1", command)
    # A shell reads <NAME> as a redirect, so a pasted command would fail before it reached the script.
    return re.sub(r"<([A-Z][A-Z_]*)>", r"\1", command)


def add_sample_limit(command, arguments):
    """End the command with a small --max-samples when the script has one, so a first run is short and cheap."""
    has_option = any(argument["name"] == "--max-samples" for argument in arguments)
    if not has_option or "--max-samples" in command:
        return command
    return f"{command} --max-samples {FIRST_RUN_SAMPLES}"


def command_for_model(command, model):
    """Return the command with --model set to this model, before any --max-samples at the end."""
    if re.search(r"--model \S+", command):
        return re.sub(r"--model \S+", f"--model {model}", command)
    limit = re.search(r" --max-samples \d+$", command)
    if limit:
        return f"{command[:limit.start()]} --model {model}{limit.group(0)}"
    return f"{command} --model {model}"


def command_problems(recipes):
    """Return the commands that would not run as pasted: template text, or brackets outside quotes."""
    problems = []
    for recipe in recipes:
        if not recipe["command"]:
            continue
        unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", "", recipe["command"])
        found = sorted(set(re.findall(r"[{}<>]", unquoted)))
        if found:
            problems.append(f"{recipe['script']} ({' '.join(found)})")
    return problems


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


# One public demo input per task. Commands in these tasks use it in place of the user's own input, so they run
# as pasted and a first run is small. Add a task here when it has a demo input; nothing is needed per recipe.
DEMO_INPUTS = {
    "OCR": {
        "input": "uv-scripts/ocr-demo",
        "about": "seven scanned pages from the uv-scripts/ocr-demo dataset",
        "replace": [(r"\bINPUT_DATASET\b", "uv-scripts/ocr-demo"), (r"\bOUTPUT_DATASET\b", "your-username/{stem}-demo")],
    },
    "Transcription": {
        "input": "hf://buckets/uv-scripts/transcription-demo/podcast",
        "about": "a 4-minute NASA podcast clip from the uv-scripts/transcription-demo bucket",
        "replace": [
            (r"hf://buckets/your-username/[\w.-]+:/input:ro", "hf://buckets/uv-scripts/transcription-demo/podcast:/input:ro")
        ],
    },
}
OUTPUT_BUCKET = re.compile(r"hf://buckets/your-username/([\w.-]+)[^\s:]*:/output")


def apply_demo_input(recipe):
    """Point the command at its task's demo input. Returns a note for the page, or None if nothing changed."""
    demo = DEMO_INPUTS.get(recipe["task"])
    if not demo or not recipe["command"]:
        return None
    stem = recipe["script"].split("/")[-1].removesuffix(".py")
    command = recipe["command"]
    for pattern, replacement in demo["replace"]:
        command = re.sub(pattern, replacement.format(stem=stem), command)
    if command == recipe["command"]:
        return None
    recipe["command"] = command
    recipe["demo_input"] = demo["input"]
    note = f"This command runs on {demo['about']}. Replace it with your own data for a real run."
    bucket = OUTPUT_BUCKET.search(command)
    if bucket:
        note += f" Create the output bucket first: hf buckets create your-username/{bucket.group(1)} --private"
    return note


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
            documented = first_documented_command(text, tree, path.name, url)
            if jobs_header:
                command, source = header_command(url, arguments, documented), "header"
            elif documented:
                command, source = format_command(documented[0], url, documented[1]), "documented"
            else:
                command, source = None, "flags-needed"
            if command:
                command = add_sample_limit(command, arguments)

            script = f"{folder}/{path.name}"
            is_training = path.name.startswith(("train-", "finetune"))
            is_tested = source == "header" or script in SMOKE_TESTED_EXAMPLES
            declared = is_tested and not is_training and support != "experimental" and script not in NOT_DECLARED
            models = [model for model in models if model not in NOT_HUB_MODELS]
            if not models:
                continue
            recipe = {
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
                "demo_input": None,
            }
            recipe["demo_note"] = apply_demo_input(recipe)
            recipes.append(recipe)
            for tested_model in TESTED_MODELS.get(script, []):
                has_model_option = any(argument["name"] == "--model" for argument in arguments)
                if not command or not has_model_option:
                    sys.exit(f"TESTED_MODELS: {script} needs a command and a --model option")
                recipes.append({**recipe, "models": [tested_model], "command": command_for_model(command, tested_model)})
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
    flavor = ""
    if recipe["flavor"]:
        # The page fills the price from the public Jobs hardware list when it loads, so it is never stale.
        flavor = f' · {html.escape(recipe["flavor"])}<span class="price" data-flavor="{html.escape(recipe["flavor"])}"></span>'
    summary = f'<p class="muted">{html.escape(recipe["summary"])}</p>' if recipe["summary"] else ""
    if recipe["source"] == "flags-needed":
        body = f'<p class="small"><a href="{html.escape(recipe["page"])}">Read the script</a> for its options. There is no ready-made command yet.</p>'
    else:
        body = command_block(recipe["command"])
        if recipe.get("demo_note"):
            body += f'<p class="small muted">{html.escape(recipe["demo_note"])}</p>'
    name = recipe["script"].split("/")[-1]
    return (
        f'<div class="recipe"><p><a href="{html.escape(recipe["page"])}"><code>{html.escape(name)}</code></a> '
        f'<span class="{status_class} small">{status}{flavor}</span></p>{summary}{body}'
        f'<p class="small muted">From <a href="https://huggingface.co/datasets/{html.escape(recipe["repo"])}">{html.escape(recipe["repo"])}</a>, with more options and examples.</p></div>'
    )


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def date_html(created):
    """Month and year the model was created on the Hub, such as "Nov 2026"."""
    if not created:
        return ""
    year, month = created[:4], int(created[5:7])
    label = f"{MONTHS[month - 1]} {year}"
    return f'<span class="date muted" title="Created on the Hub on {created}">{label}</span>'


def size_label(parameters):
    """Parameter count from the Hub, such as "509M" or "3.3B", or "" when the Hub has none."""
    if not parameters:
        return ""
    if parameters >= 10e9:
        return f"{round(parameters / 1e9)}B"
    if parameters >= 1e9:
        return f"{parameters / 1e9:.1f}B"
    return f"{round(parameters / 1e6)}M"


def model_row_html(task, model, recipes, hub, extra):
    ranked = sorted(recipes, key=rank)
    best = ranked[0]
    info = hub.get(model, {})
    # The collapsed row shows the model's age and size; the GPU and its price are in the opened row.
    # A missing command is the only status shown.
    size = size_label(info.get("parameters"))
    facts = [f"{size} params" if size else ""]
    if best["source"] == "flags-needed":
        facts.append(ROW_STATUS["flags-needed"])
    row_status = " · ".join(fact for fact in facts if fact)
    row_class = "warn" if best["source"] == "flags-needed" else "muted"
    description = f'<span class="desc muted">{html.escape(best["summary"])}</span>' if best["summary"] else ""
    search_text = " ".join([model, task] + [f"{recipe['script']} {recipe['summary']}" for recipe in ranked]).lower()
    others = ranked[1:]
    more = ""
    if others:
        label = f"{len(others)} more recipe{'s' if len(others) > 1 else ''}"
        more = f'<details class="more"><summary>{label}</summary>' + "".join(recipe_html(recipe) for recipe in others) + "</details>"
    return (
        f'<li data-task="{html.escape(task)}" data-model="{html.escape(model.lower())}" data-text="{html.escape(search_text)}"'
        + (' data-extra="1"' if extra else "")
        + '><details>'
        f'<summary><span class="name">{html.escape(model)}</span>{date_html(info.get("created"))}<span class="state {row_class}">{html.escape(row_status)}</span>{description}</summary>'
        + f'<p class="model-link small muted"><a href="https://huggingface.co/{html.escape(model)}">Open the model page</a>'
        + f' · <a href="{SPACE_URL}?model={html.escape(model)}">Link to this model</a></p>'
        + recipe_html(best) + more + "</details></li>"
    )


ROWS_SHOWN = 5  # per task, before "Show N more"; search and task filters show every row


def render_index(recipes, hub):
    template = (ROOT / "tools" / "run-on-jobs-template.html").read_text()
    entries = {}
    for recipe in recipes:
        for model in recipe["models"]:
            entries.setdefault((recipe["task"], model), []).append(recipe)

    groups = []
    used_tasks = []
    for task in TASK_ORDER:
        models = sorted((model for entry_task, model in entries if entry_task == task), key=str.lower)
        # Newest first. The sort is stable, so models with the same date (or no date) stay alphabetical.
        models = sorted(models, key=lambda model: hub.get(model, {}).get("created") or "", reverse=True)
        if not models:
            continue
        used_tasks.append((task, len(models)))
        rows = "".join(
            model_row_html(task, model, entries[(task, model)], hub, extra=index >= ROWS_SHOWN)
            for index, model in enumerate(models)
        )
        hidden = len(models) - ROWS_SHOWN
        more = f'<button class="show-more" hidden>Show {hidden} more</button>' if hidden > 0 else ""
        groups.append(f'<section data-group><h2>{html.escape(task)}</h2><ul class="models">{rows}</ul>{more}</section>')

    filters = '<button data-task="All" aria-pressed="true">All</button>' + "".join(
        f'<button data-task="{html.escape(task)}" aria-pressed="false">{html.escape(task)} <span class="count">{count}</span></button>'
        for task, count in used_tasks
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
- To send a person to one model, link to `https://huggingface.co/spaces/uv-scripts/run-on-jobs?model=<org>/<name>`. The page opens at that model.
- `kind`: `run` applies a model to data; `train` fine-tunes one.
- `demo_input`: set when `command` reads a small public demo input instead of the user's data. Run it as is to try the recipe, then replace that input with the user's data.
- `model_created`: the date each model was created on the Hub (YYYY-MM-DD), to find the newest. This is when the repo was created, which can be some weeks before the public release.
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
- Commands set `--max-samples 10` where the script has it. Remove it for a full run, and add `--private` if the output shouldn't be public.

## Check it

- Use `--detach` for long runs, then `hf jobs logs -f <id>`, `hf jobs wait <id>` or `hf jobs inspect <id>`. `hf jobs cancel <id>` stops one.
- Look at the output rows. A completed job doesn't prove they're correct.
"""


def generate(recipes, hub):
    catalog = {
        "recipes": [
            {
                **{key: value for key, value in recipe.items() if key != "demo_note"},
                "model_created": {model: hub.get(model, {}).get("created") for model in recipe["models"]},
            }
            for recipe in sorted(recipes, key=lambda recipe: recipe["script"])
        ],
        "data_tools": [{"repo": f"uv-scripts/{folder}", "description": label} for folder, label in DATA_TOOLS.items()],
    }
    return {
        "index.html": render_index(recipes, hub),
        "README.md": render_readme(recipes),
        "AGENTS.md": render_agents(),
        "recipes.json": json.dumps(catalog, indent=1, ensure_ascii=False) + "\n",
    }


def all_models(recipes):
    return sorted({model for recipe in recipes for model in recipe["models"]}, key=str.lower)


def fetch_model(model):
    """Return ({"created": date or None, "parameters": count or None}, error or None) for one model."""
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{model}", timeout=30) as response:
            info = json.load(response)
    except urllib.error.HTTPError as error:
        return {}, f"HTTP {error.code}"
    except (urllib.error.URLError, TimeoutError) as error:
        return {}, f"no response: {error}"
    created = info.get("createdAt", "")[:10] or None
    parameters = (info.get("safetensors") or {}).get("total")
    return {"created": created, "parameters": parameters}, None


def fetch_models(recipes):
    """Look up every model on the Hub. Returns {model: (info, error or None)}; see fetch_model."""
    models = [model for model in all_models(recipes) if model not in NOT_HUB_MODELS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(models, pool.map(fetch_model, models)))


def verify_models(recipes, hub_info):
    missing = []
    for model in declared_models(recipes):
        _, error = hub_info.get(model, ({}, "not looked up"))
        if error:
            missing.append(f"{model} ({error})")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-models", action="store_true", help="Check that declared models exist on the Hub")
    args = parser.parse_args()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(ROOT / "tools" / "run-on-jobs-assets", OUTPUT_DIR / "assets")
    recipes = load_recipes()
    hub_info = fetch_models(recipes)
    hub = {model: info for model, (info, _) in hub_info.items()}
    for name, content in generate(recipes, hub).items():
        (OUTPUT_DIR / name).write_text(content)

    dated = sum(1 for info in hub.values() if info.get("created"))
    print(f"{len(recipes)} recipes, {len(declared_models(recipes))} declared models, {dated} with a Hub date")
    problems = command_problems(recipes)
    if problems:
        print("Commands that would not run as pasted:", ", ".join(problems))
        return 1
    if args.verify_models:
        missing = verify_models(recipes, hub_info)
        if missing:
            print("Declared models not found:", ", ".join(missing))
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
