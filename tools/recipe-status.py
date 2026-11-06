#!/usr/bin/env python3
"""Show the maintenance tier and the last smoke test of each recipe with a [tool.hf-jobs] header.

    python3 tools/recipe-status.py            # print the table
    python3 tools/recipe-status.py --write    # write maintenance/STATUS.md
    python3 tools/recipe-status.py --check    # exit 1 on a broken header, a bad record or a rule break

The tier comes from each script's [tool.uv-scripts] table. Smoke tests come from
maintenance/verifications.jsonl. See MAINTAINING.md for both.

--check cannot see the model revision in a script's code. For a frozen recipe, the evidence is the
model_revision in its smoke-test record; check the code by hand when you freeze or change it.

Scripts are only parsed, never imported or run. Standard library only.
"""

import argparse
import json
import re
import sys
import tomllib
from datetime import date
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
MAINTENANCE_DIR = ROOT / "maintenance"
VERIFICATIONS = MAINTENANCE_DIR / "verifications.jsonl"
STATUS = MAINTENANCE_DIR / "STATUS.md"

# Top-level folders that hold tooling, not recipes.
NOT_RECIPE_FOLDERS = {"tools", "tests", "skills", "maintenance", "run-on-jobs"}
TIERS = {"active", "frozen"}
OUTCOMES = {"pass", "fail"}
REQUIRED_RECORD_FIELDS = ["date", "script", "image", "model_revision", "outcome"]
FROZEN_CHECK_DAYS = 120  # quarterly, plus some slack
ACTIVE_CHECK_DAYS = 45  # monthly, plus some slack

# The reference regex from PEP 723 (https://peps.python.org/pep-0723/#reference-implementation).
PEP723_BLOCK = re.compile(
    r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
)
DIGEST_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class HeaderError(Exception):
    pass


def read_header(text):
    """Return the PEP 723 script block as a dict, or None if the script has none.

    Raises HeaderError if a block is there but is malformed, duplicated or not valid TOML.
    """
    blocks = [m for m in PEP723_BLOCK.finditer(text) if m.group("type") == "script"]
    if len(blocks) > 1:
        raise HeaderError("more than one '# /// script' block")
    if not blocks:
        if re.search(r"(?m)^# /// script", text):
            raise HeaderError(
                "'# /// script' block is not closed by '# ///', or has a line without '#'"
            )
        return None
    lines = []
    for line in blocks[0].group("content").splitlines(keepends=True):
        if line.startswith("# "):
            lines.append(line[2:])
        else:
            lines.append(line[1:])
    try:
        return tomllib.loads("".join(lines))
    except tomllib.TOMLDecodeError as error:
        raise HeaderError(f"header is not valid TOML: {error}") from error


def is_iso_date(value):
    try:
        date.fromisoformat(str(value)[:10])
    except ValueError:
        return False
    return True


def find_recipes():
    """Return (recipes, problems): one dict per script with a [tool.hf-jobs] or
    [tool.uv-scripts] table, and the header errors found on the way."""
    recipes = []
    problems = []
    for path in sorted(ROOT.glob("*/*.py")):
        folder = path.parent.name
        if folder in NOT_RECIPE_FOLDERS or folder.startswith("."):
            continue
        script = f"{folder}/{path.name}"
        try:
            header = read_header(path.read_text())
        except HeaderError as error:
            problems.append(f"{script}: {error}")
            continue
        if header is None:
            continue
        tool = header.get("tool", {})
        jobs = tool.get("hf-jobs", {})
        lifecycle = tool.get("uv-scripts", {})
        if not jobs and not lifecycle:
            continue
        recipes.append(
            {
                "script": script,
                "image": jobs.get("image", ""),
                "exclude_newer": str(tool.get("uv", {}).get("exclude-newer", "")),
                "maintenance": lifecycle.get("maintenance", ""),
                "frozen_on": str(lifecycle.get("frozen_on", "")),
                "superseded_by": lifecycle.get("superseded_by", ""),
            }
        )
    return recipes, problems


def read_verifications():
    """Return (records by script in file order, problems). Bad lines are reported and skipped."""
    records = {}
    problems = []
    if not VERIFICATIONS.exists():
        return records, problems
    for number, line in enumerate(VERIFICATIONS.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{VERIFICATIONS.relative_to(ROOT)}:{number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"{where}: not valid JSON")
            continue
        if not isinstance(record, dict):
            problems.append(f"{where}: not a JSON object")
            continue
        missing = [field for field in REQUIRED_RECORD_FIELDS if not record.get(field)]
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
            continue
        if not is_iso_date(record["date"]):
            problems.append(f"{where}: date {record['date']!r} is not YYYY-MM-DD")
            continue
        if record["outcome"] not in OUTCOMES:
            problems.append(
                f"{where}: outcome {record['outcome']!r}, expected pass or fail"
            )
            continue
        records.setdefault(record["script"], []).append(record)
    return records, problems


def latest_record(records_for_script):
    """The newest record by date; on the same date, the one appended last."""
    latest = None
    for record in records_for_script:
        if latest is None or record["date"][:10] >= latest["date"][:10]:
            latest = record
    return latest


def days_since(iso_date):
    return (date.today() - date.fromisoformat(iso_date[:10])).days


def check_frozen(recipe, latest):
    """Rule breaks for a frozen recipe. The latest record must be a pass of the current freeze."""
    script = recipe["script"]
    problems = []
    if not recipe["frozen_on"]:
        problems.append(f"{script}: frozen without frozen_on")
    elif not is_iso_date(recipe["frozen_on"]):
        problems.append(
            f"{script}: frozen_on {recipe['frozen_on']!r} is not YYYY-MM-DD"
        )
    if not recipe["exclude_newer"]:
        problems.append(f"{script}: frozen without [tool.uv] exclude-newer")
    if recipe["image"] and not DIGEST_IMAGE.match(recipe["image"]):
        problems.append(
            f"{script}: frozen image is not '<name>@sha256:<64 hex>': {recipe['image']}"
        )

    if latest is None:
        problems.append(f"{script}: frozen without a smoke-test record")
        return problems
    if latest["outcome"] != "pass":
        problems.append(f"{script}: latest smoke test ({latest['date'][:10]}) failed")
        return problems
    if latest["image"] != recipe["image"]:
        problems.append(
            f"{script}: latest pass ran image {latest['image']}, not the header's image"
        )
    if latest.get("exclude_newer", "") != recipe["exclude_newer"]:
        problems.append(
            f"{script}: latest pass ran with exclude_newer {latest.get('exclude_newer', '')!r}, "
            f"not the header's {recipe['exclude_newer']!r}"
        )
    if (
        is_iso_date(recipe["frozen_on"])
        and latest["date"][:10] < recipe["frozen_on"][:10]
    ):
        problems.append(
            f"{script}: latest pass ({latest['date'][:10]}) is older than frozen_on"
        )
    if not COMMIT_SHA.match(latest["model_revision"]):
        problems.append(
            f"{script}: latest pass has model_revision {latest['model_revision']!r}, not a 40-hex commit"
        )
    return problems


def check(recipes, records):
    """Rule breaks for recipes that declare a tier."""
    problems = []
    for recipe in recipes:
        script, tier = recipe["script"], recipe["maintenance"]
        if not tier:
            continue
        if tier not in TIERS:
            problems.append(
                f"{script}: maintenance = {tier!r}, expected one of {sorted(TIERS)}"
            )
            continue
        if tier == "frozen":
            problems.extend(
                check_frozen(recipe, latest_record(records.get(script, [])))
            )
        elif recipe["frozen_on"] or recipe["exclude_newer"]:
            problems.append(
                f"{script}: active, but has frozen_on or exclude-newer (see MAINTAINING.md)"
            )
        if recipe["superseded_by"]:
            successor = ROOT / script.split("/")[0] / recipe["superseded_by"]
            if not successor.exists():
                problems.append(
                    f"{script}: superseded_by {recipe['superseded_by']} does not exist"
                )
    return problems


def cell(text):
    """Escape text for a Markdown table cell."""
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def overdue(recipe, record):
    if not recipe["maintenance"]:
        return False
    limit = (
        FROZEN_CHECK_DAYS if recipe["maintenance"] == "frozen" else ACTIVE_CHECK_DAYS
    )
    return record is None or days_since(record["date"]) > limit


def render(recipes, records):
    tiered = [r for r in recipes if r["maintenance"]]
    lines = [
        "# Recipe maintenance status",
        "",
        "Generated by `python3 tools/recipe-status.py --write`. Do not edit by hand.",
        "Tiers and rules: [MAINTAINING.md](../MAINTAINING.md).",
        "",
        f"Tiered: {len(tiered)} of {len(recipes)} recipes with a header.",
        "",
        "| Recipe | Tier | Image | Packages | Last smoke test | Result |",
        "|---|---|---|---|---|---|",
    ]
    for recipe in sorted(
        recipes, key=lambda r: (r["maintenance"] == "", r["maintenance"], r["script"])
    ):
        record = latest_record(records.get(recipe["script"], []))

        image = recipe["image"]
        if "@sha256:" in image:
            name, digest = image.split("@sha256:", 1)
            image = f"{name}@{digest[:12]}"
        image = f"`{cell(image)}`" if image else "default"

        tier = recipe["maintenance"] or "—"
        if recipe["superseded_by"]:
            tier += f" (→ {recipe['superseded_by']})"

        if recipe["exclude_newer"]:
            packages = f"as of {recipe['exclude_newer'][:10]}"
        else:
            packages = "latest"

        if record:
            when = record["date"][:10]
            job = record.get("job_url")
            if job:
                result = f"[{record['outcome']}]({quote(job, safe=':/?=&#%')})"
            else:
                result = record["outcome"]
        else:
            when, result = "never", ""
        if overdue(recipe, record):
            when += " ⚠️ due"

        row = [
            f"`{cell(recipe['script'])}`",
            cell(tier),
            image,
            cell(packages),
            cell(when),
            result,
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--write", action="store_true", help=f"write {STATUS.relative_to(ROOT)}"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 on a broken header, a bad record or a rule break",
    )
    args = parser.parse_args()

    recipes, header_problems = find_recipes()
    records, record_problems = read_verifications()
    unknown = sorted(set(records) - {r["script"] for r in recipes})
    if unknown:
        print(
            f"note: records for scripts without a header: {', '.join(unknown)}",
            file=sys.stderr,
        )

    if args.check:
        problems = header_problems + record_problems + check(recipes, records)
        for problem in problems:
            print(problem)
        untiered = sum(1 for r in recipes if not r["maintenance"])
        print(
            f"{len(problems)} problem(s); {untiered} of {len(recipes)} recipes have no tier yet."
        )
        sys.exit(1 if problems else 0)

    for problem in header_problems + record_problems:
        print(f"warning: {problem}", file=sys.stderr)
    table = render(recipes, records)
    if args.write:
        MAINTENANCE_DIR.mkdir(exist_ok=True)
        STATUS.write_text(table)
        print(f"wrote {STATUS.relative_to(ROOT)}")
    else:
        print(table)


if __name__ == "__main__":
    main()
