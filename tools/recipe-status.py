#!/usr/bin/env python3
"""Show the maintenance tier and the last smoke test of each recipe with a [tool.hf-jobs] header.

    python3 tools/recipe-status.py            # print the table
    python3 tools/recipe-status.py --write    # write maintenance/STATUS.md
    python3 tools/recipe-status.py --check    # exit 1 if a tiered recipe breaks a MAINTAINING.md rule

The tier comes from each script's [tool.uv-scripts] table. Smoke tests come from
maintenance/verifications.jsonl. See MAINTAINING.md for both.

Scripts are only parsed, never imported or run. Standard library only.
"""

import argparse
import json
import re
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAINTENANCE_DIR = ROOT / "maintenance"
VERIFICATIONS = MAINTENANCE_DIR / "verifications.jsonl"
STATUS = MAINTENANCE_DIR / "STATUS.md"

# Top-level folders that hold tooling, not recipes.
NOT_RECIPE_FOLDERS = {
    "tools",
    "tests",
    "skills",
    "maintenance",
    "run-on-jobs",
    ".claude",
    ".github",
}
TIERS = {"active", "frozen"}
FROZEN_CHECK_DAYS = 120  # quarterly, plus some slack
ACTIVE_CHECK_DAYS = 45  # monthly, plus some slack


def read_header(text):
    """Return the PEP 723 header as a dict, or {} if there is none or it does not parse."""
    match = re.search(r"(?m)^# /// script\n((?:#.*\n)*?)# ///", text)
    if not match:
        return {}
    lines = [
        line[2:] if line.startswith("# ") else line[1:]
        for line in match.group(1).splitlines()
    ]
    try:
        return tomllib.loads("\n".join(lines))
    except tomllib.TOMLDecodeError:
        return {}


def find_recipes():
    """Return one dict per script that has a [tool.hf-jobs] or [tool.uv-scripts] table."""
    recipes = []
    for path in sorted(ROOT.glob("*/*.py")):
        folder = path.parent.name
        if folder in NOT_RECIPE_FOLDERS or folder.startswith("."):
            continue
        tool = read_header(path.read_text()).get("tool", {})
        jobs = tool.get("hf-jobs", {})
        lifecycle = tool.get("uv-scripts", {})
        if not jobs and not lifecycle:
            continue
        recipes.append(
            {
                "script": f"{folder}/{path.name}",
                "image": jobs.get("image", ""),
                "exclude_newer": tool.get("uv", {}).get("exclude-newer", ""),
                "maintenance": lifecycle.get("maintenance", ""),
                "frozen_on": lifecycle.get("frozen_on", ""),
                "superseded_by": lifecycle.get("superseded_by", ""),
            }
        )
    return recipes


def read_verifications():
    """Return the latest smoke-test record for each script."""
    latest = {}
    if not VERIFICATIONS.exists():
        return latest
    for number, line in enumerate(VERIFICATIONS.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            sys.exit(f"{VERIFICATIONS.name}:{number}: not valid JSON")
        script = record.get("script")
        if script and record.get("date", "") >= latest.get(script, {}).get("date", ""):
            latest[script] = record
    return latest


def days_since(iso_date):
    return (date.today() - date.fromisoformat(iso_date[:10])).days


def check(recipes, latest):
    """Return a list of rule breaks for recipes that declare a tier."""
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
        record = latest.get(script)
        if tier == "frozen":
            if not recipe["frozen_on"]:
                problems.append(f"{script}: frozen without frozen_on")
            if not recipe["exclude_newer"]:
                problems.append(f"{script}: frozen without [tool.uv] exclude-newer")
            if recipe["image"] and "@sha256:" not in recipe["image"]:
                problems.append(
                    f"{script}: frozen image is a tag, not a digest: {recipe['image']}"
                )
            if not record or record.get("outcome") != "pass":
                problems.append(
                    f"{script}: frozen without a passing record in {VERIFICATIONS.name}"
                )
        if recipe["superseded_by"]:
            successor = ROOT / script.split("/")[0] / recipe["superseded_by"]
            if not successor.exists():
                problems.append(
                    f"{script}: superseded_by {recipe['superseded_by']} does not exist"
                )
    return problems


def overdue(recipe, record):
    if not recipe["maintenance"]:
        return False
    limit = (
        FROZEN_CHECK_DAYS if recipe["maintenance"] == "frozen" else ACTIVE_CHECK_DAYS
    )
    return not record or days_since(record["date"]) > limit


def render(recipes, latest):
    tiered = [r for r in recipes if r["maintenance"]]
    lines = [
        "# Recipe maintenance status",
        "",
        "Generated by `python3 tools/recipe-status.py --write`. Do not edit by hand.",
        "Tiers and rules: [MAINTAINING.md](../MAINTAINING.md).",
        "",
        f"{len(recipes)} recipes with a header; {len(tiered)} have a tier.",
        "",
        "| Recipe | Tier | Image | Packages | Last smoke test | Result |",
        "|---|---|---|---|---|---|",
    ]
    for recipe in sorted(
        recipes, key=lambda r: (r["maintenance"] == "", r["maintenance"], r["script"])
    ):
        record = latest.get(recipe["script"])
        image = recipe["image"]
        if "@sha256:" in image:
            image = image.split("@sha256:")[0] + "@" + image.split("@sha256:")[1][:12]
        tier = recipe["maintenance"] or "—"
        if recipe["superseded_by"]:
            tier += f" (→ {recipe['superseded_by']})"
        packages = (
            f"as of {recipe['exclude_newer'][:10]}"
            if recipe["exclude_newer"]
            else "latest"
        )
        if record:
            when = record["date"][:10]
            job = record.get("job_url")
            result = f"[{record['outcome']}]({job})" if job else record["outcome"]
        else:
            when, result = "never", ""
        if overdue(recipe, record):
            when += " ⚠️ due"
        image = f"`{image}`" if image else "default"
        lines.append(
            f"| `{recipe['script']}` | {tier} | {image} | {packages} | {when} | {result} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--write", action="store_true", help=f"write {STATUS.relative_to(ROOT)}"
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if a tiered recipe breaks a rule"
    )
    args = parser.parse_args()

    recipes = find_recipes()
    latest = read_verifications()
    unknown = sorted(set(latest) - {r["script"] for r in recipes})
    if unknown:
        print(
            f"note: records for scripts without a header: {', '.join(unknown)}",
            file=sys.stderr,
        )

    if args.check:
        problems = check(recipes, latest)
        for problem in problems:
            print(problem)
        untiered = sum(1 for r in recipes if not r["maintenance"])
        print(
            f"{len(problems)} problem(s); {untiered} of {len(recipes)} recipes have no tier yet."
        )
        sys.exit(1 if problems else 0)

    table = render(recipes, latest)
    if args.write:
        MAINTENANCE_DIR.mkdir(exist_ok=True)
        STATUS.write_text(table)
        print(f"wrote {STATUS.relative_to(ROOT)}")
    else:
        print(table)


if __name__ == "__main__":
    main()
