#!/usr/bin/env bash
# verify-superset.sh <folder> <hf_repo_id> [repo_type]
#
# Refuse (exit 1) if hub-sync would DELETE any file on the Hub repo. hub-sync runs
#   hf upload <repo> <folder> --exclude=".git*" --exclude=".github*" --delete="*"
# which deletes every Hub file (except the root .gitattributes) that this upload does
# not write. So a Hub file is safe only if the folder holds it on disk AND the upload
# does not exclude it. A Hub `.gitignore` or `.github/...` file is always deleted.
#
# Fails CLOSED on every error mode; passes for a not-yet-created (brand-new) repo.
# Runs both locally (after `git add <folder>`) and as a CI pre-step in the sync workflow.
# HF_ENDPOINT overrides the Hub URL (used by tests).
set -euo pipefail

folder="${1:?usage: verify-superset.sh <folder> <hf_repo_id> [repo_type]}"
repo="${2:?usage: verify-superset.sh <folder> <hf_repo_id> [repo_type]}"
rtype="${3:-dataset}"   # dataset | space | model — selects the /api/<type>s tree endpoint
HF_TOKEN="${HF_TOKEN:-$(hf auth token 2>/dev/null || true)}"
export HF_TOKEN

exec python3 - "$folder" "$repo" "$rtype" <<'PY'
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from fnmatch import fnmatchcase

folder, repo, rtype = sys.argv[1], sys.argv[2], sys.argv[3]
endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
token = os.environ.get("HF_TOKEN", "").strip()

# What `hf upload` skips: hub-sync's --exclude patterns plus huggingface_hub's
# DEFAULT_IGNORE_PATTERNS (v2.4.0). Matched with fnmatchcase, like huggingface_hub.
UPLOAD_EXCLUDES = [".git*", ".github*"]
DEFAULT_IGNORE_PATTERNS = [
    ".git", ".git/*", "*/.git", "**/.git/**",
    ".cache/huggingface", ".cache/huggingface/*", "*/.cache/huggingface", "**/.cache/huggingface/**",
]
# huggingface_hub never deletes this one file, even with --delete="*".
NEVER_DELETED = ".gitattributes"


def block(message):
    print(f"❌ BLOCK — {message}; cannot verify, failing closed.")
    sys.exit(1)


if token and not re.fullmatch(r"[\x21-\x7e]+", token):
    block("HF_TOKEN contains whitespace or control characters")
headers = {"Authorization": f"Bearer {token}"} if token else {}


def fetch(url):
    """Return (status, body, link_header). Retries 429, 5xx and network errors 3 times."""
    status = 0
    for attempt in (1, 2, 3):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read(), response.headers.get("Link", "")
        except urllib.error.HTTPError as error:
            status = error.code
            if status != 429 and not 500 <= status < 600:
                return status, b"", ""
            reason = f"HTTP {status}"
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            status, reason = 0, f"network error ({type(error).__name__})"
        print(f"… Hub API {reason} for {repo}; retry {attempt}/3 after {attempt}s…", file=sys.stderr)
        time.sleep(attempt)
    return status, b"", ""


def next_page(link_header):
    """The rel="next" URL from a Link header, or None. Blocks on a header it can't read."""
    if not link_header:
        return None
    found = None
    for part in link_header.split(","):
        match = re.fullmatch(r'\s*<([^>]+)>\s*((?:;\s*[^;]+)*)\s*', part)
        if not match:
            block(f"Hub API returned a Link header that could not be parsed for {repo}")
        params = [p.strip() for p in match.group(2).split(";") if p.strip()]
        rels = [p.split("=", 1)[1].strip().strip('"') for p in params if p.lower().startswith("rel=")]
        if any("next" in rel.split() for rel in rels):
            found = match.group(1)
    return found


# 1. Every file on the Hub repo, across all tree pages.
url = f"{endpoint}/api/{rtype}s/{repo}/tree/main?recursive=true"
hub_files = set()
seen_urls = set()
page = 0
while url:
    if url in seen_urls:
        block(f"Hub API pagination repeated a page for {repo}")
    seen_urls.add(url)
    page += 1
    try:
        status, body, link = fetch(url)
    except Exception as error:  # noqa: BLE001 — any unexpected failure must block, not pass
        block(f"listing {repo} failed ({type(error).__name__})")
    if status == 404 and page == 1:
        print(f"✅ OK — {repo} doesn't exist yet (brand-new repo); nothing on the Hub to preserve.")
        sys.exit(0)
    if status != 200:
        block(f"Hub API returned HTTP {status} for {repo} (page {page})")
    try:
        entries = json.loads(body)
    except ValueError:
        block(f"Hub API returned a response that is not JSON for {repo} (page {page})")
    if not isinstance(entries, list):
        block(f"Hub API returned {type(entries).__name__}, not a list, for {repo} (page {page})")
    for entry in entries:
        kind = entry.get("type") if isinstance(entry, dict) else None
        path = entry.get("path") if isinstance(entry, dict) else None
        if kind not in ("file", "directory") or not isinstance(path, str) or not path:
            block(f"Hub API returned an unexpected tree entry for {repo} (page {page})")
        if kind == "file":
            hub_files.add(path)
    url = next_page(link)

# 2. Every file this upload will write: tracked in git, present on disk, not excluded.
try:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", folder], check=True, capture_output=True
    ).stdout.decode()
except (subprocess.CalledProcessError, OSError, UnicodeDecodeError) as error:
    block(f"git ls-files failed for '{folder}' ({type(error).__name__})")
prefix = folder.rstrip("/") + "/"
uploaded = set()
for tracked in filter(None, listed.split("\0")):
    if not tracked.startswith(prefix) or not os.path.isfile(tracked):
        continue
    relpath = tracked[len(prefix):]
    if any(fnmatchcase(relpath, p) for p in UPLOAD_EXCLUDES + DEFAULT_IGNORE_PATTERNS):
        continue
    uploaded.add(relpath)

# 3. Hub files the sync would delete.
at_risk = hub_files - {NEVER_DELETED}
if not at_risk:
    print(f"✅ OK — {repo} has no content files to preserve (empty or .gitattributes-only).")
    sys.exit(0)
missing = sorted(at_risk - uploaded)
if missing:
    print("❌ BLOCK — these Hub files are missing locally (or excluded from the upload) and WOULD BE DELETED on sync:")
    print("\n".join(missing))
    sys.exit(1)
print(f"✅ OK — '{folder}' is a superset of {repo} ({len(at_risk)} Hub files all present).")
PY
