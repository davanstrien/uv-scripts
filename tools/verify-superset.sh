#!/usr/bin/env bash
# verify-superset.sh <folder> <hf_repo_id> [repo_type]
#
# Refuse (exit 1) if hub-sync would DELETE any file on the Hub repo. hub-sync runs
#   hf upload <repo> <folder> --exclude=<git/CI paths> --delete="*"
# which deletes every Hub file (except the root .gitattributes) that this upload does
# not write. So a Hub file is safe only if the folder holds it on disk AND the upload
# does not exclude it. A Hub `.gitignore` or `.github/...` file is always deleted.
# It compares against the files on disk, which is what `hf upload` reads: in CI that is
# the checkout; locally, untracked files count too, so check from a clean tree.
#
# Fails CLOSED on every error mode; passes for a not-yet-created (brand-new) repo.
# Runs both locally and as a CI pre-step in the sync workflow.
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
import sys
import time
import urllib.error
import urllib.request
from fnmatch import fnmatchcase
from pathlib import Path

folder, repo, rtype = sys.argv[1], sys.argv[2], sys.argv[3]
endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
token = os.environ.get("HF_TOKEN", "").strip()

# What `hf upload` skips: hub-sync's --exclude patterns plus huggingface_hub's
# DEFAULT_IGNORE_PATTERNS (v2.4.0). Matched with fnmatchcase, like huggingface_hub.
# The union of hub-sync v0.2.0 (".git*", ".github*") and v0.3.0 (".git*", "*/.git/",
# "*/.github/"; huggingface_hub turns a trailing "/" into "/*"). Treating more paths as
# excluded only makes the guard block more, so it is safe under either version.
UPLOAD_EXCLUDES = [".git*", ".github*", "*/.git/*", "*/.github/*"]
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
    found = []
    for part in link_header.split(","):
        match = re.fullmatch(r"\s*<([^>]*)>(.*)", part, flags=re.DOTALL)
        if not match or not match.group(1):
            block(f"Hub API returned a Link header that could not be parsed for {repo}")
        for param in match.group(2).split(";"):
            if not param.strip():
                continue
            name, sep, value = param.partition("=")
            if name.strip().lower() != "rel":
                continue
            relations = value.strip().strip('"').lower().split()
            if not sep or not relations:
                block(f"Hub API returned a Link header with an empty rel for {repo}")
            if "next" in relations:
                found.append(match.group(1))
    if len(found) > 1:
        block(f"Hub API returned more than one next page for {repo}")
    # Fail closed if "next" appears anywhere outside the URLs but no next page was read.
    if not found and "next" in re.sub(r"<[^>]*>", "", link_header).lower():
        block(f"Hub API returned a Link header that mentions next but could not be read for {repo}")
    return found[0] if found else None


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

# 2. Every file this upload will write, found the way huggingface_hub v2.4.0 finds them
#    (HfApi._prepare_upload_folder_additions): resolve the folder, sorted(glob("**/*")),
#    keep is_file(), relative POSIX paths, then drop excluded paths. Matching normalises
#    "\" to "/" first, as huggingface_hub's filter_repo_objects does.
def normalise(value):
    return value.replace("\\", "/")


ignore = [normalise(p) for p in UPLOAD_EXCLUDES + DEFAULT_IGNORE_PATTERNS]
try:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        block(f"'{folder}' is not a directory")
    on_disk = [p.relative_to(root).as_posix() for p in sorted(root.glob("**/*")) if p.is_file()]
except OSError as error:
    block(f"could not read '{folder}' ({type(error).__name__})")
uploaded = {rel for rel in on_disk if not any(fnmatchcase(normalise(rel), p) for p in ignore)}

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
