#!/usr/bin/env bash
# verify-superset.sh <folder> <hf_repo_id> [repo_type]
#
# Refuse (exit 1) if any file currently on the Hub repo is MISSING from the local
# folder, because hub-sync runs `hf upload --delete="*"` and would DELETE it on sync.
#
# Fails CLOSED on every error mode; passes for a not-yet-created (brand-new) repo.
# Runs both locally (after `git add <folder>`) and as a CI pre-step in the sync workflow.
set -euo pipefail

folder="${1:?usage: verify-superset.sh <folder> <hf_repo_id> [repo_type]}"
repo="${2:?usage: verify-superset.sh <folder> <hf_repo_id> [repo_type]}"
rtype="${3:-dataset}"   # dataset | space | model — selects the /api/<type>s tree endpoint
HF_TOKEN="${HF_TOKEN:-$(hf auth token 2>/dev/null || true)}"
export HF_TOKEN

# List every file on the Hub repo, one path per line.
# Exit codes: 0 = listed, 3 = repo doesn't exist (404 on the first page), 1 = cannot verify.
# It follows the tree's pagination (`Link: <...>; rel="next"`) so large repos are listed
# in full, retries transient failures (429, 5xx, network) with short backoff, and treats
# any response that isn't a JSON list of {type, path} entries as "cannot verify".
# HF_ENDPOINT overrides the Hub URL (used by tests).
list_status=0
hub_raw=$(python3 - "$rtype" "$repo" <<'PY'
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

rtype, repo = sys.argv[1], sys.argv[2]
endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
token = os.environ.get("HF_TOKEN", "")
headers = {"Authorization": f"Bearer {token}"} if token else {}


def block(message):
    print(f"❌ BLOCK — {message}; cannot verify, failing closed.", file=sys.stderr)
    sys.exit(1)


def fetch(url):
    """Return (status, body, link_header). Retries 429, 5xx and network errors."""
    status, body, link = 0, b"", ""
    for attempt in (1, 2, 3):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read(), response.headers.get("Link", "")
        except urllib.error.HTTPError as error:
            status, body, link = error.code, b"", ""
            if status != 429 and not 500 <= status < 600:
                return status, body, link
            reason = f"HTTP {status}"
        except (urllib.error.URLError, TimeoutError) as error:
            status, reason = 0, f"network error ({error})"
        print(f"… Hub API {reason} for {repo}; retry {attempt}/3 after {attempt}s…", file=sys.stderr)
        time.sleep(attempt)
    return status, body, link


url = f"{endpoint}/api/{rtype}s/{repo}/tree/main?recursive=true"
paths = []
page = 0
while url:
    page += 1
    status, body, link = fetch(url)
    if status == 404 and page == 1:
        sys.exit(3)
    if status != 200:
        block(f"Hub API returned HTTP {status} for {repo} (page {page})")
    try:
        entries = json.loads(body)
    except ValueError:
        block(f"Hub API returned a response that is not JSON for {repo} (page {page})")
    if not isinstance(entries, list):
        block(f"Hub API returned {type(entries).__name__}, not a list, for {repo} (page {page})")
    for entry in entries:
        if not isinstance(entry, dict) or "type" not in entry or "path" not in entry:
            block(f"Hub API returned an unexpected tree entry for {repo} (page {page})")
        if entry["type"] == "file":
            paths.append(entry["path"])
    next_link = re.search(r'<([^>]+)>;\s*rel="next"', link)
    url = next_link.group(1) if next_link else None

print("\n".join(paths))
PY
) || list_status=$?

case "$list_status" in
  0) : ;;
  3) echo "✅ OK — $repo doesn't exist yet (brand-new repo); nothing on the Hub to preserve."; exit 0 ;;
  *) echo "❌ BLOCK — could not list the files on $repo; failing closed."; exit 1 ;;
esac

# .git*/.gitattributes are excluded by the action anyway, so don't treat them as "missing".
# grep exits 1 when nothing is left (an empty or .gitattributes-only repo); that is not an error.
hub=$(printf '%s\n' "$hub_raw" | { grep -vE '^(\.git|$)' || true; } | sort)

if [ -z "$hub" ]; then
  echo "✅ OK — $repo has no content files to preserve (empty or .gitattributes-only)."; exit 0
fi

# Tracked files == checked-out tree (CI) or staged tree (local, after `git add`).
local=$(git ls-files "$folder" | sed "s#^$folder/##" | sort)

missing=$(comm -23 <(printf '%s\n' "$hub") <(printf '%s\n' "$local"))
if [ -n "$missing" ]; then
  echo "❌ BLOCK — these Hub files are missing locally and WOULD BE DELETED on sync:"
  echo "$missing"
  exit 1
fi

echo "✅ OK — '$folder' is a superset of $repo ($(printf '%s\n' "$hub" | grep -c .) Hub files all present)."
