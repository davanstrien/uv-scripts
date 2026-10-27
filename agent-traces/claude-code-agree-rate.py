# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""How often do you take your coding agent's recommendation?

Claude Code can pause mid-task and ask you a multiple-choice question (the
AskUserQuestion tool), often marking one option "(Recommended)". This script
scans your local Claude Code logs and reports how often you picked the
recommended option when one was offered.

Runs on your own machine over ~/.claude/projects — no Hub, no GPU, no network.
Read-only; the default output is aggregate counts only.

Specific to Claude Code logs (the AskUserQuestion tool + ~/.claude/projects
JSONL format). Other coding agents record sessions differently.

Run it (nothing to install but uv):
    uv run https://raw.githubusercontent.com/davanstrien/uv-scripts/main/agent-traces/claude-code-agree-rate.py
    uv run claude-code-agree-rate.py --examples 20      # show each answered case
    uv run claude-code-agree-rate.py --json             # machine-readable
    uv run claude-code-agree-rate.py --path ~/some/dir  # scan a different log dir

Privacy: --examples prints the text of your questions and answers, which can
include private project details. The default (aggregate) output is safe to
share; --examples output is not.
"""
import argparse
import glob
import hashlib
import json
import os
import re

PAIR = re.compile(r'"([^"]*?)"\s*=\s*"([^"]*?)"')  # "question"="answer"
ANSWERED = "have been answered"

# Only these lines can matter: the AskUserQuestion tool_use and its answered
# tool_result. Gating on raw bytes before json.loads skips the ~99% of lines
# (and whole sessions) that never touch the tool — a >2x speedup on large logs.
_FILE_GATE = b"AskUserQuestion"
_LINE_GATE = (b"AskUserQuestion", b"have been answered", b"doesn't want to proceed")


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower().replace("(recommended)", "")).strip()


def iter_objs(path):
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return
    if _FILE_GATE not in raw:  # whole file irrelevant — skip without parsing
        return
    for line in raw.splitlines():
        if not any(g in line for g in _LINE_GATE):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def blocks(obj):
    content = (obj.get("message") or {}).get("content")
    return content if isinstance(content, list) else []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=os.path.expanduser("~/.claude/projects"),
                        help="directory of Claude Code .jsonl logs (default: ~/.claude/projects)")
    parser.add_argument("--examples", type=int, default=0,
                        help="show N answered cases. WARNING: prints your question/answer text — do not share")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.path, "**", "*.jsonl"), recursive=True)

    seen = set()
    followed = other_listed = free_text = skipped = 0
    total_calls = answered_calls = 0
    examples = []

    for f in files:
        objs = list(iter_objs(f))
        # map AskUserQuestion tool_use id -> its questions
        questions_by_id = {}
        for o in objs:
            for b in blocks(o):
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "AskUserQuestion":
                    questions_by_id[b.get("id")] = (b.get("input") or {}).get("questions", [])
        # map tool_use id -> answer string
        result_by_id = {}
        for o in objs:
            for b in blocks(o):
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in questions_by_id:
                    c = b.get("content")
                    result_by_id[b["tool_use_id"]] = c if isinstance(c, str) else json.dumps(c)

        for tid, questions in questions_by_id.items():
            total_calls += 1
            result = result_by_id.get(tid, "")
            is_answered = ANSWERED in result and bool(PAIR.search(result))
            if is_answered:
                answered_calls += 1
            ans = {norm(q): a for q, a in PAIR.findall(result)} if is_answered else {}
            for q in questions:
                rec = next((o["label"] for o in q.get("options", [])
                            if "(recommended)" in (o.get("label") or "").lower()), None)
                if not rec:
                    continue
                qtext = q.get("question", "")
                chosen = ans.get(norm(qtext))
                key = hashlib.md5((norm(qtext) + "|" + norm(rec) + "|" + norm(str(chosen))).encode()).hexdigest()
                if key in seen:  # dedup resumed-session duplicates
                    continue
                seen.add(key)
                if not is_answered or chosen is None:
                    skipped += 1
                    continue
                labels = {norm(o.get("label", "")) for o in q.get("options", [])}
                multi = q.get("multiSelect")
                if norm(chosen) == norm(rec) or (multi and norm(rec) and norm(rec) in norm(chosen)):
                    followed += 1
                    tag = "followed-rec"
                elif norm(chosen) in labels:
                    other_listed += 1
                    tag = "other-listed"
                else:
                    free_text += 1
                    tag = "free-text"
                examples.append({"tag": tag, "recommended": rec, "chosen": chosen, "question": qtext})

    answered = followed + other_listed + free_text
    summary = {
        "files_scanned": len(files),
        "askuserquestion_calls": total_calls,
        "answered_calls": answered_calls,
        "rec_questions_answered": answered,
        "rec_questions_skipped": skipped,
        "followed_recommendation": followed,
        "chose_other_listed": other_listed,
        "free_text_other": free_text,
        "rate": round(followed / answered, 3) if answered else None,
    }

    if args.json:
        print(json.dumps({"summary": summary, "examples": examples}, indent=2))
        return

    if answered:
        print(f"You took the recommended option {followed / answered:.0%} of the time ({followed}/{answered}).\n")
    else:
        print("No answered questions with a (Recommended) option found.\n")
    print(f"files scanned                         : {summary['files_scanned']}")
    print(f"AskUserQuestion calls                 : {total_calls}  ({answered_calls} answered)")
    print("-" * 56)
    print(f"questions with a (Recommended) option : {answered + skipped}")
    print(f"  answered                            : {answered}")
    print(f"  skipped / rejected                  : {skipped}")
    print(f"    -> followed recommendation        : {followed}")
    print(f"    -> chose a different listed option: {other_listed}")
    print(f"    -> free-text / Other              : {free_text}")
    if args.examples:
        print("\nexamples (your text — do not share):")
        for e in examples[: args.examples]:
            mark = "Y" if e["tag"] == "followed-rec" else "n"
            print(f"  [{mark}] {e['recommended'][:34]!r} -> {str(e['chosen'])[:34]!r}")
            print(f"      {e['question'][:90]}")


if __name__ == "__main__":
    main()
