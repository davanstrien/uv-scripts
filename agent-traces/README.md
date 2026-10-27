---
viewer: false
tags:
  - uv-script
  - agent-traces
  - claude-code
  - introspection
license: apache-2.0
---

# Agent Traces

Tools for working with **agent traces** — the logs coding agents leave behind. (`agent-traces` is also a [Hugging Face dataset format](https://huggingface.co/datasets?format=format:agent-traces); the scripts here analyse your *local* traces today and can grow toward converting them to that Hub format.)

Unlike the rest of this repo, these are **local-introspection** scripts: they read logs on your own machine, not the Hub, so they run with plain `uv run` — no Jobs, no GPU, no token.

## Available scripts

### claude-code-agree-rate.py

**How often do you take your coding agent's recommendation?**

Claude Code can pause mid-task and ask you a multiple-choice question (the `AskUserQuestion` tool), often marking one option **"(Recommended)"**. This script scans your local Claude Code logs (`~/.claude/projects`) and reports how often you picked the recommended option when one was offered.

Specific to Claude Code's log format — other coding agents record sessions differently, so an equivalent for another tool would be its own script in this folder.

Run it (nothing to install but [uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
uv run https://raw.githubusercontent.com/davanstrien/uv-scripts/main/agent-traces/claude-code-agree-rate.py
```

Example output:

```
You took the recommended option 72% of the time (33/46).

files scanned                         : 485
AskUserQuestion calls                 : 238  (207 answered)
--------------------------------------------------------
questions with a (Recommended) option : 53
  answered                            : 46
  skipped / rejected                  : 7
    -> followed recommendation        : 33
    -> chose a different listed option: 4
    -> free-text / Other              : 9
```

The breakdown matters more than the headline: "didn't follow" splits into *picked a different option* (a real override) and *answered in free text* (often just asking a follow-up question, not overriding).

Flags:

- `--examples N` — show N answered cases (recommended option → what you chose)
- `--json` — machine-readable output
- `--path DIR` — scan a different log directory (default `~/.claude/projects`)

**Privacy:** the default output is aggregate counts only and is safe to share. `--examples` prints the text of your questions and answers, which can include private project details — don't paste that publicly.

### Notes / caveats

- Relies on the `(Recommended)` label convention — a soft string, not a structured field — so it only catches recommendations marked that way.
- Counts only questions you actually answered; skipped or clarified prompts are reported separately, not in the rate.
- Stdlib only — no dependencies, no network. A raw-bytes pre-filter skips sessions that never call the tool, so it stays fast even over gigabytes of logs.
