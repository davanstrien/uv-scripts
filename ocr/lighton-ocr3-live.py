# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.3.1",
#     "pillow>=10",
#     # LightOn's official client, pinned to a commit (not on PyPI). Used for its page
#     # resize, grounding parser and dollar-delimiter repair so outputs match upstream.
#     "lightonocr @ https://github.com/lightonai/LightOnOCR/archive/36755d461be079737860a5f03ae0c803501269e9.tar.gz",
# ]
#
# [tool.hf-jobs]
# image = "vllm/vllm-openai:v0.30.0"
# flavor = "rtx-pro-6000"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
EXPERIMENTAL: lighton-ocr3-saturate.py with a live view you can watch (and screen-record) while the Job runs.

Same OCR, flags and output as lighton-ocr3-saturate.py; use that script for normal runs and
read its description for the model settings, the measured speeds and the repair pass. This
one adds `--live-port`: a page served from inside the Job that shows every finished page
moving past at the speed the model reads them, the pages read a second, and the text the
model wrote for one of them.

  hf jobs uv run --detach --expose-public 7860 \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr3-live.py \\
      <input-dataset> <output-dataset> --live-port 7860

The Job log prints the link: https://<job id>--7860.hf.jobs (public while the Job runs).
The header picks rtx-pro-6000, the fastest flavor per dollar in lighton-ocr3-saturate.py's
table (about 10 pages per second with the 0.8B); add --flavor a10g-small for a small run.

What the page shows, and how it is paced:
- Each tile on the belt is one finished page. A page is shown only after its text came
  back, and the belt moves at the rate the pages finish, so its speed is the throughput.
  The page draws a finished page about 1.5 seconds after the model returned it.
- "Pages a second" is the number of pages finished in the last 10 seconds, divided by 10.
  It covers reading only, not the server start (about 4 minutes). After the run it is the
  average over the whole reading time.
- The text under the belt is one recent finished page's text, changed every few seconds.
  It is a sample: at several pages a second most pages are never shown with their text.
- Add ?view=single to the link for one page at a time: each finished page is laid on the
  last at the same pace, with that page's text beside it.
- The live view costs a 480 px JPEG per page in the preparation threads and one small
  request per open tab every half second. It never changes what is sent or written.
- The page shows the main pass. Pages re-sent by --repair-max-tokens are not shown again.
- The Job, and with it the page, ends 20 seconds after the run. An open tab keeps the
  final numbers on screen.

The SERVING dict below is the per-model tuning prior, as in lighton-ocr3-saturate.py.
"""

import argparse
import base64
import collections
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Serving starting values for LightOnOCR-3 (0.8B and 4B). Per-value provenance:
# - image: vLLM 0.30.0, the version LightOn's client pins and tests (0.27.1 makes the
#   models loop, per the LightOnOCR README).
# - context 18000 and serve_args: the LightOnOCR README "Models" table for the 0.8B and 4B,
#   plus its `--limit-mm-per-prompt '{"image": 1}'`.
# - max_tokens/temperature/top_p: the model card's vLLM example, verbatim.
# - longest_edge 2048: the card's rendering tip and the client's page size for these models.
# - patch 32: 16 px patches with a 2x2 merge, so a 2048x2048 page is at most 64x64 = 4,096
#   image tokens (used for the context check).
# - --max-num-batched-tokens 16384: speed only. vLLM 0.30 serves with 2048 on A10G, L4 and
#   A100 (8192 on other GPUs of 70 GB or more), less than one page's image tokens. Measured on a100-large:
#   6.20 and 6.30 pages/s against 5.83 without it, outputs within the run-to-run noise.
# - pages_per_gib: see window_cap().
# - api_servers 4 when CPU_CORES >= 12: speed only. With one vLLM API server, a100-large and
#   rtx-pro-6000 had 128-256 pages in flight but only 36-60 running in the engine and none
#   waiting: the rest queued for image preprocessing in the API process. Measured, 4 API
#   servers against one: a100-large 7.29 and 7.18 against 6.20 and 6.30 pages/s,
#   rtx-pro-6000 10.0 against 9.15, a10g-large 2.74 against 2.49; outputs within the
#   run-to-run noise. Small flavors (a10g-small 3 cores, l4x1 8) keep one, to leave CPU
#   for page preparation.
SERVING = {
    "models": ["lightonai/LightOnOCR-3-0.8B", "lightonai/LightOnOCR-3-4B"],
    "image": "vllm/vllm-openai:v0.30.0",
    "context": 18000,
    "serve_args": [
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
        "--max-num-batched-tokens", "16384",
    ],
    "pages_per_gib": 6,
    "api_servers": 4,
    "api_servers_min_cores": 12,
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
    "repair_temperature": 0.1,
    "longest_edge": 2048,
    "patch": 32,
}


def window_cap() -> int:
    """Most pages in flight that this machine's memory allows.

    Every page in flight holds about 0.1 GiB of image data on the host (the client's copy and
    the server's preprocessed image), on top of 7 to 9 GiB for the engine itself. On
    a10g-small (14.9 GiB limit), 64 pages in flight reached the limit and the Job stopped
    making progress; 32 peaked at 12.0 GiB, and up to 39 peaked at 13.0 and 14.8 GiB on two
    runs. So the window may grow by SERVING "pages_per_gib" pages per GiB above 10 GiB: 29 on
    a10g-small (which is GPU-bound from about 20 pages in flight), about 100 on l4x1, the full
    512 on a100-large. The limit is read from the container (cgroup v2), then from the MEMORY
    variable that HF Jobs sets.
    """
    limit_gib = None
    try:
        with open("/sys/fs/cgroup/memory.max") as handle:
            value = handle.read().strip()
        if value != "max":
            limit_gib = int(value) / 2**30
    except (OSError, ValueError):
        pass
    if limit_gib is None and os.environ.get("MEMORY"):
        try:  # e.g. "16.0G" (decimal gigabytes)
            limit_gib = float(os.environ["MEMORY"].upper().rstrip("GIB ")) * 1e9 / 2**30
        except ValueError:
            pass
    if limit_gib is None:
        return 512
    return max(8, min(512, int((limit_gib - 10) * SERVING["pages_per_gib"])))


def flag_value(tokens: list[str], name: str):
    """Last value given for `name` in a `vllm serve` argument list (`--x v` or `--x=v`), else None."""
    value = None
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            value = tokens[index + 1]
        elif token.startswith(name + "="):
            value = token.split("=", 1)[1]
    return value


def saved_truncations(output_dataset: str) -> set:
    """Ids whose saved result in data/ stopped at max_tokens and that have no saved repair.

    A restarted run skips rows already in data/, so without this the pages truncated
    before the restart would never reach the repair pass. Reads only three columns.
    """
    import fsspec
    import pyarrow.parquet as pq

    def records(subdir, columns):
        fs, root = fsspec.core.url_to_fs(f"hf://datasets/{output_dataset}/{subdir}")
        if not fs.exists(root):
            return
        for path in fs.glob(f"{root}/part-*.parquet"):
            with fs.open(path, "rb") as handle:
                yield from pq.read_table(handle, columns=columns).to_pylist()

    finish = {}
    for record in records("data", ["id", "error", "finish_reason"]):
        if record["error"] is None:
            finish[record["id"]] = record["finish_reason"]
    repaired = {r["id"] for r in records("repair", ["id", "error"]) if r["error"] is None}
    return {row_id for row_id, reason in finish.items() if reason == "length" and row_id not in repaired}


def max_image_tokens(longest_edge: int) -> int:
    """Upper bound on the image tokens of a page fitted to longest_edge (square page)."""
    side = -(-longest_edge // SERVING["patch"])
    return side * side


def to_pil(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"])
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    raise ValueError(f"unsupported image value: {type(value)}")


def page_png(value, longest_edge: int) -> bytes:
    """RGB page fitted to longest_edge with LightOn's resize, as PNG bytes.

    compress_level=1 instead of Pillow's default 6: PNG is lossless at every level, so the
    pixels are identical, and level 1 encodes several times faster (CPU is the limit on
    small flavors).
    """
    return png_bytes(fitted_page(value, longest_edge))


def fitted_page(value, longest_edge: int):
    """RGB page fitted to longest_edge with LightOn's resize."""
    from lightonocr import fit

    return fit(to_pil(value).convert("RGB"), longest_edge)


def png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def chat_request(model: str, png: bytes, mode: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    """One page as the client sends it: the image, plus the word "grounding" in grounding mode."""
    from lightonocr import GROUNDING_PROMPT

    url = "data:image/png;base64," + base64.b64encode(png).decode()
    content = [{"type": "image_url", "image_url": {"url": url}}]
    if mode == "grounding":
        content.append({"type": "text", "text": GROUNDING_PROMPT})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def postprocess(raw: str, mode: str, fix_dollars: bool):
    """Return (markdown, layout_blocks JSON or None) for one model output, as lighton-ocr3.py."""
    from lightonocr import fix_escaped_dollars, parse_blocks, to_markdown

    text = fix_escaped_dollars(raw) if fix_dollars else raw
    if mode != "grounding":
        return text, None
    blocks = parse_blocks(text)
    if not blocks:
        return text, json.dumps([])  # no markers parsed: keep the text so nothing is lost
    return to_markdown(blocks), json.dumps([b.to_dict() for b in blocks], ensure_ascii=False)


def count_pages(dataset: str, config, split: str, limit):
    """Number of rows the run will read, from the dataset's metadata (None if it does not say)."""
    try:
        from datasets import load_dataset_builder

        total = load_dataset_builder(dataset, name=config).info.splits[split].num_examples
    except Exception as e:  # noqa: BLE001 - the count is only shown; the run must not depend on it
        print(f"Live view: page count unknown ({type(e).__name__})", file=sys.stderr)
        return limit
    return min(total, limit) if limit else total


LIVE_PAGE = """<!doctype html><meta charset="utf-8"><title>LightOnOCR-3 live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
/* A copy stand: a neutral ground, and the paper is the only colour on it. */
:root{--ground:#2b2f34;--ink:#f4f2ed;--quiet:#a4aab1;--rule:rgba(244,242,237,.22);--paper:#fbfaf7;
 --lift:0 0 0 1px rgba(0,0,0,.10),0 .7vmin 2vmin rgba(0,0,0,.34)}
@media (prefers-color-scheme:light){:root{--ground:#c9cdd1;--ink:#15171a;--quiet:#4a4f56;--rule:rgba(21,23,26,.22);
 --lift:0 0 0 1px rgba(0,0,0,.07),0 .5vmin 1.6vmin rgba(0,0,0,.20)}}
*{box-sizing:border-box}
html{font-size:clamp(15px,2.1vmin,24px)}
body{margin:0;height:100vh;overflow:hidden;background:var(--ground);color:var(--ink);
 font-family:"Avenir Next","Helvetica Neue",system-ui,sans-serif;line-height:1.35;
 display:grid;grid-template-rows:minmax(0,1fr) auto}
#belt{position:relative;display:flex;flex-direction:column;gap:1.7vmin;padding:2.2vmin 0;min-height:0;overflow:hidden}
.row{position:relative;flex:1;min-height:0}
.track{position:absolute;inset:0;will-change:transform}
.track img{position:absolute;top:0;height:100%;object-fit:cover;object-position:top;background:var(--paper);
 border-radius:2px;box-shadow:var(--lift)}
#progress{height:2px;background:var(--rule)}
#progress div{height:2px;width:0;background:var(--ink)}
#band{display:grid;grid-template-columns:minmax(0,1fr);gap:1.1rem;padding:1.5rem 4vmin 1.9rem}
#rate{font-size:2.55rem;font-weight:600;line-height:1.08;letter-spacing:-.012em;margin:0 0 .45rem;
 font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#run,#count{margin:0}
#count{color:var(--quiet);font-variant-numeric:tabular-nums}
#sample{margin:0;min-width:0;transition:opacity .4s}
#sample.out{opacity:0}
#text{margin:0;font-family:"Iowan Old Style",Charter,Georgia,serif;font-size:1.02rem;line-height:1.5;height:4.5em;
 overflow:hidden;white-space:pre-line}
#from{margin:.3rem 0 0;color:var(--quiet);font-size:.82rem}
@media (min-aspect-ratio:13/10){#band{grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:5vmin;align-items:end}}
/* ?view=single: one page at a time, each laid on the last, at the same pace */
.single #belt{display:block;padding:0}
#pile{position:absolute;inset:3vmin 0 2.4vmin}
#pile img{position:absolute;left:50%;top:0;height:100%;aspect-ratio:.74;object-fit:cover;object-position:top;
 background:var(--paper);border-radius:3px;box-shadow:var(--lift)}
.single #text{height:10.5em}
@media (min-aspect-ratio:13/10){
 body.single{grid-template-columns:minmax(0,9fr) minmax(0,11fr);grid-template-rows:minmax(0,1fr) auto}
 .single #belt{grid-row:1;grid-column:1}.single #progress{grid-row:2;grid-column:1/-1}
 .single #band{grid-row:1;grid-column:2;grid-template-columns:minmax(0,1fr);align-content:center;gap:2.2rem;padding-left:1vmin}
 .single #text{height:12em}}
</style>
<section id="belt" aria-label="Pages the model has read"></section>
<div id="progress"><div id="bar"></div></div>
<section id="band">
 <div><p id="rate">Starting</p><p id="run"></p><p id="count"></p></div>
 <figure id="sample" class="out"><p id="text"></p><figcaption id="from"></figcaption></figure>
</section>
<script>
const $ = id => document.getElementById(id), fmt = n => Math.round(n).toLocaleString("en-US");
const clock = s => { s = Number.isFinite(s) ? Math.max(0, Math.round(s)) : 0;
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const NL = String.fromCharCode(10);
const LAG = 1.5;  // seconds a finished page waits before it is drawn, so the belt moves evenly
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;  // then the belt moves in steps
const single = new URLSearchParams(location.search).get("view") === "single";
document.body.classList.toggle("single", single);
let since = 0, last = null, misses = 0, first = true, stopped = false;
let rows = [], tileW = 0, pitch = 0, beltW = 0, offset = 0, drawn = 0, speed = 0;
const waiting = [], ready = [];  // finished pages: image still loading, then loaded; oldest first
let shown = 0, shownWords = 0, recent = [];

function layout() {
  if (single) { const pile = document.createElement("div"); pile.id = "pile"; $("belt").replaceChildren(pile); return; }
  const belt = $("belt"), vmin = Math.min(innerWidth, innerHeight), gap = vmin * 0.017;
  const h = belt.clientHeight - vmin * 0.044, n = Math.max(2, Math.min(5, Math.round(h / (vmin * 0.26))));
  const rowH = (h - gap * (n - 1)) / n;
  tileW = Math.round(rowH * 0.74); pitch = tileW + gap; beltW = belt.clientWidth; offset = drawn = 0;
  belt.replaceChildren();
  rows = Array.from({length: n}, () => {
    const row = document.createElement("div"), track = document.createElement("div");
    row.className = "row"; track.className = "track"; row.append(track); belt.append(row);
    return {track, tail: 0, tiles: []};
  });
}

// One finished page becomes one tile, placed at the end of the shortest row.
function place(p) {
  shown += 1; shownWords += p.words;
  if (p.missing) return;
  const row = rows.reduce((a, b) => b.tail < a.tail ? b : a), x = Math.max(row.tail, offset + beltW);
  p.el.style.left = x + "px"; p.el.style.width = tileW + "px";
  row.track.append(p.el); row.tiles.push({x, el: p.el}); row.tail = x + pitch;
  recent.push(p); if (recent.length > 40) recent.shift();
}

// Single view: the page goes on top of the pile, with a small turn so earlier pages show at the edges.
let credit = 0;
function deal(p) {
  shown += 1; shownWords += p.words;
  if (p.missing) return;
  const turn = still ? 0 : (Math.random() - 0.5) * 2.4, dx = (Math.random() - 0.5) * 1.4, dy = (Math.random() - 0.5) * 0.9;
  p.el.style.transform = `translate(calc(-50% + ${dx}vmin), ${dy}vmin) rotate(${turn}deg)`;
  const pile = $("pile"); pile.append(p.el);
  while (pile.children.length > 10) pile.firstChild.remove();
  recent.push(p); if (recent.length > 40) recent.shift();
  write(p);
}

// The belt moves so that the pages waiting to be drawn clear in LAG seconds: while pages
// finish at a steady rate that is the rate itself, and when none are left it stops.
let before = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - before) / 1000); before = now;
  const want = ready.length ? Math.max(1, ready.length / LAG) : 0;
  const target = rows.length ? want * pitch / rows.length : 0;
  if (single) {  // the same pace, counted in pages instead of distance
    speed += (want - speed) * Math.min(1, dt * 2.5); credit = Math.min(3, credit + speed * dt);
    while (ready.length && (ready[0].missing || credit >= 1)) { if (!ready[0].missing) credit -= 1; deal(ready.shift()); }
    words(); requestAnimationFrame(frame); return;
  }
  speed += (target - speed) * Math.min(1, dt * 2.5);
  offset += speed * dt;
  while (ready.length && (ready[0].missing || Math.min(...rows.map(r => r.tail)) <= offset + beltW)) place(ready.shift());
  if (!still || offset - drawn >= pitch) drawn = offset;
  for (const r of rows) {
    r.track.style.transform = `translateX(${-drawn}px)`;
    while (r.tiles.length && r.tiles[0].x + pitch < drawn) r.tiles.shift().el.remove();
  }
  words();
  requestAnimationFrame(frame);
}

// The three lines under the belt, as sentences.
function words() {
  const s = last; if (!s) return;
  const all = s.finished && !ready.length && !waiting.length, done = all ? s.pages_done : shown;
  const of = s.pages_total ? ` of ${fmt(s.pages_total)}` : "";
  if (s.pages_total) $("bar").style.width = Math.min(100, 100 * done / s.pages_total) + "%";
  $("run").textContent = s.model.split("/").pop() + (s.gpu_name ? ` on one ${s.gpu_name}, ` : " on ") + "Hugging Face Jobs";
  if (!s.reading) {
    $("rate").textContent = s.phase;
    $("count").textContent = `Started ${clock(s.uptime_s)} ago` + (s.pages_total ? `, ${fmt(s.pages_total)} pages to read` : "");
  } else {
    const rate = s.finished ? s.pages_done / Math.max(1, s.reading_s) : s.rate_pps;
    $("rate").textContent = rate ? `${rate.toFixed(1)} pages a second` : "Reading the first pages";
    $("count").textContent = stopped ? "The Job stopped. Check the Job logs."
      : all ? `All ${fmt(done)} pages read in ${clock(s.reading_s)}`
      : `${fmt(done)}${of} pages read in ${clock(s.reading_s)}`;
  }
}

function ingest(d) {
  const s = d.stats; last = s;
  document.title = s.model.split("/").pop() + " live";
  // A late open or a reload starts from the newest pages, not the whole history.
  let pages = d.pages;
  if (first) { pages = pages.slice(-12); first = false;
    shown = s.pages_done - pages.length; shownWords = s.words - pages.reduce((a, p) => a + p.words, 0); }
  for (const p of d.pages) since = Math.max(since, p.seq);
  for (const p of pages) {
    const el = new Image(); p.el = el; el.alt = "";
    el.onload = () => { p.loaded = true; flush(); };
    el.onerror = () => {  // one retry, then the page is counted without a tile
      if (p.retried) { p.missing = true; p.loaded = true; flush(); }
      else { p.retried = true; setTimeout(() => { el.src = "img/" + p.img + "?retry"; }, 400); }
    };
    el.src = "img/" + p.img; waiting.push(p);
  }
}

// Pages enter the belt in the order they finished, each once its image has loaded.
function flush() { while (waiting.length && waiting[0].loaded) ready.push(waiting.shift()); }

// A page's text as the model wrote it, without table and heading markup.
function write(p) {
  const plain = (p.text || "").replace(/<[^>]+>/g, " ").replace(/^#+ /gm, "").replaceAll("**", "")
    .replace(/ {2,}/g, " ").replace(new RegExp("( ?" + NL + " ?){2,}", "g"), NL).trim();
  $("text").textContent = plain || "(no text on this page)";
  $("from").textContent = single ? "The text of the page on top" : "The text of one of these pages";
  $("sample").classList.remove("out");
}

// Belt view: one recent page's text, changed every few seconds. (Single view writes every page.)
let sampled = null;
function sample() {
  if (single) return;
  const withText = recent.filter(p => (p.text || "").length > 200), p = (withText.length ? withText : recent).at(-1);
  if (p && p !== sampled) {
    sampled = p; $("sample").classList.add("out");
    setTimeout(() => write(p), 400);
  }
  setTimeout(sample, 4000);
}

async function poll() {
  try {
    ingest(await (await fetch("feed.json?since=" + since)).json()); misses = 0;
  } catch (e) {
    misses += 1;
    if (misses >= 6 && last && !last.finished) stopped = true;
  }
  setTimeout(poll, 500);
}
layout(); addEventListener("resize", layout);
poll(); sample(); requestAnimationFrame(frame);
</script>"""


class LiveView:
    """Finished pages and run counters, served as a web page from inside the Job (`--live-port`)."""

    def __init__(self, port: int, model: str, pages_total, keep: int = 240, stale_s: float = 1800):
        self.model, self.pages_total = model, pages_total
        self.pages = collections.deque(maxlen=keep)  # finished pages, newest last
        self.images: dict[str, bytes] = {}  # key -> JPEG, for prepared and kept pages
        self.prepared: dict[str, tuple[str, float]] = {}  # row id -> (image key, time prepared)
        self.stale_s = stale_s  # a prepared page not back after this long failed or was dropped
        self.lock = threading.Lock()
        self.made = self.seq = self.pages_done = self.words = 0
        self.booted = time.time()
        self.started = None  # first page prepared: the rate and the clock leave out the server start
        self.finished = None
        self.done_times = collections.deque()  # when each recent page finished, for the rolling rate
        self.gpu_name = None
        live = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path.startswith("/feed.json"):
                        try:
                            since = int(self.path.partition("since=")[2] or 0)
                        except ValueError:
                            since = 0
                        self.reply(json.dumps(live.feed(since)).encode(), "application/json")
                    elif self.path.startswith("/img/"):
                        with live.lock:
                            jpeg = live.images.get(self.path[len("/img/"):].partition("?")[0])
                        if jpeg is None:
                            self.send_error(404)
                        else:
                            self.reply(jpeg, "image/jpeg", cache="max-age=86400")
                    else:
                        self.reply(LIVE_PAGE.encode(), "text/html; charset=utf-8")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the viewer closed the tab mid-reply

            def reply(self, body: bytes, ctype: str, cache: str = "no-store"):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", cache)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep the Job log for the OCR run
                pass

        class Server(ThreadingHTTPServer):
            request_queue_size = 128  # a tab asks for several page images at once

        server = Server(("0.0.0.0", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Thread(target=self.name_gpu, daemon=True).start()
        # An exposed Job port is served at https://<job id>--<port>.hf.jobs.
        where = (f"https://{os.environ['JOB_ID']}--{port}.hf.jobs (needs --expose-public {port})"
                 if os.environ.get("JOB_ID") else f"http://localhost:{port}")
        print(f"Live view: {where}", file=sys.stderr, flush=True)

    def name_gpu(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10, check=True).stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            return
        if len(out) == 1:  # the page says "one <GPU>": leave it out on a multi-GPU flavor
            with self.lock:
                self.gpu_name = out[0].strip()

    def add_thumbnail(self, row_id: str, image):
        thumb = image.copy()
        thumb.thumbnail((480, 480))
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=78)
        with self.lock:
            if self.started is None:
                self.started = time.time()
            self.made += 1
            key = f"{self.made}.jpg"
            self.images[key] = buffer.getvalue()
            old = self.prepared.get(row_id)
            if old:  # prepared again for a retry: drop the earlier copy
                self.images.pop(old[0], None)
            self.prepared[row_id] = (key, time.time())

    def add_page(self, row_id: str, text: str):
        now = time.time()
        tail = row_id.rsplit(":", 1)[-1]
        with self.lock:
            key, _ = self.prepared.pop(row_id, ("", now))
            if len(self.pages) == self.pages.maxlen:  # the oldest kept page leaves: drop its image
                self.images.pop(self.pages[0]["img"], None)
            words = len(text.split())
            self.seq += 1
            self.pages_done += 1
            self.words += words
            self.done_times.append(now)
            self.pages.append({"seq": self.seq, "img": key, "text": text[:700], "words": words,
                               "n": int(tail) + 1 if tail.isdigit() else None})

    def finish(self):
        with self.lock:
            self.finished = time.time()

    def feed(self, since: int) -> dict:
        now = time.time()
        with self.lock:
            for row_id, (key, t) in list(self.prepared.items()):
                if now - t > self.stale_s:
                    del self.prepared[row_id]
                    self.images.pop(key, None)
            while self.done_times and now - self.done_times[0] > 10:
                self.done_times.popleft()
            end = self.finished or now
            reading_s = end - self.started if self.started else 0
            rate = len(self.done_times) / 10 if reading_s >= 10 and not self.finished else 0
            phase = "Finished" if self.finished else "Reading pages" if self.pages_done else \
                "Preparing the first pages" if self.started else "Starting the model server"
            return {"stats": {"model": self.model, "gpu_name": self.gpu_name, "phase": phase,
                              "reading": self.pages_done > 0, "finished": self.finished is not None,
                              "uptime_s": round(now - self.booted), "reading_s": round(reading_s),
                              "pages_total": self.pages_total, "pages_done": self.pages_done,
                              "words": self.words, "rate_pps": round(rate, 1)},
                    "pages": [p for p in self.pages if p["seq"] > since][-120:]}


def main():
    ap = argparse.ArgumentParser(description="LightOnOCR-3 batch OCR via saturate, with a live view")
    ap.add_argument("input_dataset", help="Input dataset repo id (rows with an image column)")
    ap.add_argument("output_dataset", help="Output dataset repo id (created private if missing)")
    ap.add_argument("--model", default=SERVING["models"][0], choices=SERVING["models"])
    ap.add_argument("--mode", default="plain", choices=["plain", "grounding"],
                    help="plain = markdown; grounding = labelled layout blocks with 0-1000 boxes")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--max-samples", "--limit", dest="limit", type=int, default=None,
                    help="Process only the first N rows (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"],
                    help="Output cap per page (default 4096, the model card's)")
    ap.add_argument("--temperature", type=float, default=SERVING["temperature"])
    ap.add_argument("--top-p", type=float, default=SERVING["top_p"],
                    help="Default 0.9, the model card's; LightOn's client uses 1.0")
    ap.add_argument("--longest-edge", type=int, default=SERVING["longest_edge"],
                    help="Downscale pages to at most this many pixels on the longest edge")
    ap.add_argument("--no-fix-dollars", action="store_true",
                    help="Keep the model's text in `markdown` without repairing escaped dollars")
    ap.add_argument("--repair-max-tokens", type=int, default=0,
                    help="Re-send pages that hit --max-tokens once, with this cap and "
                         "temperature 0.1, into repair/ (default 0: off; 8192 suggested)")
    ap.add_argument("--window", default="auto",
                    help="Pages in flight: 'auto' (saturate adapts it) or a fixed number")
    ap.add_argument("--prepare-workers", type=int, default=None,
                    help="Threads that decode, resize and encode pages (default: CPU_CORES)")
    ap.add_argument("--serve-args", default="",
                    help="Extra `vllm serve` flags, one quoted string, e.g. "
                         "\"--max-num-batched-tokens 16384\"")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    ap.add_argument("--live-port", type=int, default=None,
                    help="Serve the live view on this port (on Jobs, also pass --expose-public "
                         "with the same port)")
    args = ap.parse_args()

    user_serve_args = shlex.split(args.serve_args)
    if args.longest_edge < 1:
        ap.error("--longest-edge must be at least 1")
    try:  # --serve-args may set its own --max-model-len; vLLM uses the last one given
        context = int(flag_value(user_serve_args, "--max-model-len") or SERVING["context"])
    except ValueError:
        ap.error("--max-model-len in --serve-args must be a whole number of tokens")
    room = context - max_image_tokens(args.longest_edge) - 64
    for flag, value in (("--max-tokens", args.max_tokens), ("--repair-max-tokens", args.repair_max_tokens)):
        if value > room:
            ap.error(f"{flag} {value} can overflow the {context}-token context with a "
                     f"{args.longest_edge} px page; use at most {room}")
    if args.max_tokens < 1:
        ap.error("--max-tokens must be at least 1")
    if args.repair_max_tokens and args.repair_max_tokens <= args.max_tokens:
        ap.error("--repair-max-tokens must be larger than --max-tokens")
    if args.window != "auto" and not (args.window.isdigit() and int(args.window) >= 1):
        ap.error("--window must be 'auto' or a number of at least 1")

    import pyarrow as pa
    from saturate import Auto, Engine, Fixed, dataset_rows, pump

    if args.id_column is None:
        ids = "index"
    elif args.id_column in ("index", "content"):
        # saturate reads these two names as id strategies, so a column with one of them
        # needs a callable. It returns the raw value, so saturate still refuses a null id.
        def ids(example):
            return example[args.id_column]
    else:
        ids = args.id_column

    rows = dataset_rows(
        args.input_dataset, config=args.config, split=args.split,
        columns=[args.image_column], ids=ids, limit=args.limit,
    )

    # Declared, so a part holding only error rows has the same columns as the rest
    # (datasets.load_dataset reads every part with one schema).
    schema = pa.schema([
        ("id", pa.string()), ("error", pa.string()), ("markdown", pa.string()), ("raw", pa.string()),
        ("finish_reason", pa.string()), ("layout_blocks", pa.string()), ("model", pa.string()),
        ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()),
    ])  # fmt: skip

    # Pages that hit --max-tokens, kept on local disk for the repair pass: those truncated
    # in this run (stashed in parse) and those saved as truncated by an earlier, interrupted
    # run (stashed as the stream passes them; the main pass skips them).
    repair_dir = tempfile.mkdtemp(prefix="lighton-ocr3-repair-") if args.repair_max_tokens else None
    earlier = saved_truncations(args.output_dataset) if repair_dir else set()
    truncated = []

    def stash(row_id, image):
        path = os.path.join(repair_dir, hashlib.sha1(row_id.encode()).hexdigest() + ".png")
        with open(path, "wb") as handle:
            handle.write(page_png(image, args.longest_edge))
        truncated.append((row_id, path))

    live = None
    if args.live_port:
        live = LiveView(args.live_port, args.model,
                        count_pages(args.input_dataset, args.config, args.split, args.limit))

    def to_request(row):
        image = fitted_page(row[args.image_column], args.longest_edge)
        if live:
            live.add_thumbnail(row["__id"], image)
        return chat_request(args.model, png_bytes(image), args.mode, args.max_tokens, args.temperature, args.top_p)

    def make_parse(stash_truncated: bool):
        def parse(row, body):
            choice = body["choices"][0]
            raw = (choice["message"].get("content") or "").strip()
            markdown, blocks = postprocess(raw, args.mode, fix_dollars=not args.no_fix_dollars)
            usage = body.get("usage") or {}
            if stash_truncated and choice.get("finish_reason") == "length":
                stash(row["__id"], row[args.image_column])
            if live and "__id" in row:  # repair rows carry no id: the live view shows the main pass
                live.add_page(row["__id"], markdown)
            return {
                "markdown": markdown,
                "raw": raw,
                "finish_reason": choice.get("finish_reason"),
                "layout_blocks": blocks,
                "model": args.model,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        return parse

    def with_ids(source):
        # Keep each row's id on the row, so parse can name the page it stashes for repair.
        for row_id, row in source:
            if row_id in earlier:
                stash(row_id, row[args.image_column])
            yield row_id, {**row, "__id": row_id}

    cap = window_cap()
    if args.window == "auto":
        window = Auto(initial=min(64, cap), max_limit=cap)
        print(f"window: auto, at most {cap} pages in flight (host memory)", file=sys.stderr)
    else:
        window = Fixed(int(args.window))
        if int(args.window) > cap:
            print(f"warning: --window {args.window} is above the {cap} pages this machine's "
                  "memory allows; the Job may run out of memory", file=sys.stderr)
    cores = max(1, int(float(os.environ.get("CPU_CORES") or "4")))
    workers = args.prepare_workers if args.prepare_workers is not None else cores
    extra = [
        "--max-model-len", str(SERVING["context"]),
        *SERVING["serve_args"],
        *user_serve_args,
    ]
    # Several API server processes share the image decoding and preprocessing, which one
    # process cannot keep up with on a fast GPU. Only where there are cores to spare.
    if cores >= SERVING["api_servers_min_cores"] and flag_value(extra, "--api-server-count") is None:
        extra += ["--api-server-count", str(SERVING["api_servers"])]
        print(f"server: --api-server-count {SERVING['api_servers']} ({cores} CPU cores)", file=sys.stderr)

    output = f"hf://datasets/{args.output_dataset}/data"
    with Engine(args.model, engine="vllm", extra_args=extra) as endpoint:
        stats = pump(with_ids(rows), to_request, make_parse(bool(repair_dir)),
                     endpoint, output, window=window, prepare_workers=workers,
                     retry_errors=args.retry_errors, schema=schema)
        print("LIGHTON_OCR3_SATURATE " + stats.to_json(), flush=True)
        if live:
            live.finish()

        if repair_dir and truncated:
            print(f"repair: re-sending {len(truncated)} pages that hit --max-tokens "
                  f"with a cap of {args.repair_max_tokens}", file=sys.stderr, flush=True)

            def repair_rows():
                for row_id, path in truncated:
                    with open(path, "rb") as handle:
                        yield row_id, {"png": handle.read()}

            def repair_request(row):
                return chat_request(args.model, row["png"], args.mode, args.repair_max_tokens,
                                    SERVING["repair_temperature"], args.top_p)

            repair_stats = pump(repair_rows(), repair_request, make_parse(False), endpoint,
                                f"hf://datasets/{args.output_dataset}/repair", window=window,
                                retry_errors=args.retry_errors, schema=schema)
            print("LIGHTON_OCR3_SATURATE_REPAIR " + repair_stats.to_json(), flush=True)

    print(f"https://huggingface.co/datasets/{args.output_dataset} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows, "
          f"{len(truncated) if repair_dir else 'n/a'} truncated pages re-sent to repair/)", file=sys.stderr)
    if live:
        time.sleep(20)  # the Job (and this page) ends with the script: let open tabs fetch the final numbers


if __name__ == "__main__":
    main()
