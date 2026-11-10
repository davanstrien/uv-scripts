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
moving past at the speed the model reads them, pages per second, one finished page with its
text, and the run's counters.

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
- "Pages per second" is the number of pages finished in the last 10 seconds, divided by 10.
  It covers reading only, not the server start (about 4 minutes).
- The page with text is one recent finished page, changed every few seconds. It is a
  sample: at several pages per second most pages are never shown with their text.
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
:root{--bg:#f4f2ee;--fg:#1c1c1e;--muted:#66645f;--card:#fff;--line:#dfdcd5;--accent:#d9622b;--ok:#2f7d5b}
@media (prefers-color-scheme:dark){:root{--bg:#101113;--fg:#ececef;--muted:#9096a0;--card:#191b1f;--line:#2b2e34;--accent:#ff8a3d;--ok:#5cc29a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,sans-serif;height:100vh;
 display:grid;grid-template-rows:auto minmax(0,1fr) auto;overflow:hidden}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;padding:22px 32px 16px;min-width:0}
h1{font-size:30px;line-height:1.15;margin:0;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#sub{font-size:18px;color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#sub.done{color:var(--ok);font-weight:600}#sub.stop{color:var(--accent);font-weight:600}
#sub.busy::before{content:"● ";color:var(--accent);animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
#titles{min-width:0}
#rateBox{text-align:right;flex:none}
#rate{font-size:88px;font-weight:700;line-height:.95;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.label{font-size:14px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
#belt{position:relative;display:flex;flex-direction:column;gap:12px;min-height:0;overflow:hidden;
 border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:12px 0;
 -webkit-mask-image:linear-gradient(90deg,transparent,#000 5%,#000 96%,transparent);
 mask-image:linear-gradient(90deg,transparent,#000 5%,#000 96%,transparent)}
.row{position:relative;flex:1;min-height:0;overflow:hidden}
.track{position:absolute;inset:0;will-change:transform}
.track img{position:absolute;top:0;height:100%;object-fit:cover;object-position:top;background:var(--card);
 border:1px solid var(--line);box-shadow:0 1px 3px rgba(0,0,0,.12)}
#wait{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:22px;color:var(--muted)}
footer{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:32px;padding:18px 32px 22px;min-width:0}
#spot{display:grid;grid-template-columns:auto minmax(0,1fr);gap:20px;height:200px;min-width:0}
#spotImg{height:200px;width:144px;object-fit:cover;object-position:top;border:1px solid var(--line);background:var(--card)}
#spotText{overflow:hidden;white-space:pre-wrap;font:18px/1.5 Georgia,serif;max-width:70ch;height:176px;
 -webkit-mask-image:linear-gradient(#000 70%,transparent);mask-image:linear-gradient(#000 70%,transparent)}
.fade{animation:fade .45s ease-out}@keyframes fade{from{opacity:0}}
#stats{display:grid;grid-template-columns:auto auto;gap:6px 36px;align-content:start}
.mid{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.15}
#wide{grid-column:1/-1}
.bar{height:8px;background:var(--line);border-radius:4px;margin-top:6px}.bar div{height:8px;background:var(--accent);border-radius:4px;width:0}
@media (max-width:900px){header{padding:16px 18px 12px}h1{font-size:22px}#sub{font-size:15px}#rate{font-size:60px}
 footer{grid-template-columns:minmax(0,1fr);gap:16px;padding:14px 18px 18px}#spot{height:150px}#spotImg{height:150px;width:108px}
 #spotText{height:132px;font-size:16px}#stats{grid-template-columns:repeat(3,auto);order:-1}.mid{font-size:24px}}
</style>
<header>
 <div id="titles"><h1 id="model">LightOnOCR-3</h1><div id="sub" class="busy">Starting</div></div>
 <div id="rateBox"><div id="rate">–</div><div class="label">pages per second</div></div>
</header>
<section id="belt"><div id="wait"></div></section>
<footer>
 <div id="spot"><img id="spotImg" alt="" hidden><div><div class="label" id="spotMeta"></div><div id="spotText"></div></div></div>
 <div id="stats">
  <div id="wide"><div class="label">Pages read</div><div class="mid"><span id="done">0</span><span class="label" id="of"></span></div>
   <div class="bar"><div id="bar"></div></div></div>
  <div><div class="label">Words</div><div class="mid" id="words">0</div></div>
  <div><div class="label">Reading for</div><div class="mid" id="elapsed">0:00</div></div>
 </div>
</footer>
<script>
const $ = id => document.getElementById(id), fmt = n => Math.round(n).toLocaleString("en-US");
const clock = s => { s = Number.isFinite(s) ? Math.max(0, Math.round(s)) : 0;
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const LAG = 1.5;  // seconds a finished page waits before it is drawn, so the belt moves evenly
let since = 0, last = null, misses = 0, first = true;
let rows = [], tileW = 0, pitch = 0, beltW = 0, offset = 0, speed = 0;
const ready = [];  // finished pages whose image has loaded, oldest first
let shown = 0, shownWords = 0, recent = [];

function layout() {
  const belt = $("belt"), h = belt.clientHeight - 24, n = h > 560 ? 3 : h > 300 ? 2 : 1, gap = 12;
  const rowH = (h - gap * (n - 1)) / n;
  tileW = Math.round(rowH * 0.7); pitch = tileW + gap; beltW = belt.clientWidth; offset = 0;
  belt.querySelectorAll(".row").forEach(r => r.remove());
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

// The belt moves so that the pages waiting to be drawn clear in LAG seconds: while pages
// finish at a steady rate that is the rate itself, and when none are left it stops.
let before = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - before) / 1000); before = now;
  const want = ready.length ? Math.max(1, ready.length / LAG) : 0;
  const target = rows.length ? want * pitch / rows.length : 0;
  speed += (target - speed) * Math.min(1, dt * 2.5);
  offset += speed * dt;
  while (ready.length && (ready[0].missing || Math.min(...rows.map(r => r.tail)) <= offset + beltW)) place(ready.shift());
  for (const r of rows) {
    r.track.style.transform = `translateX(${-offset}px)`;
    while (r.tiles.length && r.tiles[0].x + pitch < offset) r.tiles.shift().el.remove();
  }
  counters();
  requestAnimationFrame(frame);
}

function counters() {
  if (!last) return;
  const done = last.finished && !ready.length ? last.pages_done : shown;
  $("done").textContent = fmt(done);
  $("words").textContent = fmt(last.finished && !ready.length ? last.words : shownWords);
  if (last.pages_total) { $("of").textContent = ` of ${fmt(last.pages_total)}`;
    $("bar").style.width = Math.min(100, 100 * done / last.pages_total) + "%"; }
}

function ingest(d) {
  const s = d.stats; last = s;
  $("model").textContent = s.model.split("/").pop();
  document.title = s.model.split("/").pop() + " live";
  const where = (s.gpu_name ? `one ${s.gpu_name} GPU on ` : "") + "Hugging Face Jobs";
  const sub = $("sub");
  if (s.finished) { sub.className = "done"; sub.textContent = `Finished · ${fmt(s.pages_done)} pages in ${clock(s.reading_s)} · ${where}`; }
  else if (s.reading) { sub.className = ""; sub.textContent = where; }
  else { sub.className = "busy"; sub.textContent = `${s.phase} · ${clock(s.uptime_s)}`; }
  $("wait").textContent = s.reading ? "" : `${s.phase} …`;
  $("rate").textContent = s.finished ? (s.pages_done / Math.max(1, s.reading_s)).toFixed(1) : s.rate_pps ? s.rate_pps.toFixed(1) : "–";
  $("elapsed").textContent = clock(s.reading_s);
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
const waiting = [];
function flush() { while (waiting.length && waiting[0].loaded) ready.push(waiting.shift()); }

function spotlight() {
  const withText = recent.filter(p => (p.text || "").length > 200), p = (withText.length ? withText : recent).at(-1);
  if (p && p.img !== $("spotImg").dataset.img) {
    const img = $("spotImg"), el = $("spotText");
    img.hidden = false; img.src = "img/" + p.img; img.dataset.img = p.img;
    $("spotMeta").textContent = "One of the pages just read" + (p.n ? ` · row ${fmt(p.n)}` : "");
    el.textContent = (p.text || "(no text on this page)").replace(/^#+ /gm, "");
    for (const x of [img, el]) { x.classList.remove("fade"); void x.offsetWidth; x.classList.add("fade"); }
  }
  setTimeout(spotlight, 3000);
}

async function poll() {
  try {
    ingest(await (await fetch("feed.json?since=" + since)).json()); misses = 0;
  } catch (e) {
    misses += 1;
    if (misses >= 6 && last && !last.finished) { $("sub").className = "stop"; $("sub").textContent = "Job stopped · check the Job logs"; }
  }
  setTimeout(poll, 500);
}
layout(); addEventListener("resize", layout);
poll(); spotlight(); requestAnimationFrame(frame);
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
