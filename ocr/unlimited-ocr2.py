# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.2.0",
#     "pillow>=10",
# ]
#
# [tool.hf-jobs]
# image = "lmsysorg/sglang:v0.5.20-cu130"
# flavor = "rtx-pro-6000"
# timeout = "4h"
# secrets = ["HF_TOKEN"]
# ///
"""
OCR whole documents (book volumes, folders of scans) with Baidu Unlimited-OCR-2.

Unlimited-OCR-2 reads many pages in ONE request and keeps context across page
breaks: a paragraph that runs onto the next page comes back tagged as a
continuation, not as two fragments. This script groups page images into
documents and sends 16 consecutive pages per request.

  hf jobs uv run --detach \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/unlimited-ocr2.py \\
      "hf://buckets/biglam/britannica/source/pages/1771EncyclopediaBritannicaNLS/*.jp2" \\
      <user>/britannica-1771

The [tool.hf-jobs] table above sets the image, GPU, timeout and HF_TOKEN.
The GPU needs >= 48 GB: on 24 GB cards a 16-page request (~31k tokens of KV)
thrashes the cache. rtx-pro-6000 was the fastest per dollar in our tests.

Input: a glob or folder of page images (a Bucket, a mounted dir, a local dir).
A document is the parent folder plus the file name without its trailing page
number (`Volume 1, A-B_0007.jp2` -> `Volume 1, A-B`); pages sort by that number.

Output: `data/part-*.parquet`, one row per 16-page window:
`{id, doc_id, first_page, page_files, status, page_markdown, raw, ...}`.
`page_markdown` lines up with `page_files`; `raw` is the model's output with its
layout blocks and boxes. `status` is `truncated` or `page_mismatch` when the model
ran out of budget or lost a page boundary (then `page_markdown` is empty and only
`raw` is kept). Re-running the same command resumes where it stopped.

Live view: add `--live-port 7860` and `--expose-public 7860` to `hf jobs uv run`, and
the Job serves a page showing each page image next to its text as windows finish.

About 2% of dense 18th-century pages (mostly mathematics) send the model into a
loop. `unlimited-ocr2-saturate.py` is the tuned version of this script: it
re-reads those pages one at a time, decodes JPEG 2000 faster, reads PDFs and adds
Baidu's own Markdown post-processing. It may change more often.

How it runs: Unlimited-OCR-2 needs Baidu's own SGLang build (no vLLM support).
The script installs that wheel into the image's Python, starts the server in the
Job, and sends requests with saturate (adaptive concurrency, resumable output).

Model: https://huggingface.co/baidu/Unlimited-OCR-2 (3.3B, MIT)
"""

import argparse
import base64
import collections
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The model card's launch settings. The image is the SGLang release whose pins
# match Baidu's wheel (torch 2.13.0, transformers 5.12.1); move both together.
SERVING = {
    "model": "baidu/Unlimited-OCR-2",
    "served_name": "Unlimited-OCR-2",
    "wheel": "https://huggingface.co/baidu/Unlimited-OCR-2/resolve/main/wheel/"
             "sglang-0.5.20.dev10%2Bg8d62f5fb9c-py3-none-any.whl",
    "extra_pip": ["fastokens==0.3.2"],
    "env": {
        "SGLANG_UOCR_EANABLE_DOC_LIMIT": "1",  # (sic) upstream spelling
        "SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW": "10",
        "SGLANG_UOCR_DOC_BASE_PAGE_WINDOW": "10",
        # Per-page output cap. A normal book page is ~2,000 characters (p99 ~6,700); the
        # tokens past 8000 on a page are repetition loops, which held whole runs' tails.
        "SGLANG_UOCR_MAX_PAGE_TOKENS": "8000",
        "SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB": "64",
        "SGLANG_UOCR_SHM_WRAP_MODEL_DATA": "1",
        "SGLANG_UOCR_SHM_WRAP_MIN_MB": "1",
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    "serve_args": [
        "--chat-template", "unlimited-ocr",
        "--tokenizer-backend", "fastokens", "--tokenizer-worker-num", "8",
        "--mem-fraction-static", "0.7",
        "--max-running-requests", "64",
        "--max-prefill-tokens", "2048",
        "--enable-custom-logit-processor", "--enable-streaming-session",
        "--allow-auto-truncate", "--disable-radix-cache",
        "--mm-preprocess-cache-size-mb", "0",
        "--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", "64",
        "--cuda-graph-backend-prefill", "full",
        "--cuda-graph-max-bs-prefill", "16384",
        "--cuda-graph-bs-prefill", "1024", "2048",
        "--cuda-graph-config", '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}',
    ],
}

# The card's no-repeat n-gram guard: DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
# i.e. json({"callable": dill.dumps(cls).hex()}). dill pickles an importable class
# by reference (module + name), so this is a fixed string and needs no sglang here.
NGRAM_PROCESSOR = json.dumps({"callable": (
    "80059559000000000000008c2a73676c616e672e7372742e73616d706c696e672e637573746f6d5f6c6f6769"
    "745f70726f636573736f72948c26446565707365656b4f43524e6f5265706561744e4772616d4c6f67697450"
    "726f636573736f729493942e")})

PAGE_EXTS = (".jp2", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
FURNITURE = {"page_number", "header", "footer"}
BLOCK_RE = re.compile(r"<BLOCK>(?P<type>[a-z_]+)(?:<LEVEL(?P<level>\d)>)?\[[\d, ]*\]<FORK>")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def doc_key(rel_path: str) -> str:
    """`item/Volume 1, A-B_0007.jp2` -> `item/Volume 1, A-B`; bare `0007.jp2` -> `.`."""
    stem = rel_path.rsplit(".", 1)[0]
    doc = re.sub(r"[_\-. ]?\d+$", "", stem)
    return doc if doc and not doc.endswith("/") else (doc or ".")


def list_windows(pattern: str, pages_per_request: int, max_pages):
    """List page files, group them into documents, cut each into windows -> (id, row)."""
    from saturate import bucket_rows

    if not re.search(r"[*?\[]", pattern):
        pattern = pattern.rstrip("/") + "/**"
    docs: dict[str, list[tuple[str, str]]] = {}
    for rid, row in bucket_rows(pattern, read=False):
        if rid.lower().endswith(PAGE_EXTS):
            docs.setdefault(doc_key(rid), []).append((rid, row["path"]))
    if not docs:
        sys.exit(f"no page images matched {pattern!r}")

    n_pages = 0
    for doc in sorted(docs, key=natural_key):
        pages = sorted(docs[doc], key=lambda p: natural_key(p[0]))
        for start in range(0, len(pages), pages_per_request):
            if max_pages is not None and n_pages >= max_pages:
                return
            window = pages[start: start + pages_per_request]
            if max_pages is not None:
                window = window[: max_pages - n_pages]
            n_pages += len(window)
            # The id encodes the page range, so a --max-pages-truncated window is not
            # mistaken for the full window when a later run resumes.
            yield f"{doc}@{start:06d}-{start + len(window):06d}", {
                "doc_id": doc, "first_page": start,
                "page_files": [rid for rid, _ in window],
                "paths": [p for _, p in window],
            }


def load_page(fs, path: str, max_side: int, live=None) -> str:
    """Read one page image, downscale it, return base64 JPEG (and give the live view a thumbnail)."""
    from PIL import Image

    with fs.open(path, "rb") as f:
        data = f.read()
    img = Image.open(io.BytesIO(data))
    if img.format == "JPEG2000":
        # Decode straight at a lower wavelet level while the page stays >= max_side: the
        # same pixels after downscaling, for a fraction of the CPU. Some codestreams fail
        # at a reduced level, so step back towards a full decode until one works.
        reduce = 0
        while max(img.size) >> (reduce + 1) >= max_side:
            reduce += 1
        while True:
            try:
                img = Image.open(io.BytesIO(data))
                img.reduce = reduce
                img.load()
                break
            except OSError:
                if reduce == 0:
                    raise
                reduce -= 1
    else:
        img.load()
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    if live is not None:
        live.add_thumbnail(path, img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    if live is not None:
        live.offer(path, b64)
    return b64


def page_request(images_b64: list[str], stream: bool = False) -> dict:
    """The chat request for consecutive pages, with the model card's sampling and guard settings."""
    n = len(images_b64)
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in images_b64]
    content.append({"type": "text", "text": "<PAGE><image>" * n + "Multi page parsing."})
    return {
        "model": SERVING["served_name"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8000 * n,  # output budget for the whole window
        "temperature": 0,
        "skip_special_tokens": False,  # <PAGE>/<BLOCK> markers are the structure
        "images_config": {"crop_mode": False},  # base mode: one 1024 px view per page
        "custom_logit_processor": NGRAM_PROCESSOR,
        "custom_params": {
            "ngram_size": 35, "window_size": 1024 if n > 1 else 128,
            "page_token_id": 128827, "eos_token_id": 1, "max_page_tokens": 8000,
        },
        "stream": stream,
    }


LIVE_PAGE = """<!doctype html><meta charset="utf-8"><title>Unlimited-OCR-2 live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f4f2ee;--fg:#1c1c1e;--muted:#66645f;--card:#fff;--line:#dfdcd5;--accent:#d9622b;--ok:#2f7d5b}
@media (prefers-color-scheme:dark){:root{--bg:#101113;--fg:#ececef;--muted:#9096a0;--card:#191b1f;--line:#2b2e34;--accent:#ff8a3d;--ok:#5cc29a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,sans-serif;height:100vh;
 display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;overflow:hidden}
header{display:flex;align-items:center;gap:14px;padding:14px 20px 6px;min-width:0}
h1{font-size:20px;margin:0;font-weight:650;white-space:nowrap}
#banner{font-size:14px;padding:3px 10px;border-radius:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
#banner.done{background:var(--ok);color:#fff}#banner.stop{background:var(--accent);color:#fff}
.label{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
#flight{display:flex;align-items:center;gap:12px;padding:6px 20px 10px;border-bottom:1px solid var(--line);min-height:80px;min-width:0}
#flightThumbs{display:flex;gap:6px;overflow:hidden;flex:1;min-width:0}
#flightThumbs img{width:46px;aspect-ratio:3/4;object-fit:cover;background:var(--line);border:1px solid var(--line);opacity:.55;animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{50%{opacity:.95}}
#middle{display:grid;grid-template-columns:minmax(0,1fr) 300px;min-height:0}
#spot{display:grid;grid-template-columns:minmax(0,45%) minmax(0,1fr);gap:24px;padding:16px 20px;min-height:0}
#spotImg{max-width:100%;max-height:100%;object-fit:contain;justify-self:center;border:1px solid var(--line);background:var(--card)}
#spotRight{display:flex;flex-direction:column;min-height:0;min-width:0}
#spotMeta{font-size:14px;color:var(--muted);margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#spotMeta.live::before{content:"● ";color:var(--accent);animation:blink 1s steps(1) infinite}
#spotText{flex:1;overflow:hidden;white-space:pre-wrap;font:19px/1.55 Georgia,serif;max-width:62ch;
 -webkit-mask-image:linear-gradient(transparent,#000 12%);mask-image:linear-gradient(transparent,#000 12%)}
.fade{animation:fade .5s ease-out}@keyframes fade{from{opacity:0}}
#wait{grid-column:1/-1;display:flex;flex-direction:column;gap:16px;min-height:0;min-width:0}
#steps{font-size:17px;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 22px}
#steps .now{color:var(--fg);font-weight:600}#steps .now::before{content:"● ";color:var(--accent);animation:blink 1s steps(1) infinite}
#steps .past::before{content:"✓ ";color:var(--ok)}#steps b{font-weight:inherit;font-variant-numeric:tabular-nums;margin-left:6px}
@keyframes blink{50%{opacity:0}}
#prep{display:flex;flex-wrap:wrap;gap:10px;overflow:hidden;align-content:flex-start;min-height:0}
#prep img{height:190px;aspect-ratio:3/4;object-fit:cover;background:var(--line);border:1px solid var(--line);animation:pulse 1.6s ease-in-out infinite}
aside{border-left:1px solid var(--line);padding:16px 20px;display:flex;flex-direction:column;gap:16px;min-width:0}
.big{font-size:52px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.05}
.mid{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
.bar{height:8px;background:var(--line);border-radius:4px;margin-top:6px}.bar div{height:8px;background:var(--accent);border-radius:4px;width:0;transition:width .8s}
#spark{width:100%;height:44px;display:block}.bad{color:var(--accent)}
#strip{display:flex;gap:8px;padding:10px 20px 14px;border-top:1px solid var(--line);overflow:hidden;height:150px}
#strip figure{margin:0;flex:none;animation:enter .6s ease-out}
#strip img{width:80px;aspect-ratio:3/4;object-fit:cover;background:var(--line);border:1px solid var(--line);display:block}
#strip figcaption{font-size:13px;color:var(--muted);text-align:center;width:80px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
@keyframes enter{from{opacity:0;transform:translateX(-30px)}}
@media (max-width:800px){body{overflow:auto;height:auto;display:block}#middle,#spot{grid-template-columns:minmax(0,1fr)}
 aside>div{min-width:0}
 aside{order:-1;border-left:0;display:grid;grid-template-columns:1fr 1fr}#spotImg{max-height:40vh}#strip{height:auto}
 #prep img{height:120px}.big{font-size:40px}}
</style>
<header><h1>Unlimited-OCR-2 · live</h1><span id="banner"></span></header>
<section id="flight"><span class="label">Ready for the model<br><span id="flightN" class="mid">0</span></span><div id="flightThumbs"></div></section>
<section id="middle">
 <div id="spot"><div id="wait"><div id="steps"></div><div id="prep"></div></div></div>
 <aside>
  <div><div class="label">Pages per minute</div><div class="big" id="rate">–</div>
   <svg id="spark" viewBox="0 0 100 40" preserveAspectRatio="none"></svg></div>
  <div><div class="label">Pages read</div><div class="mid" id="done">0</div><div class="label" id="of"></div>
   <div class="bar"><div id="bar"></div></div></div>
  <div><div class="label">GPU load</div><div class="mid" id="gpu">–</div><div class="bar"><div id="gpuBar"></div></div></div>
  <div><div class="label">Words extracted</div><div class="mid" id="words">0</div></div>
  <div><div class="label">Reading for</div><div class="mid" id="elapsed">0:00</div></div>
  <div id="failedBox" hidden><div class="label">Needs a retry</div><div class="mid bad" id="failed">0</div></div>
 </aside>
</section>
<section id="strip"></section>
<script>
const $ = id => document.getElementById(id), fmt = n => n.toLocaleString();
const clock = s => { s = Number.isFinite(s) ? Math.max(0, Math.round(s)) : 0;
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
let since = 0, last = null, misses = 0, newest = null, shown = null, spotReady = false;
const flight = new Map();

function showWaiting(s) {
  const steps = $("steps"); steps.replaceChildren();
  s.phases.forEach((p, i) => {
    const now = i === s.phases.length - 1, el = document.createElement("span"), t = document.createElement("b");
    el.className = now ? "now" : "past"; el.textContent = p.name;
    t.textContent = clock((now ? s.uptime_s : s.phases[i + 1].at_s) - p.at_s); el.append(t); steps.append(el);
  });
}

function buildSpot() {
  $("spot").innerHTML = '<img id="spotImg" alt=""><div id="spotRight"><div id="spotMeta"></div><div id="spotText"></div></div>';
  spotReady = true;
}

function ingest(d) {
  const s = d.stats; last = s;
  if (!spotReady && d.pages.length) buildSpot();
  if (!spotReady) showWaiting(s);
  $("rate").textContent = s.rate_ppm ? s.rate_ppm.toFixed(0) : "–";
  $("done").textContent = fmt(s.pages_done);
  $("of").textContent = `of ${fmt(s.pages_total)}`;
  $("bar").style.width = (100 * s.pages_done / Math.max(1, s.pages_total)) + "%";
  $("elapsed").textContent = clock(s.reading_s);
  $("words").textContent = fmt(s.words);
  $("failed").textContent = fmt(s.pages_failed); $("failedBox").hidden = !s.pages_failed;
  if (s.gpu) { $("gpu").textContent = `${s.gpu.util}%`; $("gpuBar").style.width = s.gpu.util + "%"; }
  drawSpark(s.spark);
  $("flightN").textContent = fmt(s.in_flight);
  const keep = new Set(d.in_flight);
  for (const [k, imgs] of flight) if (!keep.has(k)) { imgs.forEach(i => i.remove()); flight.delete(k); }
  for (const k of d.in_flight) if (!flight.has(k)) {
    const a = new Image(); a.src = "img/" + k; $("flightThumbs").append(a);
    const imgs = [a];
    if (!spotReady) { const b = new Image(); b.src = "img/" + k; $("prep").append(b); imgs.push(b); }
    flight.set(k, imgs);
  }
  // Every finished page goes on the strip, so its motion is the real throughput.
  // A late open or reload starts from the latest window, not the whole history.
  const fresh = since === 0 ? d.pages.slice(-16) : d.pages;
  for (const p of d.pages) since = Math.max(since, p.seq);
  for (const p of fresh) addToStrip(p);
  // Spotlight the newest page that has text; blank leaves and plates only when nothing else came back.
  const withText = fresh.filter(p => (p.text || "").length > 200);
  if (fresh.length) newest = (withText.length ? withText : fresh).at(-1);
  if (s.pages_total && s.pages_done >= s.pages_total) banner("done", `Finished · ${fmt(s.pages_done)} pages in ${clock(s.reading_s)}`);
  else banner("", s.phase);
}

function banner(cls, text) { $("banner").className = cls; $("banner").textContent = text; }

function drawSpark(v) {
  if (!v || v.length < 2 || Math.max(...v) === 0) { $("spark").innerHTML = ""; return; }
  const max = Math.max(...v);
  const path = v.map((y, i) => `${i ? "L" : "M"}${(i / (v.length - 1) * 100).toFixed(1)},${(38 - y / max * 34).toFixed(1)}`).join("");
  $("spark").innerHTML = `<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
}

function addToStrip(p) {
  const f = document.createElement("figure"), i = new Image(), c = document.createElement("figcaption");
  i.src = "img/" + p.img; c.textContent = "p. " + p.page + (p.status === "ok" ? "" : " · retry");
  f.title = p.doc; f.append(i, c); $("strip").prepend(f);
  while ($("strip").children.length > 40) $("strip").lastChild.remove();
}

// Spotlight: the page the model is reading right now, its text arriving token by token (one
// streaming request beside the batch). Between live pages: the newest finished page.
let live = null, liveImg = "";
async function pollStream() {
  try {
    live = await (await fetch("stream.json")).json();
    if (live && !spotReady) buildSpot();
    if (live && spotReady) {
      if (live.img && live.img !== liveImg) { liveImg = live.img; $("spotImg").src = "img/" + live.img; }
      $("spotMeta").className = live.done ? "" : "live";
      $("spotMeta").textContent = (live.done ? "Just read · " : `Reading now · ${live.tok_s} tokens/s · `) + `${live.doc} · page ${live.page}`;
      const el = $("spotText");
      el.textContent = live.text.replace(/^#+ /gm, "") || "…";
      el.scrollTop = el.scrollHeight;
    }
  } catch (e) {}
  setTimeout(pollStream, live && !live.done ? 250 : 1000);
}

function spotlight() {
  if (spotReady && !live && newest && newest !== shown) {
    const p = newest; shown = p;
    $("spotImg").src = "img/" + p.img;
    $("spotMeta").textContent = `${p.doc} · page ${p.page}` + (p.status === "ok" ? "" : ` · ${p.status}`);
    $("spotMeta").className = p.status === "ok" ? "" : "bad";
    const el = $("spotText");
    el.textContent = (p.text || "(no text on this page)").replace(/^#+ /gm, "").slice(0, 1500);
    el.classList.remove("fade"); void el.offsetWidth; el.classList.add("fade");
  }
  setTimeout(spotlight, 2500);
}

async function poll() {
  try {
    ingest(await (await fetch("feed.json?since=" + since)).json()); misses = 0;
  } catch (e) {
    misses += 1;
    if (misses >= 3 && last) {
      if (last.pages_total && last.pages_done >= last.pages_total) banner("done", `Finished · ${fmt(last.pages_done)} pages in ${clock(last.reading_s)} · the Job has exited`);
      else banner("stop", "Job stopped · check the Job logs");
    }
  }
  setTimeout(poll, 2000);
}
poll(); spotlight(); pollStream();
</script>"""


class LiveView:
    """The last pages read, served as a web page from inside the Job (`--live-port`)."""

    def __init__(self, port: int, pages_total: int, names: dict, keep: int = 300, stale_s: float = 3600):
        self.pages = collections.deque(maxlen=keep)
        self.images: dict[str, bytes] = {}  # key -> JPEG, for in-flight and kept pages
        self.in_flight: dict[str, tuple[str, float]] = {}  # page path -> (image key, time loaded)
        self.stale_s = stale_s  # a window not back after this long was given up on
        self.lock = threading.Lock()
        self.seq = 0
        self.pages_total, self.pages_done, self.pages_failed, self.words = pages_total, 0, 0, 0
        self.booted = time.time()
        self.started = None  # first page loaded: the rate and elapsed time leave out server start-up
        self.phases = []
        self.recent = collections.deque()  # (time, pages) of finished windows, for the rolling rate
        self.spark = collections.deque(maxlen=30)  # pages per minute in 60 s buckets
        self.gpu = None
        self.names = names  # page path -> (document, page number)
        self.offered = None  # (path, base64 JPEG) of the newest prepared page
        self.stream = None  # the page being read live, token by token
        live = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path.startswith("/stream.json"):
                        with live.lock:
                            body = json.dumps(live.stream)
                        self.reply(body.encode(), "application/json")
                    elif self.path.startswith("/feed.json"):
                        try:
                            since = int(self.path.partition("since=")[2] or 0)
                        except ValueError:
                            since = 0
                        self.reply(json.dumps(live.feed(since)).encode(), "application/json")
                    elif self.path.startswith("/img/"):
                        with live.lock:
                            jpeg = live.images.get(self.path[len("/img/"):])
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

        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        threading.Thread(target=self.tick, daemon=True).start()
        # An exposed Job port is served at https://<job id>--<port>.hf.jobs.
        where = (f"https://{os.environ['JOB_ID']}--{port}.hf.jobs (needs --expose-public {port})"
                 if os.environ.get("JOB_ID") else f"http://localhost:{port}")
        print(f"Live view: {where}", file=sys.stderr, flush=True)

    def offer(self, path: str, b64: str):
        with self.lock:
            self.offered = (path, b64)

    def start_streaming(self, url: str):
        """Read one page at a time with streaming on, so the page can show the model's text as it is produced.

        This is one extra single-page request beside the batch (a few percent of the GPU); its
        output is only shown, never written.
        """
        threading.Thread(target=self._stream_loop, args=(url,), daemon=True).start()

    def _stream_loop(self, url: str):
        import urllib.request

        while True:
            with self.lock:
                offered, self.offered = self.offered, None
                key = self.in_flight.get(offered[0], ("",))[0] if offered else ""
            if not offered:
                time.sleep(0.5)
                continue
            path, b64 = offered
            doc, page = self.names.get(path, ("", 0))
            raw, tokens, t0 = "", 0, time.time()
            with self.lock:
                self.stream = {"img": key, "doc": doc, "page": page, "text": "", "tok_s": 0, "done": False}
            try:
                req = urllib.request.Request(f"{url}/v1/chat/completions",
                                             data=json.dumps(page_request([b64], stream=True)).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as resp:
                    for line in resp:
                        line = line.decode("utf-8", "replace").strip()
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        delta = (json.loads(line[6:])["choices"][0].get("delta") or {}).get("content") or ""
                        raw += delta
                        tokens += 1
                        text = clean_page(raw.split("<PAGE>")[-1], keep_furniture=False)
                        with self.lock:
                            self.stream.update(text=text[-1500:], tok_s=round(tokens / max(0.1, time.time() - t0)))
            except (OSError, ValueError, KeyError, IndexError) as e:  # the live view must never affect the run
                print(f"Live view stream: {type(e).__name__}: {e}", file=sys.stderr)
                time.sleep(5)
            with self.lock:
                self.stream["done"] = True
            time.sleep(3)  # leave the finished page on screen for a moment

    def phase(self, name: str):
        with self.lock:
            self.phases.append({"name": name, "at_s": round(time.time() - self.booted)})

    def tick(self):
        """Every few seconds: GPU memory and load (so start-up visibly moves), and the sparkline."""
        last_bucket, done_then = time.time(), 0
        while True:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=10, check=True).stdout.splitlines()[0]
                util, used, total = (float(x) for x in out.split(","))
                gpu = {"util": round(util), "mem_used_gb": used / 1024, "mem_total_gb": total / 1024}
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                gpu = None
            with self.lock:
                self.gpu = gpu
                if self.started and time.time() - last_bucket >= 60:
                    self.spark.append(round((self.pages_done - done_then) * 60 / (time.time() - last_bucket), 1))
                    last_bucket, done_then = time.time(), self.pages_done
            time.sleep(3)

    def add_thumbnail(self, path: str, img):
        thumb = img.copy()
        thumb.thumbnail((600, 600))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)
        with self.lock:
            if self.started is None:
                self.started = time.time()
                self.phases.append({"name": "Reading pages", "at_s": round(self.started - self.booted)})
            self.seq += 1
            key = f"{self.seq}.jpg"
            self.images[key] = buf.getvalue()
            old = self.in_flight.get(path)
            if old:  # prepared again for a retry: drop the earlier copy
                self.images.pop(old[0], None)
            self.in_flight[path] = (key, time.time())

    def add_window(self, row: dict, status: str, texts):
        now = time.time()
        with self.lock:
            self.pages_done += len(row["paths"])
            self.recent.append((now, len(row["paths"])))
            for i, (path, rid) in enumerate(zip(row["paths"], row["page_files"], strict=True)):
                text = texts[i] if texts and i < len(texts) else ""
                self.words += len(text.split())
                self.pages_failed += status != "ok"
                if len(self.pages) == self.pages.maxlen:  # the oldest kept page leaves: drop its image
                    self.images.pop(self.pages[0]["img"], None)
                self.seq += 1
                key, _ = self.in_flight.pop(path, ("", now))
                self.pages.append({"seq": self.seq, "doc": row["doc_id"].rsplit("/", 1)[-1],
                                   "page": row["first_page"] + i + 1, "file": rid, "status": status,
                                   "text": text[:4000], "img": key})

    def feed(self, since: int) -> dict:
        now = time.time()
        with self.lock:
            for path, (key, t) in list(self.in_flight.items()):
                if now - t > self.stale_s:  # never came back: stop showing it as being read
                    del self.in_flight[path]
                    self.images.pop(key, None)
            while self.recent and now - self.recent[0][0] > 300:
                self.recent.popleft()
            span = min(300, now - self.started) if self.started else 0
            rate = sum(n for _, n in self.recent) * 60 / span if span >= 20 else 0
            return {"stats": {"phase": self.phases[-1]["name"] if self.phases else "Starting",
                              "phases": self.phases,
                              "uptime_s": round(now - self.booted),
                              "reading_s": round(now - self.started) if self.started else 0,
                              "pages_total": self.pages_total, "pages_done": self.pages_done,
                              "pages_failed": self.pages_failed, "words": self.words,
                              "in_flight": len(self.in_flight), "rate_ppm": round(rate, 1),
                              "spark": list(self.spark), "gpu": self.gpu},
                    "in_flight": [key for key, _ in self.in_flight.values()][-24:],
                    "pages": [p for p in self.pages if p["seq"] > since]}


def clean_page(raw_page: str, keep_furniture: bool) -> str:
    """One page of model output -> plain markdown.

    Drops block tags and boxes, turns titles into `#` headings, drops page
    numbers and running heads, and joins `<MERGE>`d blocks (e.g. verse lines).
    """
    blocks: list[str] = []
    merge_next = False
    matches = list(BLOCK_RE.finditer(raw_page))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_page)
        body = raw_page[m.end(): end].strip()
        merge_prev, merge_next = merge_next, body.endswith("<MERGE>")
        body = body.removesuffix("<MERGE>").strip()
        kind = m.group("type")
        if not body or (kind in FURNITURE and not keep_furniture):
            continue
        if kind == "title" and m.group("level"):
            body = "#" * int(m.group("level")) + " " + body
        if merge_prev and blocks:
            blocks[-1] += "\n" + body
        else:
            blocks.append(body)
    return "\n\n".join(blocks)


def prepare_workers() -> int:
    """Threads that decode and encode pages: enough to keep ~40 requests fed, within the Job's CPU quota.

    os.cpu_count() reports the host's cores (190 on a Jobs node), not the quota; that many
    threads starve the loop that sends requests. Jobs sets CPU_CORES to the quota.
    """
    cores = int(os.environ.get("CPU_CORES") or len(os.sched_getaffinity(0)))
    return max(4, min(16, cores - 6))  # leave cores for SGLang's tokenizer and scheduler


def start_server_cmd() -> list[str]:
    """Install Baidu's wheel into the image's sglang Python and return the server command."""
    # The SGLang image keeps sglang in its own venv (/opt/sglang/bin), first on PATH;
    # `uv run` puts this script's venv in front, so look past it.
    own_bin = os.path.realpath(os.path.join(sys.prefix, "bin"))
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if os.path.realpath(p) != own_bin)
    py = shutil.which("python3", path=path)
    if not py or subprocess.run([py, "-c", "import sglang"], check=False,
                                capture_output=True).returncode != 0:
        sys.exit("No Python with sglang found: run this on the lmsysorg/sglang image "
                 "(`hf jobs uv run` picks it up from the script's [tool.hf-jobs] table).")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    pip = ["uv", "pip", "install", "-q", "--python", py]
    subprocess.run([*pip, "--no-deps", SERVING["wheel"]], check=True, env=env)
    subprocess.run([*pip, *SERVING["extra_pip"]], check=True, env=env)
    os.environ.update(SERVING["env"])  # inherited by the server process
    return [py, "-m", "sglang.launch_server", "--model-path", SERVING["model"],
            "--served-model-name", SERVING["served_name"], *SERVING["serve_args"],
            "--enable-metrics", "--host", "127.0.0.1", "--port", "8000"]


def main():
    ap = argparse.ArgumentParser(description="Unlimited-OCR-2 document OCR on HF Jobs")
    ap.add_argument("input", help="Glob or folder of page images: hf://buckets/..., a mounted dir, ...")
    ap.add_argument("output", help="Output dataset id (org/name) or fsspec URI (hf://buckets/...)")
    ap.add_argument("--pages-per-request", type=int, default=16,
                    help="Consecutive pages per request (default 16)")
    ap.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (for a test run)")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="Long side (px) pages are sent at. The model resizes each page to a "
                         "1024 px view, so more pixels only make the requests bigger")
    ap.add_argument("--keep-furniture", action="store_true",
                    help="Keep page numbers and running heads in page_markdown")
    ap.add_argument("--retry-errors", action="store_true", help="Retry windows recorded as errors")
    ap.add_argument("--live-port", type=int, default=None,
                    help="Serve a live view of pages as they are read on this port "
                         "(on Jobs, also pass --expose-public with the same port)")
    args = ap.parse_args()

    import fsspec
    import saturate.transport
    from saturate import Auto, Engine, pump

    rows = list(list_windows(args.input, args.pages_per_request, args.max_pages))
    print(f"{len({r['doc_id'] for _, r in rows})} documents, "
          f"{sum(len(r['paths']) for _, r in rows)} pages, {len(rows)} requests", file=sys.stderr)
    fs = fsspec.core.url_to_fs(args.input.split("*")[0] or ".")[0]
    live = None
    if args.live_port:
        names = {p: (r["doc_id"].rsplit("/", 1)[-1], r["first_page"] + i + 1)
                 for _, r in rows for i, p in enumerate(r["paths"])}
        live = LiveView(args.live_port, sum(len(r["paths"]) for _, r in rows), names)

    def to_request(row):
        return page_request([load_page(fs, p, args.max_side, live) for p in row["paths"]])

    def parse(row, body):
        choice = body["choices"][0]
        raw = choice["message"]["content"]
        raw_pages = raw.split("<PAGE>")[1:]
        if choice.get("finish_reason") == "length":
            status = "truncated"
        elif len(raw_pages) != len(row["paths"]):
            status = "page_mismatch"
        else:
            status = "ok"
        usage = body.get("usage") or {}
        page_markdown = [clean_page(p, args.keep_furniture) for p in raw_pages] if status == "ok" else None
        if live is not None:
            live.add_window(row, status, page_markdown)
        return {
            "doc_id": row["doc_id"],
            "first_page": row["first_page"],
            "page_files": row["page_files"],
            "status": status,
            "page_markdown": page_markdown,
            "raw": raw,
            "model": SERVING["model"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    is_repo_id = re.fullmatch(r"[\w-][\w.-]*/[\w.-]*[\w-]", args.output) and not os.path.exists(args.output)
    output = f"hf://datasets/{args.output}/data" if is_repo_id else args.output

    # One request is minutes of decoding. saturate 0.2.0 caps a request's retries at
    # 300 s in total, which a retried 16-page request never fits in; match the read timeout.
    saturate.transport.RETRY_BUDGET_S = 3600.0
    if live is not None:
        live.phase("Installing Baidu's SGLang build")
    cmd = start_server_cmd()
    if live is not None:
        live.phase("Loading the model")
    with Engine(SERVING["model"], engine="sglang", cmd=cmd) as endpoint:
        if live is not None:
            live.start_streaming("http://127.0.0.1:8000")
        stats = pump(rows, to_request, parse, endpoint, output,
                     window=Auto(initial=32, max_limit=64),  # long requests: start wide
                     flush_every=4,  # each row is minutes of GPU time; write often
                     read_timeout=3600, prepare_workers=prepare_workers(), retry_errors=args.retry_errors)
    print(f"{output}: {stats.rows_processed} windows written, {stats.rows_failed} errors", file=sys.stderr)
    if live is not None:
        live.phase("Finished")
        time.sleep(60)  # keep the final numbers on screen for a minute before the Job exits
    print("UNLIMITED_OCR2 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
