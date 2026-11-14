"""Serve an exported GPT-2 folder on localhost, side by side with OpenAI's GPT-2, for trying prompts.

Usage:
    python serve.py <export dir> [--compare gpt2] [--port 8765] [--device mps|cuda|cpu]
    python serve.py --blind <a.jsonl> <b.jsonl> [--votes votes.jsonl] [--count 20] [--seed 0]

--blind shows two models' saved Part B answers side by side with the model names
hidden, in a seeded random order, and appends each vote to the votes file.
"""

import argparse
import json
import random
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat.answer import read_jsonl

MAX_NEW_TOKENS = 512

STYLE = """:root { --bg:#f6f7f9; --surface:#fff; --ink:#1b2130; --muted:#5b6478; --line:#dde1e8; --accent:#2f4fb5; --gen:#1f7a4d; }
@media (prefers-color-scheme: dark) { :root { --bg:#12151c; --surface:#1a1e28; --ink:#e5e8ef; --muted:#9aa3b5; --line:#2c3240; --accent:#8ea6f0; --gen:#6fd3a0; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1200px; margin:0 auto; padding:32px 16px; display:grid; gap:16px; }
h1 { margin:0; font-size:1.5rem; }
h2 { margin:0; font-size:1rem; }
p.sub { margin:0; color:var(--muted); font-size:.92rem; }
textarea { width:100%; min-height:110px; padding:12px; border:1px solid var(--line); border-radius:8px; background:var(--surface); color:var(--ink); font:inherit; resize:vertical; }
.controls { display:flex; flex-wrap:wrap; gap:12px 20px; align-items:center; font-size:.92rem; color:var(--muted); }
.controls label { display:flex; gap:8px; align-items:center; }
.controls input[type=number] { width:84px; padding:4px 6px; border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--ink); font:inherit; }
button { padding:8px 18px; border:0; border-radius:8px; background:var(--accent); color:#fff; font:inherit; font-weight:600; cursor:pointer; }
button:disabled { opacity:.5; cursor:wait; }
#panels { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; }
.panel { display:grid; gap:8px; align-content:start; min-width:0; }
.out { white-space:pre-wrap; overflow-wrap:anywhere; padding:16px; border:1px solid var(--line); border-radius:8px; background:var(--surface); min-height:160px; }
.out .gen { color:var(--gen); }
.meta { color:var(--muted); font-size:.85rem; }
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>micrograd GPT-2</title>
<style>
__STYLE__</style></head><body><main>
<h1>micrograd GPT-2 124M</h1>
<p class="sub">Base models: they continue text, they do not follow instructions. Both get the same prompt, settings and seed.</p>
<textarea id="prompt">The water cycle is</textarea>
<div class="controls">
<label>Max new tokens <input id="max" type="number" min="1" max="512" value="120"></label>
<label>Temperature <input id="temp" type="number" min="0" max="2" step="0.1" value="0.8"></label>
<label>Top-k <input id="topk" type="number" min="0" max="50257" value="50"></label>
<label>Seed <input id="seed" type="number" min="0" value="0"></label>
<label><input id="reseed" type="checkbox" checked> New seed each run</label>
<button id="go">Generate</button>
</div>
<div id="panels"></div>
</main><script>
const MODELS = __MODELS__;
const go = document.getElementById("go"), panels = document.getElementById("panels");
const views = MODELS.map(m => {
  const panel = document.createElement("section"); panel.className = "panel";
  const title = document.createElement("h2"); title.textContent = m.label;
  const out = document.createElement("div"); out.className = "out";
  const meta = document.createElement("div"); meta.className = "meta";
  panel.append(title, out, meta); panels.append(panel);
  return { id: m.id, out, meta };
});
const value = id => +document.getElementById(id).value;
async function runOne(view, body) {
  view.meta.textContent = "generating...";
  try {
    const r = await fetch("/generate", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, model: view.id }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    view.out.textContent = "";
    view.out.append(document.createTextNode(body.prompt));
    const s = document.createElement("span"); s.className = "gen"; s.textContent = j.completion; view.out.append(s);
    view.meta.textContent = `${j.new_tokens} tokens in ${j.seconds.toFixed(2)} s, seed ${body.seed}`;
  } catch (e) { view.meta.textContent = "error: " + e.message; }
}
async function run() {
  const seedBox = document.getElementById("seed");
  if (document.getElementById("reseed").checked) seedBox.value = Math.floor(Math.random() * 1e6);
  const body = { prompt: document.getElementById("prompt").value, max_new_tokens: value("max"),
    temperature: value("temp"), top_k: value("topk"), seed: value("seed") };
  go.disabled = true;
  await Promise.all(views.map(v => runOne(v, body)));
  go.disabled = false;
}
go.onclick = run;
document.getElementById("prompt").addEventListener("keydown", e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); });
</script></body></html>
"""


BLIND_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind side by side</title>
<style>
__STYLE__.question { font-size:1.15rem; font-weight:600; }
.votes { display:flex; flex-wrap:wrap; gap:12px; }
details { color:var(--muted); font-size:.92rem; }
</style></head><body><main>
<h1>Blind side by side</h1>
<p class="sub">Two models answer the same question. Names are hidden and sides are shuffled. Keys: 1 left, 2 tie, 3 right.</p>
<div class="meta" id="progress"></div>
<div class="question" id="question"></div>
<details><summary>Reference answer</summary><p id="reference"></p></details>
<div class="votes">
<button data-choice="left">Left is better</button>
<button data-choice="tie">Tie</button>
<button data-choice="right">Right is better</button>
</div>
<div id="panels">
<section class="panel"><h2>Left</h2><div class="out" id="left"></div></section>
<section class="panel"><h2>Right</h2><div class="out" id="right"></div></section>
</div>
</main><script>
let items = [], current = null;
const $ = id => document.getElementById(id);
const buttons = [...document.querySelectorAll("button[data-choice]")];
function show() {
  const done = items.filter(i => i.voted).length;
  $("progress").textContent = `${done} of ${items.length} voted`;
  current = items.find(i => !i.voted);
  buttons.forEach(b => b.disabled = !current);
  $("question").textContent = current ? current.question : "All done. Votes are saved.";
  $("reference").textContent = current ? current.reference : "";
  $("left").textContent = current ? current.left : "";
  $("right").textContent = current ? current.right : "";
}
let busy = false;
async function vote(choice) {
  if (!current || busy) return;
  busy = true;
  buttons.forEach(b => b.disabled = true);
  let message = "";
  try {
    const r = await fetch("/vote", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index: current.index, choice }) });
    if (r.ok) current.voted = true; else message = "error: " + (await r.json()).error;
  } catch (e) { message = "error: " + e.message; }
  busy = false;
  show();
  if (message) $("progress").textContent = message;
}
buttons.forEach(b => b.onclick = () => vote(b.dataset.choice));
document.addEventListener("keydown", e => { const c = { "1": "left", "2": "tie", "3": "right" }[e.key]; if (c) vote(c); });
fetch("/items").then(r => r.json()).then(j => { items = j; show(); });
</script></body></html>
"""


def pick_device(name):
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Generator:
    def __init__(self, path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    def generate_ids(self, prompt, max_new_tokens, temperature, top_k, seed, stop_strings=None):
        max_new_tokens = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS))
        ids = self.tokenizer(prompt).input_ids or [self.tokenizer.eos_token_id]
        ids = ids[-(self.block_size - max_new_tokens):]
        sampling = float(temperature) > 0
        options = {"do_sample": sampling, "max_new_tokens": max_new_tokens,
                   "pad_token_id": self.tokenizer.eos_token_id}
        if sampling:
            options |= {"temperature": float(temperature), "top_k": int(top_k)}
        if stop_strings:
            options |= {"stop_strings": stop_strings, "tokenizer": self.tokenizer}
        torch.manual_seed(int(seed))
        with torch.no_grad():
            start = torch.tensor([ids], device=self.device)
            out = self.model.generate(start, attention_mask=torch.ones_like(start), **options)
        return out[0, len(ids):].tolist()

    def generate(self, prompt, max_new_tokens, temperature, top_k, seed):
        new = self.generate_ids(prompt, max_new_tokens, temperature, top_k, seed)
        return self.tokenizer.decode(new), len(new)


class Blind:
    CHOICES = ("left", "right", "tie")

    def __init__(self, path_a, path_b, votes, count, seed):
        rows_a = {row["id"]: row for row in read_jsonl(path_a)}
        rows_b = {row["id"]: row for row in read_jsonl(path_b)}
        self.names = {"a": next(iter(rows_a.values()))["model"], "b": next(iter(rows_b.values()))["model"]}
        if self.names["a"] == self.names["b"]:
            raise ValueError(f"both files hold answers from {self.names['a']!r}: votes could not tell them apart")
        self.votes = Path(votes)
        self.seed = seed
        rng = random.Random(seed)
        shared = sorted(rows_a.keys() & rows_b.keys())
        chosen = sorted(rng.sample(shared, min(count, len(shared))))
        voted = self.voted()
        self.items = []
        for index, question_id in enumerate(chosen):
            a, b = rows_a[question_id], rows_b[question_id]
            sides = ("a", "b") if rng.random() < 0.5 else ("b", "a")
            left, right = (a, b) if sides == ("a", "b") else (b, a)
            self.items.append({"index": index, "id": question_id, "group": a["group"], "sides": sides,
                               "question": a["question"], "reference": a["reference"],
                               "left": left["answer"], "right": right["answer"], "voted": question_id in voted})

    def voted(self):
        if not self.votes.is_file():
            return set()
        pair = (self.names["a"], self.names["b"])
        return {v["question_id"] for v in read_jsonl(self.votes) if (v["a"], v["b"]) == pair and v["seed"] == self.seed}

    def public(self):
        keys = ("index", "question", "reference", "left", "right", "voted")
        return [{key: item[key] for key in keys} for item in self.items]

    def record(self, index, choice):
        if choice not in self.CHOICES:
            raise ValueError(f"choice must be one of {', '.join(self.CHOICES)}")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.items):
            raise ValueError(f"index must be a whole number from 0 to {len(self.items) - 1}")
        item = self.items[index]
        if item["voted"]:
            raise ValueError(f"question {index} already has a vote")
        left, right = (self.names[side] for side in item["sides"])
        winner = {"left": left, "right": right, "tie": "tie"}[choice]
        vote = {"question_id": item["id"], "group": item["group"], "a": self.names["a"], "b": self.names["b"],
                "left": left, "right": right, "choice": choice, "winner": winner, "seed": self.seed,
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.votes.parent.mkdir(parents=True, exist_ok=True)
        with open(self.votes, "a") as f:
            f.write(json.dumps(vote) + "\n")
        item["voted"] = True
        return vote


class PageHandler(BaseHTTPRequestHandler):
    page = ""

    def send(self, status, body, content_type):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, body):
        self.send(200, json.dumps(body), "application/json")

    def fail(self, status, message):
        self.send(status, json.dumps({"error": message}), "application/json")

    def read_json(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def get_json(self):
        return None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, self.page, "text/html; charset=utf-8")
            return
        body = self.get_json()
        if body is None:
            self.fail(404, "not found")
        else:
            self.send_json(body)

    def log_message(self, format, *args):
        pass


def handler_for(generators, page):
    lock = threading.Lock()

    class Handler(PageHandler):
        def do_POST(self):
            if self.path != "/generate":
                self.fail(404, "not found")
                return
            try:
                request = self.read_json()
                generator = generators.get(request.get("model", "micrograd"))
                if generator is None:
                    self.fail(400, f"unknown model {request.get('model')!r}")
                    return
                started = time.time()
                with lock:
                    completion, count = generator.generate(
                        request.get("prompt", ""),
                        request.get("max_new_tokens", 120),
                        request.get("temperature", 0.8),
                        request.get("top_k", 50),
                        request.get("seed", 0),
                    )
                self.send_json({"completion": completion, "new_tokens": count, "seconds": time.time() - started})
            except (ValueError, TypeError) as error:
                self.fail(400, str(error))

    Handler.page = page
    return Handler


def blind_handler_for(blind):
    lock = threading.Lock()

    class Handler(PageHandler):
        page = BLIND_PAGE.replace("__STYLE__", STYLE)

        def get_json(self):
            return blind.public() if self.path == "/items" else None

        def do_POST(self):
            if self.path != "/vote":
                self.fail(404, "not found")
                return
            try:
                request = self.read_json()
                with lock:
                    blind.record(request.get("index"), request.get("choice"))
                self.send_json({"ok": True})
            except (ValueError, TypeError, IndexError) as error:
                self.fail(400, str(error))

    return Handler


def serve_blind(args):
    path_a, path_b = args.blind
    votes = args.votes or Path(__file__).parent / "results" / "chat" / f"votes-{Path(path_a).stem}-vs-{Path(path_b).stem}.jsonl"
    blind = Blind(path_a, path_b, votes, args.count, args.seed)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), blind_handler_for(blind))
    print(f"blind side by side of {len(blind.items)} questions on http://127.0.0.1:{args.port}, votes to {votes}", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--compare", default="gpt2", help="Hugging Face id or folder; empty to serve one model")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device")
    parser.add_argument("--blind", nargs=2, metavar=("A_JSONL", "B_JSONL"), help="blind vote on two Part B answer files")
    parser.add_argument("--votes", help="votes file for --blind")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.blind:
        serve_blind(args)
        return
    if not args.model:
        parser.error("model is required unless --blind is given")
    device = pick_device(args.device)
    generators = {"micrograd": Generator(args.model, device)}
    models = [{"id": "micrograd", "label": "micrograd GPT-2 124M (yours)"}]
    if args.compare:
        generators["compare"] = Generator(args.compare, device)
        label = "OpenAI GPT-2 124M" if args.compare == "gpt2" else args.compare
        models.append({"id": "compare", "label": label})
    page = PAGE.replace("__STYLE__", STYLE).replace("__MODELS__", json.dumps(models))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(generators, page))
    print(f"serving {', '.join(m['label'] for m in models)} on http://127.0.0.1:{args.port} ({device})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
