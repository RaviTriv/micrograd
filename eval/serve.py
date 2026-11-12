"""Serve an exported GPT-2 folder on localhost, side by side with OpenAI's GPT-2, for trying prompts.

Usage:
    python serve.py <export dir> [--compare gpt2] [--port 8765] [--device mps|cuda|cpu]
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MAX_NEW_TOKENS = 512

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>micrograd GPT-2</title>
<style>
:root { --bg:#f6f7f9; --surface:#fff; --ink:#1b2130; --muted:#5b6478; --line:#dde1e8; --accent:#2f4fb5; --gen:#1f7a4d; }
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
</style></head><body><main>
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
        self.tokenizer = GPT2TokenizerFast.from_pretrained(path)
        self.model = GPT2LMHeadModel.from_pretrained(path, dtype=torch.float32).to(device).eval()
        self.device = device
        self.block_size = self.model.config.n_positions

    def generate(self, prompt, max_new_tokens, temperature, top_k, seed):
        max_new_tokens = max(1, min(int(max_new_tokens), MAX_NEW_TOKENS))
        ids = self.tokenizer(prompt).input_ids or [self.tokenizer.eos_token_id]
        ids = ids[-(self.block_size - max_new_tokens):]
        sampling = float(temperature) > 0
        options = {"do_sample": sampling, "max_new_tokens": max_new_tokens,
                   "pad_token_id": self.tokenizer.eos_token_id}
        if sampling:
            options |= {"temperature": float(temperature), "top_k": int(top_k)}
        torch.manual_seed(int(seed))
        with torch.no_grad():
            start = torch.tensor([ids], device=self.device)
            out = self.model.generate(start, attention_mask=torch.ones_like(start), **options)
        new = out[0, len(ids):].tolist()
        return self.tokenizer.decode(new), len(new)


def handler_for(generators, page):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type):
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def fail(self, status, message):
            self.send(status, json.dumps({"error": message}), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, page, "text/html; charset=utf-8")
            else:
                self.fail(404, "not found")

        def do_POST(self):
            if self.path != "/generate":
                self.fail(404, "not found")
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
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
                body = {"completion": completion, "new_tokens": count, "seconds": time.time() - started}
                self.send(200, json.dumps(body), "application/json")
            except (ValueError, TypeError) as error:
                self.fail(400, str(error))

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--compare", default="gpt2", help="Hugging Face id or folder; empty to serve one model")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device")
    args = parser.parse_args()
    device = pick_device(args.device)
    generators = {"micrograd": Generator(args.model, device)}
    models = [{"id": "micrograd", "label": "micrograd GPT-2 124M (yours)"}]
    if args.compare:
        generators["compare"] = Generator(args.compare, device)
        label = "OpenAI GPT-2 124M" if args.compare == "gpt2" else args.compare
        models.append({"id": "compare", "label": label})
    page = PAGE.replace("__MODELS__", json.dumps(models))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(generators, page))
    print(f"serving {', '.join(m['label'] for m in models)} on http://127.0.0.1:{args.port} ({device})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
