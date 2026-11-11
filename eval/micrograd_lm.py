"""lm-evaluation-harness model that scores GPT-2 token ids through a scorer process.

The scorer (gpt_score, or fake_scorer.py) reads one JSON request per line,
{"context": [...], "continuation": [...]}, and answers one JSON line,
{"logprob": float, "greedy": bool, "scored": int}, or {"error": str}.
Tokenisation, the boundary-whitespace rule and the empty-context prefix come
from the harness's TemplateLM, the base of its Hugging Face backend.

Usage:
    lm = MicrogradLM(scorer="build/gpt_score --checkpoint gpt.bin --device cpu")
    lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"])
    lm.close()
"""

import atexit
import json
import shlex
import subprocess

import lm_eval.models  # noqa: F401
import tiktoken
from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from tqdm import tqdm


@register_model("micrograd")
class MicrogradLM(TemplateLM):
    chunk_size = 64

    def __init__(self, scorer, block_size=1024, **kwargs):
        super().__init__()
        self.command = shlex.split(scorer) if isinstance(scorer, str) else list(scorer)
        self.block_size = int(block_size)
        self.encoding = tiktoken.get_encoding("gpt2")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        atexit.register(self.close)

    @property
    def eot_token_id(self):
        return self.encoding.eot_token

    @property
    def max_length(self):
        return self.block_size

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return self.encoding.encode(string, allowed_special="all")

    def tok_decode(self, tokens, **kwargs):
        return self.encoding.decode(tokens)

    def score(self, pairs):
        requests = [{"context": c, "continuation": t} for c, t in pairs]
        for request in requests:
            self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        results = []
        for request in requests:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"scorer exited before answering {request}")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"scorer error {response['error']!r} for {request}")
            results.append((float(response["logprob"]), bool(response["greedy"])))
        return results

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
        pairs = [(context, continuation) for _, context, continuation in requests]
        results = []
        for start in tqdm(
            range(0, len(pairs), self.chunk_size),
            desc="Scoring",
            disable=disable_tqdm,
        ):
            chunk = self.score(pairs[start : start + self.chunk_size])
            for (key, _, _), answer in zip(requests[start:], chunk):
                if key is not None:
                    self.cache_hook.add_partial("loglikelihood", key, answer)
            results.extend(chunk)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        windows, counts = [], []
        for (string,) in [request.args for request in requests]:
            rolling = utils.get_rolling_token_windows(
                token_list=self.tok_encode(string),
                prefix_token=self.prefix_token_id,
                max_seq_len=self.block_size - 1,
                context_len=1,
            )
            found = [(None, *utils.make_disjoint_window(w)) for w in rolling]
            windows.extend(found)
            counts.append(len(found))
        scores = self._loglikelihood_tokens(windows, disable_tqdm=disable_tqdm)
        totals, position = [], 0
        for request, count in zip(requests, counts):
            total = sum(logprob for logprob, _ in scores[position : position + count])
            self.cache_hook.add_partial("loglikelihood_rolling", request.args, total)
            totals.append(total)
            position += count
        return totals

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("MicrogradLM does not support generation yet")

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=60)
            self.process.stdout.close()
