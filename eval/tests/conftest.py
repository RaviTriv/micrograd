import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(EVAL_DIR / ".cache" / "huggingface"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(EVAL_DIR / ".cache" / "tiktoken"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(EVAL_DIR))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from micrograd_lm import MicrogradLM  # noqa: E402

SCORER = [sys.executable, str(EVAL_DIR / "fake_scorer.py")]


class Scorer:
    def __init__(self):
        self.process = subprocess.Popen(
            SCORER, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def ask(self, line):
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def score(self, context, continuation):
        return self.ask(json.dumps({"context": context, "continuation": continuation}))

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=60)


@pytest.fixture(scope="session")
def scorer():
    process = Scorer()
    yield process
    process.close()


@pytest.fixture(scope="session")
def lm():
    model = MicrogradLM(scorer=SCORER)
    yield model
    model.close()


@pytest.fixture(scope="session")
def gpt2():
    return AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
