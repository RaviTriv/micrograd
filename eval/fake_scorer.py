"""Stand-in for gpt_score that answers with Hugging Face's GPT-2 in PyTorch.

Speaks the gpt_score protocol: one JSON request per line on stdin,
{"context": [ids], "continuation": [ids]}, one JSON response per line on
stdout, {"logprob": float, "greedy": bool, "scored": int} or {"error": str}.

Usage:
    python fake_scorer.py [--checkpoint gpt2] [--device cpu] [--block-size 1024]
"""

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM


def parse_request(line, vocab_size, block_size):
    request = json.loads(line)
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    context = request.get("context")
    continuation = request.get("continuation")
    for name, ids in (("context", context), ("continuation", continuation)):
        if not isinstance(ids, list):
            raise ValueError(f"{name} must be a list of token ids")
        for token in ids:
            if type(token) is not int or not 0 <= token < vocab_size:
                raise ValueError(f"{name} has invalid token id {token!r}")
    if not continuation:
        raise ValueError("continuation is empty")
    if len(continuation) > block_size:
        raise ValueError(f"continuation longer than block size {block_size}")
    return context, continuation


@torch.inference_mode()
def score(model, context, continuation, block_size, device):
    tokens = (context + continuation)[-block_size:]
    scored = min(len(continuation), len(tokens) - 1)
    if scored == 0:
        return {"logprob": 0.0, "greedy": True, "scored": 0}
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    logprobs = torch.log_softmax(model(inputs).logits[0], dim=-1)[-scored:]
    targets = torch.tensor(tokens[-scored:], dtype=torch.long, device=device)
    picked = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
    greedy = bool((logprobs.argmax(dim=-1) == targets).all())
    return {"logprob": float(picked.sum()), "greedy": greedy, "scored": scored}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32)
    model.to(args.device).eval()
    vocab_size = model.config.vocab_size
    print(f"fake_scorer: loaded {args.checkpoint} on {args.device}", file=sys.stderr)

    for line in sys.stdin:
        try:
            context, continuation = parse_request(line, vocab_size, args.block_size)
            response = score(model, context, continuation, args.block_size, args.device)
        except Exception as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
