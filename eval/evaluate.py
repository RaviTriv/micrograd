"""Evaluate a micrograd checkpoint with lm-evaluation-harness through gpt_score.

Writes one JSON file holding the harness output plus the checkpoint SHA-256,
repo commit, package versions, device, scorer command and wall time.

Usage:
    python evaluate.py --scorer build/gpt_score --checkpoint gpt.bin --device cpu \
        --tasks hellaswag --output results/gpt-2026-11-11.json [--limit N]
"""

import argparse
import hashlib
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import lm_eval

from micrograd_lm import MicrogradLM


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--tasks", default="hellaswag")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--block-size", type=int, default=1024)
    args = parser.parse_args()

    command = [args.scorer, "--checkpoint", args.checkpoint, "--device", args.device]
    record = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain")),
        "lm_eval_version": version("lm_eval"),
        "tiktoken_version": version("tiktoken"),
        "device": args.device,
        "scorer_command": command,
        "block_size": args.block_size,
        "limit": args.limit,
    }

    start = time.time()
    lm = MicrogradLM(scorer=command, block_size=args.block_size)
    results = lm_eval.simple_evaluate(
        model=lm, tasks=args.tasks.split(","), limit=args.limit, log_samples=False
    )
    lm.close()
    record["wall_time_seconds"] = round(time.time() - start, 1)
    record["harness"] = results

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(record, f, indent=2, default=str)
    for task, metrics in results["results"].items():
        print(task, json.dumps(metrics))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
