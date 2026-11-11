"""Compare MicrogradLM driving a scorer against the harness's Hugging Face backend.

Runs one task twice with the same settings, once with `hf` on GPT-2 and once
with MicrogradLM, then compares every per-choice log-likelihood and the metrics.

Usage:
    python crosscheck.py [--scorer "python fake_scorer.py"] [--task hellaswag] [--limit 200]
"""

import argparse
import json
import sys

import lm_eval

from micrograd_lm import MicrogradLM


def run(model, task, limit, **kwargs):
    results = lm_eval.simple_evaluate(
        model=model, tasks=[task], limit=limit, batch_size=1, log_samples=True, **kwargs
    )
    samples = {s["doc_id"]: [r[0][0] for r in s["resps"]] for s in results["samples"][task]}
    return results["results"][task], samples


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scorer", default=f"{sys.executable} fake_scorer.py")
    parser.add_argument("--task", default="hellaswag")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output")
    args = parser.parse_args()

    hf_metrics, hf_samples = run(
        "hf", args.task, args.limit, model_args="pretrained=gpt2,dtype=float32", device="cpu"
    )
    lm = MicrogradLM(scorer=args.scorer)
    mg_metrics, mg_samples = run(lm, args.task, args.limit)
    lm.close()

    assert hf_samples.keys() == mg_samples.keys()
    diffs = [
        abs(a - b)
        for doc in hf_samples
        for a, b in zip(hf_samples[doc], mg_samples[doc], strict=True)
    ]
    report = {
        "task": args.task,
        "examples": len(hf_samples),
        "loglikelihoods": len(diffs),
        "exact": sum(d == 0 for d in diffs),
        "max_abs_diff": max(diffs),
        "hf": hf_metrics,
        "micrograd": mg_metrics,
    }
    print(json.dumps(report, indent=2, default=str))
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"report": report, "hf": hf_samples, "micrograd": mg_samples}, f)


if __name__ == "__main__":
    main()
