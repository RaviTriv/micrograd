# eval

Evaluates micrograd GPT checkpoints with EleutherAI's lm-evaluation-harness
(0.4.13). Python does datasets, prompts, tokenising (`tiktoken`, `gpt2`
encoding) and scoring; the C++ `gpt_score` executable only runs the forward
pass, over a line-based JSON protocol on stdin/stdout.

| File | Purpose |
| --- | --- |
| `micrograd_lm.py` | Harness model (`MicrogradLM`, registered as `micrograd`) that sends token ids to a scorer process |
| `fake_scorer.py` | Stand-in for `gpt_score` that answers with Hugging Face's `gpt2` in PyTorch |
| `evaluate.py` | Runs the harness through `MicrogradLM` and writes one results JSON with provenance |
| `run.sh` | Entry point for evaluating a checkpoint |
| `crosscheck.py` | Compares `MicrogradLM` against the harness's built-in `hf` backend, per log-likelihood |
| `export_hf.py` | Converts a checkpoint into a Hugging Face GPT-2 model folder |
| `tests/` | Protocol and wrapper tests against the fake scorer, and the exporter test against `gpt_score` |
| `results/gpt2-hellaswag-reference.json` | OpenAI GPT-2 124M on full HellaSwag, the bar to beat |

## Setup

Python 3.10 or newer (3.12 used here). All caches stay inside `eval/.cache/`.

```bash
python3 -m venv eval/.venv
eval/.venv/bin/pip install -r eval/requirements.txt
```

The first run downloads GPT-2's weights (about 550 MB) and HellaSwag into
`eval/.cache/huggingface`. If the download fails with a 403 from a `cas`/`xet`
host, set `HF_HUB_DISABLE_XET=1`.

## Tests

```bash
cd eval && .venv/bin/python -m pytest tests -q
```

## GPT-2 reference

```bash
cd eval && HF_HOME=$PWD/.cache/huggingface .venv/bin/lm-eval run --model hf \
  --model_args pretrained=gpt2 dtype=float32 --tasks hellaswag --device cpu \
  --batch_size 1 --output_path .cache/reference
```

The harness writes `.cache/reference/gpt2/results_<timestamp>.json`; the copy
in `results/gpt2-hellaswag-reference.json` came from this command on an Apple
M3 Pro CPU.

To check the wrapper reproduces that backend exactly (200 examples, about 3 min):

```bash
cd eval && HF_HOME=$PWD/.cache/huggingface .venv/bin/python crosscheck.py --limit 200
```

`crosscheck.py --scorer "../build/gpt_score --checkpoint <gpt2 weights> --device cpu"`
runs the same comparison against the C++ scorer.

## Evaluating a checkpoint

```bash
eval/run.sh path/to/gpt.bin cpu
```

Arguments: checkpoint, device (`cpu` or `cuda`), optional comma-separated tasks
(default `hellaswag`). `GPT_SCORE` overrides the scorer path (default
`build/gpt_score`); `LIMIT=N` evaluates only the first N examples per task.

## Exporting to Hugging Face

```bash
cd eval && HF_HOME=$PWD/.cache/huggingface .venv/bin/python export_hf.py path/to/gpt.bin path/to/out
```

Writes `config.json`, `model.safetensors` (float32) and GPT-2's tokenizer
files, loadable with `GPT2LMHeadModel.from_pretrained("path/to/out")` or the
harness's `hf` backend (`--model_args pretrained=path/to/out`). Query, key and
value become `c_attn` in that order, a padded vocabulary is cut back to 50257
rows, and GELU is `gelu_new`. The exporter refuses other file versions,
non-float32 parameters, and checkpoints with `rotary`, `qk_norm` or
`relu_squared` on, which GPT-2 cannot express.

`tests/test_export_hf.py` checks the exported model's log-probs against
`gpt_score` on a small random checkpoint. It uses `build/gpt_score` (or
`GPT_SCORE`) and skips if that is missing.

## Results file

`run.sh` writes `eval/results/<checkpoint name>-<date>-<time>.json` with:

- `checkpoint`, `checkpoint_sha256`
- `repo_commit` (`git rev-parse HEAD`) and `repo_dirty`
- `lm_eval_version`, `tiktoken_version`, `device`, `scorer_command`, `block_size`, `limit`
- `wall_time_seconds`
- `harness`: the harness's own output (`results` with `acc`, `acc_norm` and
  their standard errors, task `versions`, `n-shot`, task configs, environment)

HellaSwag reports two metrics: `acc` picks the ending with the highest summed
log-likelihood, `acc_norm` divides by the ending's length in characters first.
