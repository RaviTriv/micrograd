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
| `scorecard.py` | The scorecard: Part A multiple choice, Part B generation, judging and pairwise |
| `chat/answer.py`, `chat/judge.py` | Part B prompts, stops, judge interface, grade parsing and headline |
| `chat/questions.jsonl`, `chat/dev.jsonl` | Part B held-out (100) and dev (30) questions with reference answers. Drafts awaiting owner review |
| `serve.py` | Local playground; `--blind` mode for owner votes on saved answers |
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

## Scorecard

Every model goes through the same two parts; the plan is
`docs/plans/active/2026-11-13-taby-mm-useful.md`. `<model>` is an exported folder or
a Hugging Face id. A folder needs `--name`; an id defaults to its last part,
lower-cased. A Hugging Face download fetches only config, generation config,
tokenizer and safetensors files. `knowledge` and `generate` refuse to replace an
existing result unless `--overwrite` is given. Set the cache variables first:

```bash
cd eval && export HF_HOME=$PWD/.cache/huggingface TIKTOKEN_CACHE_DIR=$PWD/.cache/tiktoken HF_HUB_DISABLE_XET=1
```

### Part A: multiple choice

```bash
.venv/bin/python scorecard.py knowledge ../thunder-results/long-a100x8-20261112/hf-final --name taby-mm --device mps
.venv/bin/python scorecard.py knowledge HuggingFaceTB/SmolLM2-135M-Instruct --device mps
```

Runs HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande and MMLU through the
harness's `hf` backend: float32, batch size 1, 0-shot, no chat template, full sets.
Writes `results/<name>-scorecard-a.json`: `summary` holds per task the reported
metric (`acc_norm` where the task has it, else `acc`) with its stderr, plus `acc`;
provenance is the model path or Hugging Face id and revision, `weights_sha256`,
repo commit and dirty flag, package versions, device, date and wall time;
`harness` is the harness's own output. `--limit N` is for smoke tests only.

### Part B: generation

```bash
.venv/bin/python scorecard.py generate <model> --name <name> --set questions --device mps
.venv/bin/python scorecard.py generate <model> --name <name> --set dev --device mps
```

Greedy, at most 200 new tokens. A model whose tokenizer has a chat template gets one
user turn through it (with the template's default system prompt, if any) and stops
at its end-of-turn token. A model without one is a base model: it gets
`Question: <question>\nAnswer:` and stops at the first blank line or end of text.
Writes `results/chat/<name>-<set>.jsonl` (one row per question: prompt, answer, new
token count, stop reason) and `results/chat/<name>-<set>.meta.json` (provenance,
settings, question-set hash, wall time).

The 100 held-out questions are never used to pick training data or checkpoints;
tune on `dev`. Both sets are drafts until the owner has read every reference answer.

### Part B: judging

```bash
.venv/bin/python scorecard.py judge results/chat/<name>-questions.jsonl --judge <judge>
.venv/bin/python scorecard.py pairwise results/chat/<new>-questions.jsonl results/chat/<old>-questions.jsonl --judge <judge> --seed 0
```

Both read the question and reference from the current question set, found through
the generation's `.meta.json`. If that file's SHA-256 differs from the one recorded at
generation, they refuse unless `--allow-stale` is given; `pairwise` also refuses two
files generated from different question sets.

`judge` grades each answer 1 to 5 for correctness and helpfulness against the
reference. Headline: percent of answers with both at 4 or more, overall and per
group. `pairwise` asks which of two answers is better, with the order shown to the
judge randomised per question from `--seed`; the new model (first file) ships only if
`a_wins` exceeds `b_wins`. Both write the judge id, the exact prompt text, the input
files' SHA-256, the question set hash and every raw reply next to the inputs. Each
reply is appended to `<output>.rows.jsonl` as it arrives, so a rerun after a crash
skips the questions already graded. A reply that cannot be parsed, or a backend
error, is kept with an `error` field; such rows are left out of the headline and
counted under `errors`, and are retried on the next run.

`--judge fake` is a deterministic word-overlap judge for tests and dry runs; its
numbers mean nothing. `--judge llm:<model id>` is the slot for the real judge and
raises `JudgeNotConfigured` until the owner picks the judge model and who pays for
it. A backend is any object with an `id` and `complete(prompt) -> str`, added to
`make_judge` in `chat/judge.py`.

### Owner check: blind side by side

```bash
.venv/bin/python serve.py --blind results/chat/<new>-questions.jsonl results/chat/<old>-questions.jsonl --count 20 --seed 0
```

Open http://127.0.0.1:8765. Shows `--count` seeded random questions, one at a time,
with the two answers on shuffled sides and no model names; keys 1, 2 and 3 vote left,
tie and right. Each vote is appended to
`results/chat/votes-<new>-vs-<old>.jsonl` (or `--votes`) with the hidden model
names, so restarting resumes where it stopped.
