#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: eval/run.sh <checkpoint> <cpu|cuda> [tasks, comma-separated, default hellaswag]" >&2
  echo "env:   GPT_SCORE=path to the scorer (default build/gpt_score), LIMIT=examples per task" >&2
  exit 2
fi

eval_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(dirname "$eval_dir")
checkpoint=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
device=$2
tasks=${3:-hellaswag}
scorer=${GPT_SCORE:-$repo_dir/build/gpt_score}

[[ -f $checkpoint ]] || { echo "run.sh: no checkpoint at $checkpoint" >&2; exit 1; }
[[ -x $scorer ]] || { echo "run.sh: no scorer executable at $scorer (set GPT_SCORE)" >&2; exit 1; }

export HF_HOME=${HF_HOME:-$eval_dir/.cache/huggingface}
export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-$eval_dir/.cache/tiktoken}
source "$eval_dir/.venv/bin/activate"

name=$(basename "$checkpoint")
output=$eval_dir/results/${name%.*}-$(date +%Y-%m-%d-%H%M%S).json

cd "$eval_dir"
python evaluate.py --scorer "$scorer" --checkpoint "$checkpoint" --device "$device" \
  --tasks "$tasks" --output "$output" ${LIMIT:+--limit "$LIMIT"}
