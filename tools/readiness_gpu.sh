#!/usr/bin/env bash
# Usage: [SKIP_A=1] [KEEP_CKPT=1] OUT=<dir> FINEWEB_DIR=<dir> tools/readiness_gpu.sh step0|resume [muon]|score|precision bf16|tf32|sweep <name> <accum> <steps> <lr> <eval_interval> [seed]

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build-cuda-cublas-on}"
OUT="${OUT:?set OUT}"
FINEWEB_DIR="${FINEWEB_DIR:?set FINEWEB_DIR}"
GPUS="${GPUS:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
mkdir -p "$OUT"

train() {
  local ranks="$1" name="$2"
  shift 2
  MASTER_PORT=$((29600 + RANDOM % 300)) bash "$ROOT/tools/launch.sh" "$ranks" \
    stdbuf -oL "$BUILD/train_gpt" --data-format fineweb --fineweb-dir "$FINEWEB_DIR" \
    --device cuda --no-recompute --seed 1337 "$@" >>"$OUT/$name.log" 2>&1
}

RESUME_FLAGS=(--bf16 --pad-vocab --batch-size 8 --block-size 1024 --max-iters 300
  --warmup-iters 30 --lr 6e-4 --min-lr 6e-5 --eval-interval 50 --eval-iters 10
  --throughput-interval 10)

case "${1:-}" in
  step0)
    train 1 step0 --bf16 --pad-vocab --batch-size 8 --block-size 1024 --max-iters 1 \
      --eval-interval 1 --eval-iters 10 --checkpoint-interval 100000 \
      --log "$OUT/step0.csv" --checkpoint "$OUT/step0.bin"
    echo "rc $?"
    grep -E "^(flags|step 0|vocab|activations)" "$OUT/step0.log"
    ;;
  resume)
    tag="${2:-baseline}"
    extra=()
    [ "$tag" = muon ] && extra=(--muon-lr 0.02)
    a="resume-$tag-A" b="resume-$tag-B"
    start=$(date +%s)
    if [ "${SKIP_A:-0}" -ne 1 ]; then
      train "$GPUS" "$a" "${RESUME_FLAGS[@]}" "${extra[@]}" --checkpoint-interval 100000 \
        --log "$OUT/$a.csv" --checkpoint "$OUT/$a.bin"
      echo "run A rc $? in $(($(date +%s) - start)) s"
    fi
    rm -f "$OUT/$b".*
    kill_at=$((150 + RANDOM % 50))
    echo "run B kill target: step $kill_at"
    train "$GPUS" "$b" "${RESUME_FLAGS[@]}" "${extra[@]}" --checkpoint-interval 100 \
      --log "$OUT/$b.csv" --checkpoint "$OUT/$b.bin" &
    launcher=$!
    while kill -0 "$launcher" 2>/dev/null; do
      last=$(grep -oE "^step [0-9]+:" "$OUT/$b.log" 2>/dev/null | tail -n 1 | grep -oE "[0-9]+")
      if [ -n "$last" ] && [ "$last" -ge "$kill_at" ]; then
        pkill -9 -f "$BUILD/train_gpt.*$OUT/$b.bin"
        echo "killed run B after log line step $last"
        break
      fi
      sleep 0.5
    done
    wait "$launcher"
    echo "run B launcher rc $? (expected nonzero after SIGKILL)"
    ls -la "$OUT/$b.bin"*
    echo "== arch change refusal"
    train 1 "$b-archcheck" "${RESUME_FLAGS[@]}" "${extra[@]}" --n-layer 6 \
      --resume "$OUT/$b.bin" --log "$OUT/$b-archcheck.csv" --checkpoint "$OUT/$b-archcheck.bin"
    echo "arch-changed resume rc $?"
    tail -n 2 "$OUT/$b-archcheck.log"
    echo "== resume"
    echo "=== resume from $OUT/$b.bin" >>"$OUT/$b.log"
    train "$GPUS" "$b" "${RESUME_FLAGS[@]}" "${extra[@]}" --checkpoint-interval 100 \
      --resume "$OUT/$b.bin" --log "$OUT/$b.csv" --checkpoint "$OUT/$b.bin"
    echo "resumed run B rc $?"
    grep -E "resuming|final val" "$OUT/$b.log"
    grep -E "final val" "$OUT/$a.log"
    awk -F, 'NR == FNR { if ($1 ~ /^[0-9]+$/) { t[$1] = $2; v[$1] = $3 } next }
      $1 ~ /^[0-9]+$/ { bt[$1] = $2; bv[$1] = $3 }
      END {
        for (s in bt) if (s + 0 >= 200 && (s in t)) {
          dt = bt[s] - t[s]; if (dt < 0) dt = -dt
          dv = bv[s] - v[s]; if (dv < 0) dv = -dv
          printf "step %s: A train %s val %s | B train %s val %s | diff train %.6f val %.6f\n", s, t[s], v[s], bt[s], bv[s], dt, dv
          if (dt > max) max = dt; if (dv > max) max = dv; rows++
        }
        printf "%d rows from step 200 compared, max difference %.6f\n", rows, max
      }' "$OUT/$a.csv" "$OUT/$b.csv"
    ;;
  score)
    ckpt="$OUT/resume-baseline-A.bin"
    cmake --build "$BUILD" -j "$(nproc)" --target gpt_score >"$OUT/build-gpt_score.log" 2>&1
    echo "build gpt_score rc $?"
    windows=$((8 * 10))
    python3 -I "$ROOT/tools/score_check.py" "$BUILD/gpt_score" "$ckpt" \
      "$FINEWEB_DIR/edufineweb_val_000000.bin" 1024 "$windows" cuda "${JOBS:-8}" \
      >"$OUT/score-check.log" 2>&1
    echo "score_check rc $?"
    tail -n 3 "$OUT/score-check.log"
    echo "trainer: $(grep 'final val' "$OUT/resume-baseline-A.log")"
    ;;
  precision)
    prec="${2:?bf16 or tf32}"
    batch="${PREC_BATCH:-16}"
    accum="${PREC_ACCUM:-16}"
    iters="${PREC_ITERS:-475}"
    flag=--bf16
    [ "$prec" = tf32 ] && flag=--tf32
    start=$(date +%s)
    train "$GPUS" "precision-$prec" "$flag" --pad-vocab --batch-size "$batch" \
      --grad-accum-steps "$accum" --block-size 1024 --max-iters "$iters" \
      --warmup-iters 20 --lr 6e-4 --min-lr 6e-5 --eval-interval 50 --eval-iters 100 \
      --checkpoint-interval 100 --throughput-interval 10 \
      --log "$OUT/precision-$prec.csv" --checkpoint "$OUT/precision-$prec.bin"
    echo "precision $prec rc $? in $(($(date +%s) - start)) s"
    grep -E "final val|gpu memory|error" "$OUT/precision-$prec.log" | tail -n 3
    grep "tok/s" "$OUT/precision-$prec.log" | tail -n 1
    ;;
  sweep)
    name="${2:?name}" accum="${3:?accum}" steps="${4:?steps}" lr="${5:?lr}"
    eval_every="${6:?eval interval}" seed="${7:-1337}"
    warmup=$(((steps * 2 + 99) / 100))
    [ "$warmup" -lt 10 ] && warmup=10
    min_lr=$(python3 -c "print($lr / 10)")
    run="sweep-$name"
    rm -f "$OUT/$run".*
    start=$(date +%s)
    train "$GPUS" "$run" --bf16 --pad-vocab --batch-size 16 --grad-accum-steps "$accum" \
      --block-size 1024 --max-iters "$steps" --warmup-iters "$warmup" --lr "$lr" \
      --min-lr "$min_lr" --eval-interval "$eval_every" --eval-iters 100 \
      --checkpoint-interval 1000000 --throughput-interval 10 --seed "$seed" \
      --log "$OUT/$run.csv" --checkpoint "$OUT/$run.bin"
    echo "$run rc $? in $(($(date +%s) - start)) s, warmup $warmup, min lr $min_lr"
    grep -E "final val|error|non-finite|nan" "$OUT/$run.log" | tail -n 2
    grep "tok/s" "$OUT/$run.log" | tail -n 1
    if [ "${KEEP_CKPT:-0}" -ne 1 ]; then
      rm -f "$OUT/$run".bin*
    fi
    ;;
  *)
    echo "usage: $0 step0|resume [muon]|score|precision bf16|tf32|sweep <name> <accum> <steps> <lr> <eval_interval> [seed]" >&2
    exit 2
    ;;
esac
