#!/usr/bin/env bash
# Usage: tools/launch.sh <nproc> <command> [args...]
# Starts <nproc> copies of <command> on this machine with RANK, LOCAL_RANK,
# WORLD_SIZE, MASTER_ADDR and MASTER_PORT set. If one copy fails, the others
# are stopped and its exit code is returned.

set -u

if [ $# -lt 2 ]; then
  echo "usage: $0 <nproc> <command> [args...]" >&2
  exit 2
fi

nproc_arg="$1"
shift
case "$nproc_arg" in
  '' | *[!0-9]*)
    echo "nproc must be a positive integer, got '$nproc_arg'" >&2
    exit 2
    ;;
esac
if [ "$nproc_arg" -lt 1 ]; then
  echo "nproc must be at least 1" >&2
  exit 2
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="$nproc_arg"

pids=()

stop_all() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null
  done
}

trap 'stop_all; exit 130' INT TERM

for ((rank = 0; rank < nproc_arg; rank++)); do
  RANK="$rank" LOCAL_RANK="$rank" "$@" &
  pids+=("$!")
done

status=0
if [ "${BASH_VERSINFO[0]}" -gt 4 ] ||
  { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -ge 3 ]; }; then
  for ((i = 0; i < nproc_arg; i++)); do
    wait -n
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$status" -eq 0 ]; then
      status="$rc"
      stop_all
    fi
  done
else
  for pid in "${pids[@]}"; do
    wait "$pid"
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$status" -eq 0 ]; then
      status="$rc"
    fi
  done
fi

exit "$status"
