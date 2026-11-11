#!/usr/bin/env bash
# Usage: tools/fetch_fineweb.sh <dir> <train_shard_count>

set -u

if [ $# -ne 2 ]; then
  echo "usage: $0 <dir> <train_shard_count>" >&2
  exit 2
fi

dir="$1"
count_arg="$2"

script_dir="$(cd "$(dirname "$0")" && pwd)"
manifest="$script_dir/../cmake/fineweb.sha256"
base_url="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards/resolve/main"
header_bytes=1024

if [ ! -f "$manifest" ]; then
  echo "manifest not found: $manifest" >&2
  exit 2
fi

available=$(grep -c '  edu_fineweb_train_' "$manifest")

case "$count_arg" in
  '' | *[!0-9]*)
    echo "train_shard_count must be a positive integer, got '$count_arg'" >&2
    exit 2
    ;;
esac
count=$((10#$count_arg))
if [ "$count" -lt 1 ] || [ "$count" -gt "$available" ]; then
  echo "train_shard_count must be between 1 and $available, got '$count_arg'" >&2
  exit 2
fi

if command -v sha256sum >/dev/null 2>&1; then
  hash_file() { sha256sum "$1" | cut -d ' ' -f 1; }
elif command -v shasum >/dev/null 2>&1; then
  hash_file() { shasum -a 256 "$1" | cut -d ' ' -f 1; }
else
  echo "need sha256sum or shasum" >&2
  exit 2
fi

mkdir -p "$dir" || exit 1

failures=0

fetch() {
  local remote="$1"
  local path="$dir/$2"
  local expected
  expected=$(awk -v name="$remote" '$2 == name { print $1 }' "$manifest")
  if [ -z "$expected" ]; then
    echo "FAIL $remote: not in manifest" >&2
    failures=$((failures + 1))
    return
  fi

  if [ -f "$path" ]; then
    if [ "$(hash_file "$path")" = "$expected" ]; then
      echo "ok   $2 (already present)"
      return
    fi
    echo "FAIL $2: checksum mismatch, removing" >&2
    rm -f "$path"
  fi

  echo "get  $remote"
  curl -fL --retry 3 -C - -o "$path.part" "$base_url/$remote"
  local curl_status=$?
  if [ "$curl_status" -eq 33 ]; then
    echo "     server cannot resume, restarting $remote"
    rm -f "$path.part"
    curl -fL --retry 3 -o "$path.part" "$base_url/$remote"
    curl_status=$?
  fi
  if [ ! -f "$path.part" ]; then
    echo "FAIL $remote: download failed (curl exit $curl_status)" >&2
    failures=$((failures + 1))
    return
  fi
  if [ "$(hash_file "$path.part")" = "$expected" ]; then
    mv "$path.part" "$path"
    echo "ok   $2"
    return
  fi
  if [ "$curl_status" -ne 0 ]; then
    echo "FAIL $remote: download incomplete (curl exit $curl_status), rerun to resume" >&2
  else
    echo "FAIL $remote: checksum mismatch, removing" >&2
    rm -f "$path.part"
  fi
  failures=$((failures + 1))
}

fetch "edu_fineweb_val_000000.bin" "edufineweb_val_000000.bin"
for shard in $(seq 1 "$count"); do
  fetch "$(printf 'edu_fineweb_train_%06d.bin' "$shard")" "edufineweb_train_${shard}.bin"
done

present=0
total_bytes=0
train_tokens=0
val_tokens=0
val="$dir/edufineweb_val_000000.bin"
if [ -f "$val" ]; then
  size=$(wc -c < "$val")
  total_bytes=$((total_bytes + size))
  val_tokens=$(((size - header_bytes) / 2))
fi
for shard in $(seq 1 "$count"); do
  path="$dir/edufineweb_train_${shard}.bin"
  if [ -f "$path" ]; then
    size=$(wc -c < "$path")
    present=$((present + 1))
    total_bytes=$((total_bytes + size))
    train_tokens=$((train_tokens + (size - header_bytes) / 2))
  fi
done

echo "train shards: $present of $count"
echo "total bytes:  $total_bytes"
echo "train tokens: $train_tokens"
echo "val tokens:   $val_tokens"

if [ "$failures" -ne 0 ] || [ "$present" -ne "$count" ] || [ ! -f "$val" ]; then
  echo "$failures file(s) failed" >&2
  exit 1
fi
