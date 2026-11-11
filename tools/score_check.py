#!/usr/bin/env python3
# Usage: tools/score_check.py <gpt_score> <checkpoint> <val_shard> <block_size> <windows> [device] [jobs]

import json
import struct
import subprocess
import sys
import threading

HEADER_INTS = 256
MAGIC = 20240520


def read_shard(path):
    with open(path, "rb") as f:
        header = struct.unpack("<256i", f.read(HEADER_INTS * 4))
        if header[0] != MAGIC or header[1] != 1:
            raise SystemExit(f"{path}: unrecognised shard header")
        count = header[2]
        data = f.read(count * 2)
    if len(data) != count * 2:
        raise SystemExit(f"{path}: truncated, header says {count} tokens")
    return struct.unpack(f"<{count}H", data)


USAGE = "usage: score_check.py <gpt_score> <checkpoint> <val_shard> <block_size> <windows> [device] [jobs]"


def score_windows(scorer, checkpoint, device, tokens, block, indices, results):
    proc = subprocess.Popen(
        [scorer, "--checkpoint", checkpoint, "--device", device],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for w in indices:
        seq = tokens[w * block: w * block + block + 1]
        request = {"context": [seq[0]], "continuation": list(seq[1:])}
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        reply = json.loads(proc.stdout.readline())
        if "error" in reply:
            results[w] = reply["error"]
            break
        results[w] = (-reply["logprob"], reply["scored"])
        print(f"window {w}: scored {reply['scored']}, mean nll {-reply['logprob'] / reply['scored']:.6f}", flush=True)
    proc.stdin.close()
    proc.wait()


def main():
    if len(sys.argv) not in (6, 7, 8):
        raise SystemExit(USAGE)
    scorer, checkpoint, shard, block, windows = sys.argv[1:6]
    device = sys.argv[6] if len(sys.argv) >= 7 else "cuda"
    jobs = int(sys.argv[7]) if len(sys.argv) == 8 else 1
    block = int(block)
    windows = int(windows)
    tokens = read_shard(shard)
    available = (len(tokens) - 1) // block
    if windows > available:
        raise SystemExit(f"only {available} windows of {block} in {shard}")

    results = {}
    threads = [threading.Thread(target=score_windows, args=(
        scorer, checkpoint, device, tokens, block, list(range(j, windows, jobs)), results))
        for j in range(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    errors = [r for r in results.values() if isinstance(r, str)]
    if errors or len(results) != windows:
        raise SystemExit(f"scored {len(results)} of {windows} windows; errors: {errors[:3]}")
    total_nll = sum(results[w][0] for w in range(windows))
    total_scored = sum(results[w][1] for w in range(windows))
    per_window = [results[w][0] / results[w][1] for w in range(windows)]
    print(f"windows {windows}, targets per window {total_scored // windows}, "
          f"total targets {total_scored}")
    print(f"scorer mean nll per token {total_nll / total_scored:.6f}")
    print(f"mean of per-window means {sum(per_window) / windows:.6f}")


if __name__ == "__main__":
    main()
