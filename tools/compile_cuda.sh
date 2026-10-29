#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${MICROGRAD_CUDA_IMAGE:-micrograd-cuda:12.8.1}"
build_dir="${MICROGRAD_CUDA_BUILD_DIR:-build-cuda}"
platform_args=()
if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
  platform_args=(--platform "$DOCKER_PLATFORM")
fi

if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build ${platform_args[@]+"${platform_args[@]}"} \
    -f "$repo_root/docker/cuda.Dockerfile" \
    -t "$image" \
    "$repo_root/docker"
fi

docker run --rm ${platform_args[@]+"${platform_args[@]}"} \
  --user "$(id -u):$(id -g)" \
  -v "$repo_root:/src" \
  -w /src \
  "$image" \
  bash -c 'cmake -S . -B "$0" -G Ninja -DMICROGRAD_CUDA=ON "${@}" && cmake --build "$0"' \
  "$build_dir" "$@"
