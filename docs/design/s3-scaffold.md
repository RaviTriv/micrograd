# S3: repository scaffold

Stream S3, 2026-10-28. Replaces the old engine's layout with one folder per
module of the JAX-style design. The old engine is at `80357a6`.

## What changed

- Removed: `include/`, `src/`, `gpt/`, `examples/`, `images/`, `test/`, old
  `bench/`, old `tools/`, old `cmake/Fetch*.cmake`.
- Added module folders, each an `INTERFACE` target `micrograd::<name>` with one
  `<name>_test.cpp`:

  | Folder | Target | Links |
  |---|---|---|
  | `kernels/` | `micrograd::kernels` | placeholder only, owned by S2 |
  | `obs/` | `micrograd::obs` | |
  | `runtime/` | `micrograd::runtime` | kernels, obs |
  | `core/` | `micrograd::core` | runtime |
  | `primitives/` | `micrograd::primitives` | core, kernels |
  | `compiler/` | `micrograd::compiler` | core, runtime, kernels |
  | `transforms/` | `micrograd::transforms` | core, compiler |
  | `lib/nn`, `lib/optim` | `micrograd::nn`, `micrograd::optim` | transforms, primitives |
  | `lib/ckpt`, `lib/data` | `micrograd::ckpt`, `micrograd::data` | core |
  | `models/gpt/` | `micrograd::gpt` | nn |
  | `tools/` | `micrograd::tools` | gpt, optim, ckpt, data, obs |
  | `bench/` | `micrograd::bench` | gpt, transforms |

  Plus empty `configs/` and this `docs/design/`.
- `cmake/MicrogradModule.cmake`: `micrograd_module(name DEPS ...)` and
  `micrograd_test(name SOURCES ... DEPS ...)`, so each module's CMakeLists is
  a few lines.
- `cmake/MicrogradOptions.cmake`: `micrograd::options`, C++20 and the old
  warning set with `-Werror`, on every module and test.
- `-DMICROGRAD_CUDA=ON` (default OFF): enables CUDA, defaults
  `CMAKE_CUDA_ARCHITECTURES` to 80, requires CUDA 12+ and cuDNN, and defines
  `micrograd::cuda` (cudart, cuBLAS, cuBLASLt, cuDNN) for S2 to link.
- `tools/cuda_smoke.{h,cu}` and `cuda_smoke_test.cpp`, built only with CUDA:
  one kernel plus `cudnnGetVersion` and `cublasLtGetVersion`, so the CUDA
  build compiles and links something real. The kernel test skips without a GPU.
- `docker/cuda.Dockerfile`: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
  (amd64 and arm64 on Docker Hub), plus cmake, ninja, g++ and `libgtest-dev`
  so the CUDA build needs no network.
- `tools/compile_cuda.sh`: builds the image if missing, then configures and
  builds the tree with `MICROGRAD_CUDA=ON` into `build-cuda/`. Extra args go
  to CMake. `DOCKER_PLATFORM=linux/amd64` forces x86; default is native.
- `.github/workflows/ci.yml`: `format` (clang-format-18), `tidy`
  (run-clang-tidy-18 on files in the compile database, excluding `build/`),
  `cpu` (clang++-18 and g++-13, build and ctest), `cuda` (runs
  `tools/compile_cuda.sh`, compile only). Dropped the sanitizer and MNIST jobs.
- `.clang-tidy`: `HeaderFilterRegex` points at the new module folders.
- `.gitignore`: `build-*/`.
- `README.md`: placeholder.

## Why this way

- `INTERFACE` targets because no module has sources yet; each switches to
  `STATIC` when its first `.cpp` lands, and the `micrograd::` alias stays.
- Link edges follow the design page's stack, so a layering violation is a
  link error later instead of a review comment. One deviation: the design page
  lists Core above Compiler, but the compiler passes over `Program`, which
  lives in core, so here compiler depends on core. S0 can change any edge.
- Tests sit next to their module (`core/core_test.cpp`) so each stream owns
  its tests with its folder.
- GTest is found on the system first (`FIND_PACKAGE_ARGS`), fetched otherwise.
  The Docker image has it, so the CUDA build works offline.
- The CUDA option lives in `cmake/MicrogradCuda.cmake` so S2 only adds
  sources to `kernels/` and links `micrograd::cuda`.

Rejected: a single top-level `tests/` folder (crosses stream boundaries);
headers or an include prefix in `core/` and `runtime/` (S0's call; nothing
here assumes `<micrograd/...>` or `"core/..."`); installing CUDA on the CI
runner directly as before (the Mac and CI would use different toolchains).

## Testing

`cmake -S . -B build && cmake --build build && ctest --test-dir build` on
this Mac (AppleClang 21, system GTest):

```
[100%] Built target micrograd_bench_test
...
13/13 Test #13: Bench.Builds .....................   Passed    0.02 sec

100% tests passed out of 13
```

Also checked:

- Fetched GTest path (`-DCMAKE_DISABLE_FIND_PACKAGE_googletest=ON`, Ninja):
  13/13 pass.
- `run-clang-tidy` (LLVM 23) on all 13 test files: clean.
- clang-format (23) on every new `.cpp`, `.h`, `.cu`: clean.
- `-DMICROGRAD_CUDA=ON` on the Mac: fails at `enable_language(CUDA)` with
  "Compiler requires the CUDA toolkit", as expected without nvcc.
- `tools/compile_cuda.sh` under macOS `/bin/bash` 3.2 with a stub `docker`:
  arguments and the optional `--platform` expand correctly.
- `ci.yml` parses as YAML.

## Unverified

- The Docker CUDA build: image not pulled or built (disk). `cuda_smoke.cu`,
  `MicrogradCuda.cmake`'s cuDNN lookup and `libgtest-dev` in the image have
  never run.
- CI on GitHub: none of the four jobs has run. The CUDA job's image is about
  10 GB; the job frees runner disk first, which may not be enough.
- clang-format-18 and clang-tidy-18 in CI vs 23 locally: small files, so
  version drift is unlikely to matter yet.
- `cuda_smoke_test.cpp` is not covered by the tidy job (it is only in the
  compile database when CUDA is on).

## Risk

Low for the code: nothing depends on the new targets yet. The breaking part is
deleting the old engine from the working tree; S2 is currently editing old
`src/backends/cuda/` files on its own branch, so merging S3 first makes S2's
branch conflict with deletions. S2 should move its work into `kernels/` before
or while rebasing.

## How to revert

Before commit: `git checkout -- . && git clean -fd` in this worktree (also
deletes the new folders). After commit: `git revert <commit>`; the old engine
is intact at `80357a6`.
