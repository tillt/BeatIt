#!/usr/bin/env bash
set -euo pipefail

TORCH_SRC="${TORCH_SRC:-/Users/till/Development/3rdparty/pytorch-v2.9.1}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if [[ ! -d "${TORCH_SRC}" ]]; then
  echo "TORCH_SRC not found: ${TORCH_SRC}" 1>&2
  exit 1
fi

pushd "${TORCH_SRC}" >/dev/null

export CMAKE_OSX_ARCHITECTURES="arm64;x86_64"
export BUILD_PYTHON=0
export BUILD_SHARED_LIBS=1
export USE_CUDA=0
export USE_NCCL=0
export USE_DISTRIBUTED=0
export USE_MPS=1

${PYTHON_BIN} tools/build_libtorch.py

popd >/dev/null

echo "Done. Libtorch should be under: ${TORCH_SRC}/build/libtorch"
