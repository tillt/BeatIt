#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_release_universal}"

if [[ -z "${BEATIT_TORCH_ROOT:-}" ]]; then
  if [[ -d "/Users/till/Development/3rdparty/pytorch-v2.9.1/torch/share/cmake/Torch" ]]; then
    BEATIT_TORCH_ROOT="/Users/till/Development/3rdparty/pytorch-v2.9.1"
  elif [[ -d "/opt/homebrew/opt/libtorch/share/cmake/Torch" ]]; then
    BEATIT_TORCH_ROOT="/opt/homebrew/opt/libtorch"
  elif [[ -d "/usr/local/opt/libtorch/share/cmake/Torch" ]]; then
    BEATIT_TORCH_ROOT="/usr/local/opt/libtorch"
  fi
fi

if [[ -z "${BEATIT_TORCH_ROOT:-}" ]]; then
  echo "BEATIT_TORCH_ROOT must be set or libtorch must be installed via Homebrew." 1>&2
  exit 1
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DBEATIT_TORCH_ROOT="${BEATIT_TORCH_ROOT}" \
  "$@"

cmake --build "${BUILD_DIR}"
