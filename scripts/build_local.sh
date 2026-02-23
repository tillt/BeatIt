#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

BEATIT_TORCH_ROOT="${BEATIT_TORCH_ROOT:-/Users/till/Development/3rdparty/pytorch-v2.9.1}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DBEATIT_TORCH_ROOT="${BEATIT_TORCH_ROOT}" \
  "$@"
cmake --build "${BUILD_DIR}"
