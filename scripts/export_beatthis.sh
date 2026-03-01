#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BEATTHIS_DIR="${BEATTHIS_DIR:-$ROOT/third_party/beat_this}"
ROTARY_DIR="${ROTARY_DIR:-$HOME/Development/3rdparty/rotary-embedding-torch}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CHECKPOINT="${CHECKPOINT:-$ROOT/models/beat_this-small0.ckpt}"
OUT="${OUT:-$ROOT/models/beatthis.pt}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -d "$BEATTHIS_DIR" ]]; then
  echo "Missing BeatThis repo at: $BEATTHIS_DIR" >&2
  exit 1
fi

if [[ ! -d "$ROTARY_DIR" ]]; then
  echo "Missing rotary-embedding-torch repo at: $ROTARY_DIR" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

export PYTHONPATH="$BEATTHIS_DIR:$ROTARY_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Exporting BeatThis:"
echo "  checkpoint: $CHECKPOINT"
echo "  output:     $OUT"
echo "  device:     $DEVICE"
echo "  python:     $PYTHON_BIN"

"$PYTHON_BIN" "$ROOT/scripts/beatthis_export_torchscript.py" \
  --checkpoint "$CHECKPOINT" \
  --out "$OUT" \
  --device "$DEVICE"
