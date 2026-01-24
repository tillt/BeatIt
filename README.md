# BeatIt

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-CMake-blue.svg)](#building)
[![Tests](https://img.shields.io/badge/tests-ctest-orange.svg)](#tests)
[![Homebrew](https://img.shields.io/badge/homebrew-tap-181717.svg)](#macos-homebrew)

Minimal BPM/beat tracker for macOS with CoreML and Torch (BeatThis) backends.

## Goals

- macOS-first, optimized for Apple's DSP stack and CoreML.
- Small, readable core suitable for embedding as a library.
- Fast beat and downbeat estimation with minimal dependencies.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

Homebrew (HEAD):

```bash
brew install --build-from-source ./Formula/beatit.rb
```

## CLI usage

```bash
./build/beatit --input /path/to/audio.mp3
```

Options:

- `-i, --input <path>` audio file (MP3/MP4/WAV/AIFF/CAF)
- `--sample-rate <hz>` resample before analysis
- `--model <path>` CoreML model path (`.mlmodelc`)
- `--ml-input <name>` CoreML input feature name
- `--ml-beat <name>` CoreML beat output name
- `--ml-downbeat <name>` CoreML downbeat output name
- `--ml-sr <hz>` CoreML feature sample rate
- `--ml-frame <frames>` CoreML feature frame size
- `--ml-hop <frames>` CoreML feature hop size
- `--ml-mels <bins>` CoreML mel bin count
- `--ml-window-hop <frames>` CoreML window hop (fixed-frame mode)
- `--ml-threshold <value>` CoreML activation threshold
- `--ml-backend <name>` ML backend (coreml, beatthis, torch)
- `--torch-model <path>` TorchScript model path
- `--torch-device <name>` Torch device (cpu, mps)
- `--torch-fps <hz>` Torch output fps (default 100)
- `--beatthis-dbn` BeatThis enable DBN postprocess
- `--ml-beattrack` use BeatTrack CoreML defaults
- `--ml-tempo-window <pct>` percent window around classic BPM
- `--ml-prefer-double` prefer double-time BPM if stronger
- `--ml-refine-constant` post-process beats into a constant grid
- `--ml-refine-csv` print CSV for constant beat events
- `--ml-refine-downbeat` use model downbeats to anchor bar phase
- `--ml-refine-halfbeat` enable half-beat phase correction
- `--ml-dbn` use DBN-style beat decoder
- `--ml-dbn-step <bpm>` DBN BPM step size
- `--ml-dbn-tol <ratio>` DBN interval tolerance (0-1)
- `--ml-dbn-floor <value>` DBN activation floor
- `--ml-dbn-bpb <beats>` DBN beats-per-bar (default 4)
- `--ml-dbn-no-downbeat` DBN ignore downbeat activations
- `--ml-dbn-downbeat <w>` DBN downbeat weight
- `--ml-dbn-tempo-pen <w>` DBN tempo change penalty per BPM
- `--ml-dbn-tempo-prior <w>` DBN tempo prior weight
- `--ml-dbn-max-cand <n>` DBN max candidate frames
- `--ml-dbn-reward <w>` DBN transition reward
- `--ml-dbn-all-cand` DBN use every frame as candidate
- `--ml-cpu-only` force CoreML CPU-only execution
- `--ml-info` print CoreML model metadata
- `--info` print decoded audio stats
- `-h, --help` show help

CLI is for development/testing; the reusable API lives in `include/beatit/analysis.h`.

## Streaming API (large files)

For long files, use `beatit::BeatitStream` to feed audio in chunks without loading
everything into memory. It produces the same `AnalysisResult` after `finalize()`.

```cpp
#include "beatit/stream.h"

beatit::CoreMLConfig ml_cfg;
beatit::BeatitStream stream(sample_rate, ml_cfg, true);

stream.push(chunk.data(), chunk.size());
beatit::AnalysisResult result = stream.finalize();
```

## Constant Beat Refiner

CBR (Constant Beat Refiner) turns beat/downbeat activations into a stable beat grid of
`BeatEvent`s with bar/marker annotations. Optional switches: downbeat anchoring and
half-beat correction.

```bash
./build/beatit --input training/moderat.wav --ml-refine-constant --ml-refine-csv --ml-cpu-only
```

## Model backends

### CoreML (BeatTrack)

Default CoreML settings match BeatTrack (torchaudio MelSpectrogram; no log scaling):

- sample rate: 44.1 kHz
- frame size: 2048
- hop size: 441
- mel bins: 81
- input layout: `[1, 1, frames, bins]`
- fixed frames: 3000 (30s at 100 fps, truncated/padded)

The loader checks `models/beatit.mlmodelc`, Homebrew `share/beatit/beatit.mlmodelc`, then
the app bundle resource `beatit.mlmodelc`. The CoreML model is expected to expose:

- input: `input` (shape `[1, 1, frames, 81]`, float32)
- outputs: `beat` and `downbeat` (frame-wise activations)

You can override names/paths by editing `beatit::CoreMLConfig` or using the CLI `--model` flag.
If you are exporting a model from another repository (e.g. BeatTrack), use the `--ml-*` flags
to match its input/output names and feature settings. For BeatTrack exports, use `.mlpackage`
and compile to `.mlmodelc` with `xcrun coremlc compile`.

### Torch (BeatThis)

BeatThis runs via TorchScript with optional DBN postprocessing. Export locally with:

```bash
scripts/export_beatthis.sh
```

The export script expects the BeatThis repo in `third_party/beat_this` and an external
clone of `rotary-embedding-torch` (default: `~/Development/3rdparty/rotary-embedding-torch`).
Override paths by setting `BEATTHIS_DIR` or `ROTARY_DIR`.

## Credits

- ML model: BeatTrack — Matthew Rice — https://github.com/mhrice/BeatTrack — MIT.
- Training data: Ballroom dataset (ISMIR 2004 tempo contest, MTG UPF) — https://mtg.upf.edu/ismir2004/contest/tempoContest/ — license per dataset provider (not redistributed here).
