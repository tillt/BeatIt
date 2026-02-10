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

Quick usage:

```bash
# CoreML (BeatTrack)
./build/beatit --input training/moderat.wav --ml-backend coreml --ml-beattrack

# Torch (BeatThis + DBN)
./build/beatit --input training/moderat.wav --ml-backend torch --torch-model models/beatthis.pt --ml-dbn
```

Model selection:

```bash
# 1) BeatThis CoreML (default if nothing else is selected)
./build/beatit --input training/moderat.wav

# 2) BeatTrack CoreML (first integrated CoreML model)
./build/beatit --input training/moderat.wav --ml-beattrack

# 3) BeatThis Torch
./build/beatit --input training/moderat.wav --ml-backend torch --torch-model models/beatthis.pt
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
- `--ml-backend <name>` ML backend (`coreml`, `torch`, or `beatthis` for external Python BeatThis)
- `--torch-model <path>` TorchScript model path
- `--torch-device <name>` Torch device (cpu, mps)
- `--torch-fps <hz>` Torch output fps (default 100)
- `--beatthis-dbn` enable DBN in external Python BeatThis backend (`--ml-backend beatthis`)
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
- `--ml-dbn-trace` DBN detailed trace logging (candidates, window, projection)
- `--ml-dbn-window-best` Select best window by activation energy
- `--ml-dbn-grid-align-downbeat` Align grid start to earliest downbeat peak
- `--ml-dbn-grid-strong-start` Anchor the projected grid to the strongest early beat peak
- `--ml-activations-window <start> <end>` Dump beat/downbeat activations between seconds
- `--ml-activations-max <n>` Cap activation dump rows (0 = no cap)
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

## DBN parameters (defaults + effect)

DBN (dynamic Bayesian network) takes frame‑level beat/downbeat activations and finds a
globally consistent beat path under tempo/meter priors. It is most helpful when raw
activations are noisy, sparse, or drift across long tracks.

Defaults below match `include/beatit/coreml.h` (current recommended tuning).

- `--ml-dbn` (default **off**): enable DBN decoding. When off, BeatIt uses its peak/threshold logic.
- `--ml-dbn-step <bpm>` (default **1.0**): grid step for candidate tempos. Smaller steps increase accuracy but grow runtime.
- `--ml-dbn-tol <ratio>` (default **0.05**): tolerance for inter‑beat interval deviations. Higher values allow more timing variation (better for swing/complex patterns) but can reduce stability.
- `--ml-dbn-floor <value>` (default **0.01**): activation floor for candidates (BeatIt mode). In `calmdad` mode, activations are epsilon‑clamped and this floor is ignored.
- `--ml-dbn-bpb <beats>` (default **4**): beats per bar assumption. Set to 3 for waltz‑like material or odd meters if needed.
- `--ml-dbn-no-downbeat` (default **false**): ignore downbeat activations entirely. Useful if downbeats are unreliable.
- `--ml-dbn-downbeat <w>` (default **1.0**): weight for downbeat activations. Higher emphasizes bar alignment; lower de‑emphasizes downbeats.
- `--ml-dbn-mode <name>` (default **calmdad**): DBN mode. `calmdad` mirrors BeatThis/madmom’s 2‑class observation style with beats‑per‑bar of 3 and 4; `beatit` keeps the original BeatIt emission model.
- `--ml-dbn-tempo-pen <w>` (default **0.05**): penalty per BPM change across the path. Higher values enforce stricter tempo continuity.
- `--ml-dbn-tempo-prior <w>` (default **0.0**): bias toward classic tempo ranges (near 120). Increase if your catalog is mostly in a narrow BPM range.
- `--ml-dbn-max-cand <n>` (default **4096**): cap on candidate frames. Higher improves coverage but costs runtime.
- `--ml-dbn-reward <w>` (default **0.7**): reward for valid transitions. Higher can prefer longer continuous paths even with weaker activations.
- `--ml-dbn-all-cand` (default **true**): use all frames as candidates (no peak picking). Helps when activations are extremely sparse or the model “goes silent”.
- `--ml-dbn-lambda <w>` (default **100.0**): BeatThis/madmom‑style transition strength (used in `calmdad` mode).
- `--ml-dbn-trace` (default **off**): emit detailed DBN trace logs (first‑2s activation stats, candidate previews, phase selection, grid projection head).
- `--ml-dbn-window-best` (default **off**): pick the DBN window by highest mean activation energy (phase energy if enabled, otherwise beat activation). Overrides intro/mid/outro selection.
- `--ml-dbn-grid-align-downbeat` (default **off**): shift the projected grid to align its first downbeat with the earliest strong downbeat peak within the first bar.
- `--ml-dbn-grid-strong-start` (default **off**): anchor the projected grid to the strongest beat peak in the first bar (instead of the earliest peak/downbeat).

### Starter sets

Ambient (few activations, long breaks):
```
--ml-dbn --ml-dbn-floor 0.01 --ml-dbn-tol 0.15 --ml-dbn-reward 0.6 --ml-dbn-max-cand 4096 --ml-dbn-all-cand
```

Straight techno (strong, continuous activations):
```
--ml-dbn --ml-dbn-floor 0.06 --ml-dbn-tol 0.03 --ml-dbn-tempo-pen 0.3 --ml-dbn-step 0.5
```

Breakbeat (strong but complex patterns):
```
--ml-dbn --ml-dbn-floor 0.03 --ml-dbn-tol 0.2 --ml-dbn-tempo-pen 0.1 --ml-dbn-downbeat 0.7
```

Universal (balanced, works across mixed catalogs) — matches current defaults:
```
--ml-dbn --ml-dbn-floor 0.01 --ml-dbn-tol 0.1 --ml-dbn-tempo-pen 0.05 --ml-dbn-reward 0.7 --ml-dbn-max-cand 4096 --ml-dbn-all-cand
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
the app bundle resource `beatit.mlmodelc`. The CoreML model expects:

- input: `input` (shape `[1, 1, frames, 81]`, float32)
- outputs: `beat` and `downbeat` (frame-wise activations)

Override names/paths via `beatit::CoreMLConfig` or `--model`. For BeatTrack exports, use
`.mlpackage` and compile to `.mlmodelc` with `xcrun coremlc compile`.

### Torch (BeatThis)

BeatThis runs via TorchScript with optional DBN postprocessing. Export locally with:

```bash
scripts/export_beatthis.sh
```

The export script expects BeatThis in `third_party/beat_this` and an external
`rotary-embedding-torch` clone (default: `~/Development/3rdparty/rotary-embedding-torch`).
Override via `BEATTHIS_DIR` or `ROTARY_DIR`.

#### 30‑second training windows vs. full‑length tracks

BeatThis is trained on fixed‑length excerpts. For full songs we split the problem into two
separate decisions:

- **Global tempo**: estimated from longer context (more stable and less sensitive to local
  ambiguities).
- **Phase anchor** (downbeat position): estimated from a clean, high‑energy window and then
  projected across the track.

This avoids tempo drift from short windows while keeping downbeat alignment responsive to
local evidence. When needed, we only do **phase flips** (bar/beat) without changing the
global BPM.

## Credits

- ML model: BeatTrack — Matthew Rice — https://github.com/mhrice/BeatTrack — MIT.
- ML model: Beat This! — Francesco Foscarin, Jan Schlueter, Gerhard Widmer — https://github.com/CPJKU/beat_this — MIT.
- Library: rotary-embedding-torch — Phil Wang — https://github.com/lucidrains/rotary-embedding-torch — MIT.
- Training data: Ballroom dataset (ISMIR 2004 tempo contest, MTG UPF) — https://mtg.upf.edu/ismir2004/contest/tempoContest/ — license per dataset provider (not redistributed here).
