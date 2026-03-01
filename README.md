# BeatIt

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-CMake-blue.svg)](#build)
[![Tests](https://img.shields.io/badge/tests-ctest-orange.svg)](#tests)

Beat/downbeat tracking library for macOS with CoreML and Torch backends.

The current product focus is:

- fast analysis for single-song files
- highly accurate BPM
- stable beat phase
- accurate downbeat phase
- drift-free projected beat grid

## Current Default Path

If you run the CLI without model switches, BeatIt uses:

- backend: CoreML
- preset: `beatthis`
- BPM range clamp: `70..180`
- DBN: enabled (calmdad mode)
- sparse probe mode: enabled

`beatthis` preset defaults are defined in `src/beatit/coreml_preset.cpp`.

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

## CLI

Basic:

```bash
./build/beatit --input /path/to/audio.wav
```

Model selection:

```bash
# BeatThis CoreML (default)
./build/beatit --input training/manucho.wav

# BeatTrack CoreML
./build/beatit --input training/manucho.wav --beattrack

# BeatThis Torch
./build/beatit --input training/manucho.wav --backend torch --torch-model models/beatthis.pt
```

Torch export:

```bash
python3 -m venv .venv-beatit-export
source .venv-beatit-export/bin/activate
pip install torch pyyaml numpy einops rotary-embedding-torch

PYTHONPATH=third_party/beat_this \
python scripts/beatthis_export_torchscript.py \
  --checkpoint models/beat_this-final0.ckpt \
  --out models/beatthis.pt \
  --device cpu
```

Notes:

- `models/beatthis.pt` is exported from `models/beat_this-final0.ckpt`.
- The export script patches rotary tracing so the TorchScript graph does not hardcode CPU device creation and can run on MPS.
- Export on CPU. The resulting `models/beatthis.pt` is then usable with `--torch-device mps`.

Important options:

- `-i, --input <path>`
- `--backend <coreml|torch|beatthis>`
  `beatthis` means the external Python BeatThis runner (`scripts/beatthis_infer.py` + checkpoint), not the in-process CoreML/Torch backends.
- `--preset <beattrack|beatthis>`
- `--model <path>`
- `--min-bpm <bpm>` / `--max-bpm <bpm>` (validated in `[70,180]`)
- `--dbn` / `--no-dbn`
- `--cpu-only`
- `--log-level <error|warn|info|debug>`
- `--model-info`

Use `./build/beatit --help` for the authoritative option list.

## Library Usage

```cpp
#include "beatit/stream.h"

beatit::BeatitConfig cfg;
if (auto preset = beatit::make_coreml_preset("beatthis")) {
    preset->apply(cfg);
}

beatit::BeatitStream stream(sample_rate, cfg, true);
double start_s = 0.0;
double duration_s = 0.0;
if (stream.request_analysis_window(&start_s, &duration_s)) {
    beatit::AnalysisResult result =
        stream.analyze_window(start_s, duration_s, total_duration_s, provider);
}
```

Contract notes:

- `request_analysis_window(...)` returns the preferred seed window size/start.
- `analyze_window(...)` is a single call from the integrator side.
- In sparse mode, BeatIt may call `provider(start, duration, out_samples)` multiple times internally (left/right/interior probes) before returning the final result.
- The provider must therefore be re-entrant for arbitrary `(start, duration)` requests within the file bounds.

Provider contract:

`BeatitStream::SampleProvider`:

```cpp
using SampleProvider =
    std::function<std::size_t(double start_seconds,
                              double duration_seconds,
                              std::vector<float>* out_samples)>;
```

What BeatIt expects:

- `start_seconds` / `duration_seconds` are absolute requests on the original file timeline.
- Returned samples must be mono float PCM at the same sample rate used to construct `BeatitStream`.
- The callback must fill `*out_samples` and return the exact valid sample count.
- On out-of-range or read failure: clear `*out_samples` and return `0`.
- The callback can be called multiple times per `analyze_window(...)` call in sparse mode, so keep it re-entrant and deterministic.

## Tests

Run all:

```bash
ctest --test-dir build --output-on-failure
```

CPU-only (for environments where GPU/MPS is unavailable):

```bash
BEATIT_TEST_CPU_ONLY=1 ctest --test-dir build --output-on-failure
```

## Credits

- BeatTrack — Matthew Rice — https://github.com/mhrice/BeatTrack — MIT
- Beat This! — Francesco Foscarin, Jan Schlueter, Gerhard Widmer — https://github.com/CPJKU/beat_this — MIT
