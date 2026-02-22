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
./build/beatit --input training/manucho.wav --ml-beattrack

# BeatThis Torch
./build/beatit --input training/manucho.wav --ml-backend torch --torch-model models/beatthis.pt
```

Important options:

- `-i, --input <path>`
- `--ml-backend <coreml|torch|beatthis>`
- `--ml-preset <beattrack|beatthis>`
- `--model <path>`
- `--ml-min-bpm <bpm>` / `--ml-max-bpm <bpm>` (validated in `[70,180]`)
- `--ml-dbn` / `--ml-no-dbn`
- `--ml-cpu-only`
- `--ml-verbose`
- `--ml-info`

Use `./build/beatit --help` for the authoritative option list.

## Library Usage

`analysis()` (single-shot):

```cpp
#include "beatit/analysis.h"
#include "beatit/coreml_preset.h"

beatit::CoreMLConfig cfg;
if (auto preset = beatit::make_coreml_preset("beatthis")) {
    preset->apply(cfg);
}

beatit::AnalysisResult result = beatit::analyze(samples, sample_rate, cfg);
```

Windowed provider flow (recommended integration pattern):

```cpp
#include "beatit/stream.h"

beatit::CoreMLConfig cfg;
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

## Tests

Run all:

```bash
ctest --test-dir build --output-on-failure
```

CPU-only (for environments where GPU/MPS is unavailable):

```bash
BEATIT_TEST_CPU_ONLY=1 ctest --test-dir build --output-on-failure
```

Main regression tracks currently covered by dedicated sparse/window tests include:

- `manucho`
- `enigma`
- `vamp`
- `bazaar`
- `intelligence`
- `neural`

## Notes

- BeatIt is a library/CLI project; it has no UI.
- Ambient or spoken-word material may legitimately return weak/no beat grids.
- Additional benchmark notes live in `DBN_BENCHMARK.md`.

## Credits

- BeatTrack — Matthew Rice — https://github.com/mhrice/BeatTrack — MIT
- Beat This! — Francesco Foscarin, Jan Schlueter, Gerhard Widmer — https://github.com/CPJKU/beat_this — MIT
- rotary-embedding-torch — Phil Wang — https://github.com/lucidrains/rotary-embedding-torch — MIT
