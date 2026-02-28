# what.wav

- Path: `training/what.wav`
- Status: current sparse-window result is not acceptable
- Tempo (BPM): `122.056` (current CPU sparse-window result)
- Current observed problem:
  - downbeat phase is one beat early
  - tempo/drift error accumulates over the file, but stays below a full beat
- Current emitted state:
  - first projected downbeat feature frame: `8`
  - expected first projected downbeat feature frame: about `33`
- Alignment inspector snapshot:
  - `fraction=0.12 median_abs_ms=12.0181`
  - `fraction=0.50 median_abs_ms=41.8594`
  - `fraction=0.75 median_abs_ms=81.5193`
  - `fraction=0.90 median_abs_ms=100.431`
- Current dedicated regression:
  - this is a known bad result, not a canonical baseline
  - the test should fail until first downbeat moves one beat later and late-file drift drops to musically acceptable levels
