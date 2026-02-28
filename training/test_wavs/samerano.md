# samerano.wav

- Path: `training/samerano.wav`
- Status: current sparse-window result is not acceptable
- Tempo (BPM): `122.49` (current CPU sparse-window result)
- Current observed problem:
  - beat grid drifts by more than one beat over the file
  - player validation and analyzer output agree that global alignment is implausible
- Alignment inspector snapshot:
  - `fraction=0.08 median_abs_ms=91.4286`
  - `fraction=0.12 median_abs_ms=148.118`
  - `fraction=0.50 median_abs_ms=198.209`
  - `fraction=0.75 median_abs_ms=150.59`
  - `fraction=0.90 median_abs_ms=101.043`
- Current dedicated regression:
  - this is a known-failure test, not a regression from a previously good baseline
  - it should fail until global drift is brought down to musically plausible levels
