# neural.wav

- Path: `training/neural.wav`
- Status: current sparse-window result is not acceptable
- Tempo (BPM): `126.0` (current CPU sparse-window result)
- Current observed problem:
  - tempo and projected beat grid look correct
  - drift is effectively negligible
  - downbeat is one beat late
- Current emitted state:
  - first projected beat sample frame: `0`
  - first projected downbeat sample frame: `21001`
- Current dedicated regression:
  - this is a downbeat-phase regression
  - the correct first downbeat should land at the start of the file, not one beat later
