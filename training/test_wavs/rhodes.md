# rhodes.wav

- Path: `training/rhodes.wav`
- Status: current sparse-window result is accepted
- Tempo (BPM): `125.0` (current CPU sparse-window result)
- First downbeat sample frame: `3525`
- Current validated behavior:
  - tempo is correct
  - drift is effectively zero over the file
  - beat grid is slightly late by a small constant amount
  - that late bias appears stable from the first beat to the last, so this is an absolute phase bias rather than tempo drift
- Current dedicated regression:
  - narrow canonical test
  - BPM sanity
  - first downbeat timing
  - default 4/4 structure
- Note:
  - analyzer-wide nearest-peak offsets do not currently agree with the player impression on this file
  - this test intentionally avoids broader local-offset assertions until that discrepancy is understood
