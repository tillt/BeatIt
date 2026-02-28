# turbo.wav

- Path: `training/turbo.wav`
- Status: current sparse-window result is accepted
- Tempo (BPM): `127.002` (current CPU sparse-window result)
- Current validated sections:
  - opening beat anchoring is correct
  - break re-entry shows correct beat and downbeat alignment
  - late section remains aligned
- Current dedicated regression:
  - narrow canonical test
  - BPM sanity plus 4/4 structure
  - broad analyzer-wide nearest-peak medians currently do not match the player view on this file and are not used as hard truth here
