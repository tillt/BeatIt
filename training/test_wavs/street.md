# street.wav

- Path: `training/street.wav`
- Status: current sparse-window result is under observation
- Tempo (BPM): `121.997` (current CPU sparse-window result)
- Current analyzer observation:
  - standalone alignment-inspector output suggests a late opening anchor that settles later in the file
  - the shared window-alignment harness does not currently reproduce that failure mode reliably
- Current dedicated regression:
  - narrow baseline test
  - BPM sanity plus default 4/4 structure
  - no hard opening-anchor assertion until the discrepancy between the dedicated analyzer and the shared harness is understood
