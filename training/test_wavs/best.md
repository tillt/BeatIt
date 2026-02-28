# best.wav

- Path: `training/best.wav`
- Tempo (BPM): `121.997` (stable, CPU sparse-window alignment test)
- Beat events: `1163`
- Downbeat events: `30`
- Projected beat events: `1163`
- Canonical decoded downbeat feature frame: `26` (`0.520s` @ 50 fps feature timeline)
- Canonical projected beat grid starts at sample frame `3711` (`0.084s @ 44.1 kHz output grid)
- The finalized projected bar marker is later than the decoded first downbeat; this file is validated by local rhythmic sections instead of a brittle first projected downbeat timestamp.
- Local rhythmic alignment checks (CPU baseline):
  - start window: `<= 20ms`
  - pre-break window: `<= 32ms`
  - post-break re-entry window: `<= 32ms`
  - end window: `<= 25ms`
- Drift probe (CPU baseline):
  - `start_median_ms=0.31746`
  - `end_median_ms=-14.263`
  - `delta_ms=-14.5805`
  - `delta_beats=0.0296464`
  - `slope_ms_per_beat=-0.0158031`
  - `odd_even_gap_ms=0.566893`
  - `early_bpm=121.997`
  - `late_bpm=121.997`
  - edge BPM delta: `0`
- Weak-break diagnostics:
  - the long middle break is not treated as a canonical rhythmic alignment window
  - generic wrapped-middle interior checks are intentionally not used for this file
- Current status:
  - downbeat phase is correct
  - beat phase is visually stable across the meaningful rhythmic sections
  - any residual drift is minor and well below the old one-beat failure mode

## Decision Path (CPU sparse baseline)

- Probe option set:
  - left candidate: `start=17.5066s bpm=121.966 conf=0.978518`
  - right candidate: `start=532.421s bpm=122.002 conf=0.980795`
  - between candidate: `start=146.235s bpm=122.031 conf=0.987724`
- Probe chooser outcome:
  - `decision=unified`
  - `selected_start=17.5066`
  - `selected_mode_err=0.000288079`
  - `score_margin=0.378999`
- Optional probe/gate branches:
  - `middle_gate_triggered=1`
  - `consistency_gate_triggered=0`
  - `interior_probe_added=1`
  - `repair=0`
- Interior diagnostics after usability gating:
  - `selected_between_start_s=37.5066`
  - `selected_middle_start_s=327.507`
  - weak middle-break windows with wrapped-sign behavior are rejected as interior candidates
- Edge refit path:
  - `phase_try_selected=0`
  - `phase_try_applied=0`
  - `global_ratio_applied=1`
  - `ratio_applied=1.00002`
  - `delta_frames=487`

Reproduce baseline:

```bash
cmake --build build --target beatit_best_window_alignment_sparse_tests -j8
cd build && BEATIT_TEST_CPU_ONLY=1 ./beatit_best_window_alignment_sparse_tests
```
