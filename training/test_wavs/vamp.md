# vamp.wav

- Path: `training/vamp.wav`
- Tempo (BPM): `126.951` (current CPU sparse-window result)
- Beat events: `649`
- Downbeat events: `31`
- Projected beat events: `649`
- First downbeat feature frame: `33` (`0.660s` @ 50 fps feature timeline)
- First downbeat sample frame: `10573` (`0.239751s` @ 44.1 kHz output grid)
- 4/4 bar phase: first bar complete, bars repeat every 4 beats
- Current regression:
  - early beat anchoring is visibly late/early relative to the local peaks depending on sign convention, with markers landing about `60-75ms` off in the early rhythmic sections
  - tempo remains close, but not exact enough; the slight error accumulates across playtime
- Drift probe (current CPU sparse result):
  - `start_median_ms` now regresses in the early windows much more strongly than the older baseline suggested
  - local 32-beat windows:
    - `0.02`: `median_abs_ms=75.0794`
    - `0.05`: `median_abs_ms=71.61`
    - `0.08`: `median_abs_ms=68.7075`
    - `0.12`: `median_abs_ms=65.6236`
    - `0.18`: `median_abs_ms=62.7211`
    - `0.90`: `median_abs_ms=30.6576`
  - these values are the reason `vamp` should now be treated as a full anchoring regression, not just a BPM tolerance miss
  - `odd_even_gap_ms=0.47619`
  - `early_bpm=126.951`
  - `late_bpm=126.951`
  - `edge_bpm_delta=0`

## Decision Path (CPU sparse baseline)

- Probe option set:
  - left candidate: `start=25.0132s bpm=126.954 conf=0.973502`
  - right candidate: `start=237.067s bpm=126.951 conf=0.981451`
  - between candidate: `start=78.0267s bpm=126.977 conf=0.967217`
- Probe chooser outcome:
  - `decision=unified`
  - `selected_start=237.067`
  - `selected_mode_err=2.40386e-05`
  - `score_margin=0.916708`
- Optional probe/gate branches:
  - `middle_gate_triggered=1`
  - `consistency_gate_triggered=0`
  - `interior_probe_added=1`
  - `repair=0`
- Edge refit path:
  - `second_pass=1`
  - `phase_try_selected=0`
  - `phase_try_applied=0`
  - `global_ratio_applied=0.999706`
  - `ratio_applied=1.0001`
  - `delta_frames=1380`

## Regression Notes

- The current failure is no longer modeled as an exact BPM mismatch.
- The dedicated test should fail on the true problem:
  - poor local beat anchoring in the early rhythmic sections
  - still-not-exact tempo over the whole file
