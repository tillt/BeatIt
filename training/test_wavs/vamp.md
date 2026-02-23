# vamp.wav

- Path: `training/vamp.wav`
- Tempo (BPM): `126.998` (stable, CPU sparse-window alignment test)
- Beat events: `649`
- Downbeat events: `31`
- Projected beat events: `649`
- First downbeat feature frame: `33` (`0.660s` @ 50 fps feature timeline)
- First downbeat sample frame: `10573` (`0.239751s` @ 44.1 kHz output grid)
- 4/4 bar phase: first bar complete, bars repeat every 4 beats
- Drift probe (CPU baseline):
  - `start_median_ms=1.95011`
  - `end_median_ms=-2.01814`
  - `delta_ms=-3.96825`
  - `delta_beats=0.00839933`
  - `slope_ms_per_beat=-0.00401013`
  - `odd_even_gap_ms=0.47619`
  - `early_bpm=126.998`
  - `late_bpm=126.998`
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
