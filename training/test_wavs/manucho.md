# manucho.wav

- Path: `training/manucho.wav`
- Tempo (BPM): `109.998` (stable, CPU sparse-window alignment test)
- Beat events: `792`
- Downbeat events: `28`
- Projected beat events: `792`
- First downbeat feature frame: `18` (`0.360s` @ 50 fps feature timeline)
- First downbeat sample frame: `2309` (`0.052358s` @ 44.1 kHz output grid)
- 4/4 bar phase: first bar complete, bars repeat every 4 beats
- Drift probe (CPU baseline):
  - `start_median_ms=0.408163`
  - `end_median_ms=-7.91383`
  - `delta_ms=-8.322`
  - `delta_beats=0.0152567`
  - `slope_ms_per_beat=-0.0141231`
  - `early_bpm=109.998`
  - `late_bpm=109.998`
  - `edge_bpm_delta=0`

## Decision Path (CPU sparse baseline)

- Probe option set:
  - left candidate: `start=17.5066s bpm=110.024 conf=0.967189`
  - right candidate: `start=384.498s bpm=109.962 conf=0.972447`
- Probe chooser outcome:
  - `decision=unified`
  - `selected_start=17.5066`
  - `selected_mode_err=0`
  - `score_margin=0.283309`
- Optional probe/gate branches:
  - `middle_gate_triggered=0`
  - `consistency_gate_triggered=0`
  - `interior_probe_added=0`
  - `repair=0`
- Edge refit path:
  - `second_pass=1`
  - `phase_try_selected=0`
  - `phase_try_applied=0`
  - `global_ratio_applied=1`
  - `ratio_applied=0.999963`
  - `delta_frames=-706`
