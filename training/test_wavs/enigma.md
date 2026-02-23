# enigma.wav

- Path: `training/enigma.wav`
- Tempo (BPM): `126.0` (stable, CPU sparse-window alignment test)
- Beat events: `777`
- Downbeat events: `32`
- Projected beat events: `777`
- First downbeat feature frame: `0` (`0.000s` @ 50 fps feature timeline)
- First downbeat sample frame: `986` (`0.022358s` @ 44.1 kHz output grid)
- 4/4 bar phase: first bar complete, bars repeat every 4 beats
- Drift probe (CPU baseline):
  - `start_median_ms=-0.47619`
  - `end_median_ms=-3.90023`
  - `delta_ms=-3.42404`
  - `delta_beats=0.00719048`
  - `slope_ms_per_beat=-0.00252449`
  - `odd_even_gap_ms=8.322`
  - `early_bpm=126`
  - `late_bpm=126`
  - `edge_bpm_delta=0`

## Decision Path (CPU sparse baseline)

- Probe option set:
  - left candidate: `start=10s bpm=126.032 conf=0.996403`
  - right candidate: `start=322.734s bpm=126.006 conf=0.970442`
- Probe chooser outcome:
  - `decision=unified`
  - `selected_start=322.734`
  - `selected_mode_err=0`
  - `score_margin=0.193751`
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
  - `ratio_applied=1.00013`
  - `delta_frames=2061`
