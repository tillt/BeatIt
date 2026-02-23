# intelligence.wav

- Path: `training/intelligence.wav`
- Tempo (BPM): `126.02` (stable, CPU sparse-window alignment test)
- Beat events: `807`
- Downbeat events: `32`
- Projected beat events: `807`
- First downbeat feature frame: `3` (`0.060s` @ 50 fps feature timeline)
- First downbeat sample frame: `6302` (`0.142902s` @ 44.1 kHz output grid)
- 4/4 bar phase: first bar complete, bars repeat every 4 beats
- Drift probe (CPU baseline):
  - `start_median_ms=14.1723`
  - `end_median_ms=22.1769`
  - `delta_ms=8.00454`
  - `delta_beats=0.0168121`
  - `slope_ms_per_beat=0.0233139`
  - `odd_even_gap_ms=0.521542`
  - `early_bpm=126.018`
  - `late_bpm=126.018`
  - `edge_bpm_delta=0`

## Decision Path (CPU sparse baseline)

- Probe option set:
  - left candidate: `start=17.5066s bpm=125.958 conf=0.975212`
  - right candidate: `start=329.815s bpm=126.021 conf=0.98085`
- Probe chooser outcome:
  - `decision=unified`
  - `selected_start=17.5066`
  - `selected_mode_err=0`
  - `score_margin=0.372428`
- Optional probe/gate branches:
  - `middle_gate_triggered=0`
  - `consistency_gate_triggered=0`
  - `interior_probe_added=0`
  - `repair=0`
- Edge refit path:
  - `second_pass=1`
  - `phase_try_selected=0`
  - `phase_try_applied=0`
  - `global_ratio_applied=0.99985`
  - `ratio_applied=0.999938`
  - `delta_frames=-1056`
