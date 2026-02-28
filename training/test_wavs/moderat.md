# moderat.wav

- Path: `training/moderat.wav`
- Tempo (BPM): `124.001` (stable, CPU sparse-window alignment test)
- Beat events: `742`
- Downbeat events: `32`
- Projected beat events: `742`
- First downbeat feature frame: `0` (`0.000s` @ 50 fps feature timeline)
- First downbeat sample frame: `3750` (`0.085034s` @ 44.1 kHz output grid)
- 4/4 bar phase: first bar complete, bars repeat every 4 beats
- Drift probe (CPU baseline):
  - `start_median_ms=1.72336`
  - `end_median_ms=0.340136`
  - `delta_ms=-1.38322`
  - `delta_beats=0.00285868`
  - `slope_ms_per_beat=0.00393408`
  - `odd_even_gap_ms=4.12698`
  - `early_bpm=123.998`
  - `late_bpm=123.998`
  - `edge_bpm_delta=0`

## Decision Path (CPU sparse baseline)

- Probe option set:
  - left candidate: `start=17.5066s bpm=124.064 conf=0.967899`
  - right candidate: `start=319s bpm=124.001 conf=0.999069`
  - between candidate: `start=92.8801s bpm=124.023 conf=0.975081`
- Probe chooser outcome:
  - `decision=unified`
  - `selected_start=17.5066`
  - `selected_mode_err=0.000328134`
  - `score_margin=0.1486`
- Optional probe/gate branches:
  - `middle_gate_triggered=1`
  - `consistency_gate_triggered=0`
  - `interior_probe_added=1`
  - `repair=0`
- Edge refit path:
  - `phase_try_selected=0`
  - `phase_try_applied=0`
  - `global_ratio_applied=1`
  - `ratio_applied=1.00019`
  - `delta_frames=3035`

Reproduce baseline:

```bash
cmake --build build --target beatit_moderat_window_alignment_sparse_tests -j8
cd build && BEATIT_TEST_CPU_ONLY=1 ./beatit_moderat_window_alignment_sparse_tests
```
