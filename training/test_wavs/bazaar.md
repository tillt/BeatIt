# bazaar.wav

- Path: `training/bazaar.wav`
- Tempo (BPM): `123.013` (stable, CPU sparse-window alignment test)
- Beat events: `770`
- Downbeat events: `32`
- Projected beat events: `770`
- First downbeat feature frame: `10` (`0.200s` @ 50 fps feature timeline)
- First downbeat sample frame: `0` (`0.000000s` @ 44.1 kHz output grid)
- Drift probe (CPU baseline):
  - `start_median_ms=-13.7868`
  - `end_median_ms=18.2313`
  - `delta_ms=32.0181`
  - `delta_beats=0.0656439`
  - `slope_ms_per_beat=0.0327633`
  - `early_bpm=123.013`
  - `late_bpm=123.013`
  - `edge_bpm_delta=0`

## Decision Path (CPU sparse baseline)

- Probe option set:
  - left candidate: `start=32.5198s bpm=122.994 conf=0.987434`
  - right candidate: `start=313.635s bpm=123.058 conf=0.98597`
- Probe chooser outcome:
  - `decision=unified`
  - `selected_start=313.635`
  - `selected_mode_err=0`
  - `score_margin=0.233681`
- Optional probe/gate branches:
  - `middle_gate_triggered=0`
  - `consistency_gate_triggered=0`
  - `interior_probe_added=0`
  - `repair=0`
- Edge refit path:
  - `second_pass=1`
  - `phase_try_selected=-1`
  - `phase_try_applied=1`
  - `global_ratio_applied=1.00016`
  - `ratio_applied=1.0004`
  - `delta_frames=6643`

## Notes

- The opening anchor is slightly early in the first local section, but the user validated the file as musically acceptable in the player.
- The canonical regression for `bazaar` is therefore tempo stability / drift-free projection, not a very tight intro-local offset check.

Reproduce baseline:

```bash
cmake --build build --target beatit_bazaar_window_alignment_sparse_tests -j8
cd build && BEATIT_TEST_CPU_ONLY=1 ./beatit_bazaar_window_alignment_sparse_tests
```
