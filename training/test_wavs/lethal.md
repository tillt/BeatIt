# lethal.wav

- Path: `training/lethal.wav`
- Tempo (BPM): `122.009` (stable, CPU sparse-window alignment test)
- Beat events: `781`
- Downbeat events: `30`
- Projected beat events: `781`
- Expected first downbeat feature frame: `32`
- Expected first downbeat sample frame: `14390` (`0.326304s` @ 44.1 kHz output grid)
- Current library output is one beat too early on the downbeat phase while keeping a stable 4/4 repeat pattern
- Drift probe (CPU baseline):
  - `start_median_ms=-13.22`
  - `end_median_ms=11.7914`
  - `delta_ms=25.0113`
  - `delta_beats=0.05086`
  - `slope_ms_per_beat=0.0408629`
  - `odd_even_gap_ms=0.0907029`
  - `early_bpm=122.009`
  - `late_bpm=122.009`
  - `edge_bpm_delta=0`

## Decision Path (CPU sparse baseline)

- Probe option set:
  - left candidate: `start=17.5066s bpm=121.969 conf=0.993633`
  - right candidate: `start=344.767s bpm=121.992 conf=0.979451`
  - between candidate: `start=99.3216s bpm=121.978 conf=0.98226`
- Probe chooser outcome:
  - `decision=unified`
  - `selected_start=99.3216`
  - `selected_mode_err=0`
  - `score_margin=0.263599`
- Optional probe/gate branches:
  - `middle_gate_triggered=0`
  - `consistency_gate_triggered=0`
  - `interior_probe_added=1`
  - `repair=0`
- Edge refit path:
  - `phase_try_selected=-1`
  - `phase_try_applied=1`
  - `global_ratio_applied=1.00009`
  - `ratio_applied=1.00007`
  - `delta_frames=1144`

Reproduce current failing regression:

```bash
cmake --build build --target beatit_lethal_window_alignment_sparse_tests -j8
cd build && BEATIT_TEST_CPU_ONLY=1 ./beatit_lethal_window_alignment_sparse_tests
```
