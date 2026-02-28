# bazaar.wav

- Path: `training/bazaar.wav`
- Status: current sparse-window result is accepted
- Tempo (BPM): `123.013`
- First downbeat feature frame: `10` (`0.200s` @ 50 fps feature timeline)
- First downbeat sample frame: `11994`
- Current validated behavior:
  - downbeat is correct
  - drift is musically negligible in the player
  - analyzer-derived slope/start-end metrics currently overstate the error on this file and are not used as hard truth here

## Dedicated Regression

- narrow canonical test
- BPM sanity
- first downbeat timing
- default 4/4 structure

Reproduce baseline:

```bash
cmake --build build --target beatit_bazaar_window_alignment_sparse_tests -j8
cd build && BEATIT_TEST_CPU_ONLY=1 ./beatit_bazaar_window_alignment_sparse_tests
```
