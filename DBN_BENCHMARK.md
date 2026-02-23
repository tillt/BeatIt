# DBN Benchmark

BeatThis (Torch backend) with DBN enabled. Each run prints two DBN config lines: pass 1 (full BPM range) and pass 2 (narrowed around pass‑1 estimate).

For each file we also capture DBN timing lines (usually two entries) and the final Estimated BPM.

## sparse probe benchmark (current)

Target: BeatThis CoreML sparse mode (`--ml-sparse-probe`) with precompiled model.

Run command:
```
python3 scripts/dbn_benchmark.py \
  --beatit build/beatit \
  --backend coreml \
  --model coreml_out_latest/BeatThis_small0.mlmodelc \
  --pass-args --ml-sparse-probe --ml-cpu-only \
  > logs/dbn_sparse_probe_latest.csv
```

Raw CSV: `logs/dbn_sparse_probe_latest.csv`

### Result snapshot

```
training/manucho.wav bpm=109.989 bpm_err=0.011 phase_err=0.031
training/moderat.wav bpm=124.019 bpm_err=0.019 phase_err=0.074
training/samerano.wav bpm=81.345 bpm_err=40.655 phase_err=0.188
training/purelove.wav bpm=70.159 bpm_err=33.841 phase_err=1.613
training/acht.wav bpm=120.771 bpm_err=0.771 phase_err=0.362
training/best.wav bpm=81.345 bpm_err=43.655 phase_err=0.049
training/eureka.wav bpm=120.031 bpm_err=0.031 phase_err=0.000
```

### Aggregate stats (7/7 runs succeeded)

```
bpm_err: median=0.771 mean=16.998 p90=43.655 max=43.655
downbeat_err_s: median=0.188 mean=0.538 p90=1.836 max=1.836
downbeat_phase_err_s: median=0.074 mean=0.331 p90=1.613 max=1.613
score: median=18.379 mean=22.377 p90=47.280 max=47.280
```

### Locked thresholds (current sparse mode)

These are now the benchmark guardrails for this exact command/model:

```
run_status == ok for all benchmark files
bpm_err <= 44.0
downbeat_phase_err_s <= 1.65
score <= 48.0
```

Note: sparse mode removes drift on most files in listening tests, but this benchmark
still shows clear tempo failures on `samerano`, `purelove`, and `best` (half/double-time
mode-selection errors). These thresholds are intentionally conservative until those
mode failures are fixed.

## baseline

Flags: `--ml-dbn`

### training/manucho.wav
Estimated BPM: 110.067
- 18:DBN config: all_candidates=false raw_candidates=21600 used_candidates=21594 pruned=20570 floor=0.05 tol=0.05 max_cand=1024 bpm=[55,215] step=1 tempos=161 bpb=4
- 19:DBN config: all_candidates=false raw_candidates=21600 used_candidates=21594 pruned=20570 floor=0.05 tol=0.05 max_cand=1024 bpm=[88.0532,264.16] step=1 tempos=177 bpb=4

### training/moderat.wav
Estimated BPM: 124.055
- 16:DBN config: all_candidates=false raw_candidates=17950 used_candidates=17944 pruned=16920 floor=0.05 tol=0.05 max_cand=1024 bpm=[55,215] step=1 tempos=161 bpb=4
- 17:DBN config: all_candidates=false raw_candidates=17950 used_candidates=17944 pruned=16920 floor=0.05 tol=0.05 max_cand=1024 bpm=[99.244,297.732] step=1 tempos=199 bpb=4

### training/samerano.wav
Estimated BPM: 122.059
- 17:DBN config: all_candidates=false raw_candidates=19757 used_candidates=19751 pruned=18727 floor=0.05 tol=0.05 max_cand=1024 bpm=[55,215] step=1 tempos=161 bpb=4
- 18:DBN config: all_candidates=false raw_candidates=19757 used_candidates=19751 pruned=18727 floor=0.05 tol=0.05 max_cand=1024 bpm=[97.6471,292.941] step=1 tempos=196 bpb=4

### training/purelove.wav
Estimated BPM: 104.737
- 12:DBN config: all_candidates=false raw_candidates=13207 used_candidates=13201 pruned=12177 floor=0.05 tol=0.05 max_cand=1024 bpm=[55,215] step=1 tempos=161 bpb=4
- 13:DBN config: all_candidates=false raw_candidates=13207 used_candidates=13201 pruned=12177 floor=0.05 tol=0.05 max_cand=1024 bpm=[83.7894,251.368] step=1 tempos=168 bpb=4

## strict

Flags: `--ml-dbn --ml-dbn-floor 0.10 --ml-dbn-max-cand 256 --ml-dbn-step 2.0 --ml-dbn-tol 0.02`

### training/manucho.wav
Estimated BPM: 107.916
- 18:DBN config: all_candidates=false raw_candidates=21600 used_candidates=21594 pruned=21338 floor=0.1 tol=0.02 max_cand=256 bpm=[55,215] step=2 tempos=81 bpb=4
- 19:DBN config: all_candidates=false raw_candidates=21600 used_candidates=21594 pruned=21338 floor=0.1 tol=0.02 max_cand=256 bpm=[86.3325,258.998] step=2 tempos=87 bpb=4

### training/moderat.wav
Estimated BPM: 59.5054
- 16:DBN config: all_candidates=false raw_candidates=17950 used_candidates=17944 pruned=17688 floor=0.1 tol=0.02 max_cand=256 bpm=[55,215] step=2 tempos=81 bpb=4
- 17:DBN config: all_candidates=false raw_candidates=17950 used_candidates=17944 pruned=17688 floor=0.1 tol=0.02 max_cand=256 bpm=[47.6044,142.813] step=2 tempos=48 bpb=4

### training/samerano.wav
Estimated BPM: 121.923
- 17:DBN config: all_candidates=false raw_candidates=19757 used_candidates=19751 pruned=19495 floor=0.1 tol=0.02 max_cand=256 bpm=[55,215] step=2 tempos=81 bpb=4
- 18:DBN config: all_candidates=false raw_candidates=19757 used_candidates=19751 pruned=19495 floor=0.1 tol=0.02 max_cand=256 bpm=[97.5385,292.615] step=2 tempos=98 bpb=4

### training/purelove.wav
Estimated BPM: 102.85
- 12:DBN config: all_candidates=false raw_candidates=13207 used_candidates=13201 pruned=12945 floor=0.1 tol=0.02 max_cand=256 bpm=[55,215] step=2 tempos=81 bpb=4
- 13:DBN config: all_candidates=false raw_candidates=13207 used_candidates=13201 pruned=12945 floor=0.1 tol=0.02 max_cand=256 bpm=[82.2801,246.84] step=2 tempos=83 bpb=4

## loose

Flags: `--ml-dbn --ml-dbn-floor 0.01 --ml-dbn-max-cand 4096 --ml-dbn-tol 0.20 --ml-dbn-all-cand`

### training/manucho.wav
Estimated BPM: 110.091
- 18:DBN config: all_candidates=true raw_candidates=21600 used_candidates=21600 pruned=0 floor=0.01 tol=0.2 max_cand=4096 bpm=[55,215] step=1 tempos=161 bpb=4
- 19:DBN config: all_candidates=true raw_candidates=21600 used_candidates=21600 pruned=0 floor=0.01 tol=0.2 max_cand=4096 bpm=[88.0724,264.217] step=1 tempos=177 bpb=4

### training/moderat.wav
Estimated BPM: 123.645
- 16:DBN config: all_candidates=true raw_candidates=17950 used_candidates=17950 pruned=0 floor=0.01 tol=0.2 max_cand=4096 bpm=[55,215] step=1 tempos=161 bpb=4
- 17:DBN config: all_candidates=true raw_candidates=17950 used_candidates=17950 pruned=0 floor=0.01 tol=0.2 max_cand=4096 bpm=[98.9181,296.754] step=1 tempos=198 bpb=4

### training/samerano.wav
Estimated BPM: 121.217
- 17:DBN config: all_candidates=true raw_candidates=19757 used_candidates=19757 pruned=0 floor=0.01 tol=0.2 max_cand=4096 bpm=[55,215] step=1 tempos=161 bpb=4
- 18:DBN config: all_candidates=true raw_candidates=19757 used_candidates=19757 pruned=0 floor=0.01 tol=0.2 max_cand=4096 bpm=[96.975,290.925] step=1 tempos=194 bpb=4

### training/purelove.wav
Estimated BPM: 104.479
- 12:DBN config: all_candidates=true raw_candidates=13207 used_candidates=13207 pruned=0 floor=0.01 tol=0.2 max_cand=4096 bpm=[55,215] step=1 tempos=161 bpb=4
- 13:DBN config: all_candidates=true raw_candidates=13207 used_candidates=13207 pruned=0 floor=0.01 tol=0.2 max_cand=4096 bpm=[83.5858,250.757] step=1 tempos=168 bpb=4

## downbeat window check (baseline)

## CoreML conversion recipe (BeatThis)

This is the working pipeline we used once tooling was updated (no einsum failures).

### Environment
- Python 3.11 venv (avoid 3.14)
- macOS 15.7.x

### Install deps (CoreML convert)
```
python3.11 -m venv .venv_coreml_latest
source .venv_coreml_latest/bin/activate
pip install -U pip
pip install -U torch coremltools numpy einops soxr
```

### Convert
```
PYTHONPATH=third_party/beat_this \
python scripts/convert_beatthis_coreml.py \
  --models final0 \
  --out coreml_out_latest \
  --precision float32 \
  --deployment iOS18 \
  --convert-to mlprogram
```

### Compile to .mlmodelc
```
xcrun coremlcompiler compile coreml_out_latest/BeatThis_final0.mlpackage coreml_out_latest
```

### Use in BeatIt (CoreML backend)
```
./build/beatit --input training/manucho.wav \
  --ml-backend coreml --ml-preset beatthis \
  --model coreml_out_latest/BeatThis_final0.mlmodelc
```

Canon windows are measured in an audio editor as onset/peak; the detector is considered correct if any downbeat falls within the window. Errors below are distance to the nearest window edge, using the closest downbeat from the full list (not just the first).

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.067,0.067,2.200,2.150,0.031,0.05,21.567
training/moderat.wav,124.055,0.055,31.020,30.932,0.063,0.088,309.375
training/samerano.wav,122.059,0.059,269.680,269.429,0.188,0.251,2694.349
training/purelove.wav,104.737,0.737,32.300,31.700,0.331,0.6,317.737
```

## downbeat window sweep (parameter tuning)

Goal: reduce the “late start” behavior (few downbeats after long breaks) while staying close to canonical onset/peak windows.

### baseline (default DBN)

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.067,0.067,2.200,2.150,0.031,0.05,21.567
training/moderat.wav,124.055,0.055,31.020,30.932,0.063,0.088,309.375
training/samerano.wav,122.059,0.059,269.680,269.429,0.188,0.251,2694.349
training/purelove.wav,104.737,0.737,32.300,31.700,0.331,0.6,317.737
training/acht.wav,120.110,0.110,328.220,287.434,40.742,40.786,2874.450
```

### baseline (with current defaults)

Defaults were updated to match the “loose + lower tempo penalty + higher reward”
configuration in README. Re‑running with defaults yields the same numbers as the
previous baseline on these files (no change).

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.067,0.067,2.200,2.150,0.031,0.05,21.567
training/moderat.wav,124.055,0.055,31.020,30.932,0.063,0.088,309.375
training/samerano.wav,122.059,0.059,269.680,269.429,0.188,0.251,2694.349
training/purelove.wav,104.737,0.737,32.300,31.700,0.331,0.6,317.737
training/acht.wav,120.110,0.110,328.220,287.434,40.742,40.786,2874.450
```

### calmdad (BeatThis 2-class + epsilon clamp) defaults

Defaults updated to mirror BeatThis/madmom DBN construction (2‑class emission with
epsilon clamp, beats_per_bar [3,4], min/max bpm 55–215, transition_lambda=100).

Instrumentation notes (BEATIT_VERBOSE=1):
- Downbeat activation is **0** for frames 0–5, then jumps to ~0.5 at frame **6**
  (0.12s @ 50fps) for **all** canonical files. Example (manucho):
  `downbeat head: 0..5 -> 0, 6 -> ~0.50, 7.. -> ~0.50`.
- Beat activation shows the same behavior (`beat head: 0..5 -> 0, 6 -> ~0.50`),
  so both beat and downbeat logits appear time‑aligned to 0.12s.
- The DBN therefore sees its **first downbeat candidates starting at 0.12s**.
  Phase selection uses these candidates, so the first projected downbeat
  defaults to 0.12s and cannot land earlier.
- This explains the consistent **+0.06..+0.08s** downbeat offset for manucho
  and moderat, and the **-0.08s** offset for samerano: the model’s downbeat
  activation appears delayed or does not expose the onset in its logits.

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.073,0.073,0.120,0.070,0.031,0.05,0.773
training/moderat.wav,124.178,0.178,0.120,0.032,0.063,0.088,0.498
training/samerano.wav,122.098,0.098,0.180,0.008,0.188,0.251,0.178
training/purelove.wav,105.095,1.095,0.300,0.031,0.331,0.6,1.405
training/acht.wav,119.597,0.403,42.220,1.434,40.742,40.786,14.743
```

### Method note (Davies/Plumbley context‑dependent beat tracking)

Goal: clarify how the paper’s approach relates to our instrumentation and why some
phase/tempo choices behave the way they do.

Key steps in the paper:
- **Mid‑level representation**: a continuous onset detection function (ODF), not
  discrete onsets. This is the same conceptual input as our beat/downbeat activation
  curves (both are continuous “novelty” signals).
- **Beat period (tempo) induction**: compute an **autocorrelation** of the ODF, then
  score a **comb filterbank** over candidate beat periods. In the *general state*,
  the combs are weighted by a Rayleigh‑like prior (favoring a typical tempo).
- **Beat alignment (phase) induction**: with beat period fixed, score **shifted combs**
  against the ODF to select the phase. For non‑causal analysis, the alignment is the
  offset from the window start to the first beat in the window.
- **Context‑dependent state**: once a stable tempo is found, re‑run beat period and
  alignment with **Gaussian weighting centered on the previously found tempo** and a
  phase prior around the predicted beat time (to avoid half/double tempo and on/off‑beat flips).
- **Two‑state model**: switch between “general” and “context‑dependent” states when a
  tempo change is detected, ensuring continuity while still allowing tempo shifts.

What we’re doing that matches this:
- The **comb/autocorr tempo anchor** instrumentation is a direct analogue of their
  beat‑period induction step (general state). It helps explain when a window prefers
  a different tempo hypothesis than the activation‑peak histogram.
- The **phase selection** logic (picking a downbeat/beat peak within a short window)
  mirrors their beat‑alignment induction (alignment combs), but we currently rely on
  **model logits** rather than an explicit comb‑alignment pass.

What we are *not* doing yet:
- We are **not** running an explicit two‑state switch with a Gaussian tempo prior.
  Our “tempo anchor” selection is single‑pass; if we want the paper’s behavior, we
  should stabilize tempo with a prior around the last known tempo (not just pick the
  strongest current window).
- Their method is **beat‑level**, not downbeat‑level; it assumes a beat ODF. Our
  downbeat selection inherits any latency/offset in the model’s downbeat logits, which
  explains the persistent ~0.10–0.12s anchor when early downbeat logits are flat.

Why this matters for our results:
- If the **first strong downbeat logits start at frame ~6 (0.12s)**, any phase
  selection that only sees those logits cannot anchor earlier, regardless of DBN.
- A **context‑dependent tempo prior** would reduce drift by preferring the last
  stable tempo hypothesis across windows (and by resisting short‑window outliers).

### Canonical run (dbn trace flag)

Run:
```
python3 scripts/dbn_benchmark.py --pass-args --ml-dbn-trace
```

Result (same CSV fields as above):
```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,0.104,0.054,0.073,0.000,0.104,0.054,198,0.104,0.054,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.587
training/moderat.wav,124.001,0.001,0.104,0.016,0.041,0.000,0.104,0.016,186,0.104,0.016,742,0.120,0.032,0.120,0.032,nan,0.063,0.088,0.161
training/samerano.wav,121.989,0.011,0.104,0.084,0.084,0.000,0.104,0.084,201,0.104,0.084,804,0.120,0.068,0.120,0.068,nan,0.188,0.251,0.851
training/purelove.wav,105.040,1.040,0.104,0.227,0.227,0.000,0.104,0.227,155,0.104,0.227,463,0.300,0.031,0.120,0.211,nan,0.331,0.6,3.310
training/acht.wav,120.016,0.016,40.099,0.643,0.643,0.000,0.104,40.638,267,0.104,40.638,1065,0.120,40.622,0.120,40.622,nan,40.742,40.786,6.449
training/best.wav,122.006,2.994,0.104,0.000,0.055,0.000,0.104,0.000,291,0.104,0.000,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,2.994
```

Notes:
- Adding `--ml-dbn-trace` does not change the numeric results; it only enables verbose trace logging inside the DBN pipeline.

### Canonical run (best window selection)

Run:
```
python3 scripts/dbn_benchmark.py --pass-args --ml-dbn-trace --ml-dbn-window-best
```

Result:
```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,0.104,0.054,0.073,0.000,0.104,0.054,198,0.104,0.054,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.587
training/moderat.wav,124.001,0.001,0.104,0.016,0.041,0.000,0.104,0.016,186,0.104,0.016,742,0.120,0.032,0.120,0.032,nan,0.063,0.088,0.161
training/samerano.wav,121.989,0.011,0.104,0.084,0.084,0.000,0.104,0.084,201,0.104,0.084,804,0.120,0.068,0.120,0.068,nan,0.188,0.251,0.851
training/purelove.wav,105.040,1.040,0.104,0.227,0.227,0.000,0.104,0.227,155,0.104,0.227,463,0.300,0.031,0.120,0.211,nan,0.331,0.6,3.310
training/acht.wav,120.016,0.016,40.099,0.643,0.643,0.000,0.104,40.638,267,0.104,40.638,1065,0.120,40.622,0.120,40.622,nan,40.742,40.786,6.449
training/best.wav,122.006,2.994,0.104,0.000,0.055,0.000,0.104,0.000,291,0.104,0.000,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,2.994
```

Notes:
- Best-window selection did not change the numerical results on these canonicals.

### Phase selection with strict downbeat/beat peaks (best window)

Goal: avoid the “everything is a candidate” phase anchor by only using strict local peaks for
phase selection (downbeat peaks first, beat peaks as fallback), while keeping the rest of
`calmdad` unchanged.

Run:
```
./build/beatit --input training/manucho.wav --ml-backend torch --torch-model models/beatthis.pt \
  --ml-preset beatthis --ml-dbn --ml-dbn-trace --ml-dbn-window-best --ml-verbose
```

Observation:
- Phase selection still locks to **phase 0** and the projected grid still starts at ~0.104–0.12s.
- Trace now shows `phase peaks for selection: picked`, but the best phase remains unchanged.
- Canonical CSV is unchanged (see “Canonical run (best window selection)” above).

### Phase selection (beat-only, strict peaks)

Change: phase selection ignores downbeat activations and only uses **strict beat peaks**
(curr >= prev + eps and curr >= next + eps, eps = 1% of max beat activation).

Run:
```
python3 scripts/dbn_benchmark.py --pass-args --ml-dbn-trace --ml-dbn-window-best
```

Result:
```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,0.104,0.054,0.073,0.000,0.104,0.054,198,0.104,0.054,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.587
training/moderat.wav,124.001,0.001,0.104,0.016,0.041,0.000,0.104,0.016,186,0.104,0.016,742,0.120,0.032,0.120,0.032,nan,0.063,0.088,0.161
training/samerano.wav,121.989,0.011,0.104,0.084,0.084,0.000,0.104,0.084,201,0.104,0.084,804,0.120,0.068,0.120,0.068,nan,0.188,0.251,0.851
training/purelove.wav,105.040,1.040,0.104,0.227,0.227,0.000,0.104,0.227,155,0.104,0.227,463,0.300,0.031,0.120,0.211,nan,0.331,0.6,3.310
training/acht.wav,120.016,0.016,40.099,0.643,0.643,0.000,0.104,40.638,267,0.104,40.638,1065,0.120,40.622,0.120,40.622,nan,40.742,40.786,6.449
training/best.wav,122.006,2.994,0.104,0.000,0.055,0.000,0.104,0.000,291,0.104,0.000,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,2.994
```

Notes:
- Still locks to **phase 0** with the same ~0.104s anchor.
- Trace confirms “phase peaks for selection (beat-only, strict): picked”.

### Phase selection (peak energy average)

Change: instead of picking the **earliest** peak, score each candidate phase by the
**average activation value** over peaks in the phase window (beat‑only, strict peaks).

Run:
```
python3 scripts/dbn_benchmark.py --pass-args --ml-dbn-trace --ml-dbn-window-best
```

Result:
```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,0.104,0.054,0.073,0.000,0.104,0.054,198,0.104,0.054,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.587
training/moderat.wav,124.001,0.001,0.104,0.016,0.041,0.000,0.104,0.016,186,0.104,0.016,742,0.120,0.032,0.120,0.032,nan,0.063,0.088,0.161
training/samerano.wav,121.989,0.011,0.104,0.084,0.084,0.000,0.104,0.084,201,0.104,0.084,804,0.120,0.068,0.120,0.068,nan,0.188,0.251,0.851
training/purelove.wav,105.040,1.040,0.104,0.227,0.227,0.000,0.104,0.227,155,0.104,0.227,463,0.300,0.031,0.120,0.211,nan,0.331,0.6,3.310
training/acht.wav,120.016,0.016,40.099,0.643,0.643,0.000,0.104,40.638,267,0.104,40.638,1065,0.120,40.622,0.120,40.622,nan,40.742,40.786,6.449
training/best.wav,122.006,2.994,0.104,0.000,0.055,0.000,0.104,0.000,291,0.104,0.000,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,2.994
```

Notes:
- Still locked to **phase 0** and the same ~0.104s anchor.

### Mapping / latency audit (manucho)

Trace snippet:
```
DBN grid project: start_frame=6 start_time=0.12 bpm=109.954 step_frames=27.2842 total_frames=21600 latency_samples=706 hop_size=441 hop_scale=2 start_sample_frame=4586 start_time_adj=0.103991
```

Interpretation:
- The grid anchor is **frame 6** at 100 fps (≈0.12 s).
- With hop_size=441 and hop_scale=2 (44.1 kHz input vs 22.05 kHz model), this maps to
  sample_frame≈4586. After subtracting latency (706 samples), the adjusted time is **~0.104 s**.
- The ~0.104 s bias comes from the **chosen start_frame (6)**, not the latency itself.

Round: regression interval (guarded by 2% vs median)

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s
training/manucho.wav,110.013,0.013,0.120,0.080
training/moderat.wav,123.966,-0.034,0.120,0.060
training/samerano.wav,121.895,-0.105,0.120,-0.080
training/purelove.wav,104.920,0.920,0.120,-0.200
```

Notes:
- BPM improved for **manucho/moderat** while keeping samerano stable.
- Downbeat remains locked at **0.12s** due to logits (see instrumentation notes above).

#### Activation peak counts (before tuning)

Collected with `BEATIT_DEBUG_BPM=1` (activation tempo estimator):

```
file,activation_peaks,threshold,activation_bpm_est
training/manucho.wav,768,0.7,111.1
training/moderat.wav,731,0.7,125.0
```

Interval stats (BEATIT_DEBUG_BPM=1):

```
file,interval_mean,interval_median,interval_std,bpm_mean,bpm_median
training/manucho.wav,27.824,27,5.149,107.821,111.111
training/moderat.wav,24.426,24,5.832,122.82,125.0
```

#### Autocorr tempo anchor (decimated activation)

BEATIT_DEBUG_BPM=1 after switching the global tempo estimator to autocorrelation:

```
file,autocorr_frames,stride,lag,bpm,score
training/manucho.wav,3086,7,3.9985,107.183,13.696
training/moderat.wav,3590,5,4.9895,120.253,12.538
```

Note: BPM estimates moved farther from canon for both manucho/moderat, so the
autocorr‑only anchor is **worse** than the activation‑peak histogram for these files.

### Early downbeat phase selection (2s window, peak ratio 0.6)

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.092,0.092,0.000,-0.040,0.031,0.05,0.040
training/moderat.wav,123.711,-0.289,0.000,-0.060,0.063,0.088,0.060
training/samerano.wav,121.212,-0.788,0.180,-0.020,0.188,0.251,0.020
training/purelove.wav,103.448,-0.552,0.280,-0.040,0.331,0.6,0.040
```

### Early downbeat phase selection (no prepend in canonical test)

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.092,0.092,0.120,0.080,0.031,0.05,0.080
training/moderat.wav,125.000,1.000,1.080,1.020,0.063,0.088,1.020
training/samerano.wav,122.449,0.449,0.120,-0.080,0.188,0.251,0.080
training/purelove.wav,105.263,1.263,0.300,-0.020,0.331,0.6,0.020
```

### Early downbeat phase selection + max delay penalty (0.3s)

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.092,0.092,0.120,0.080,0.031,0.05,0.080
training/moderat.wav,125.000,1.000,0.120,0.060,0.063,0.088,0.060
training/samerano.wav,122.449,0.449,0.120,-0.080,0.188,0.251,0.080
training/purelove.wav,105.263,1.263,0.300,-0.020,0.331,0.6,0.020
```

### calmdad (grid bpm from activation peaks, downbeat phase unchanged)

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s
training/manucho.wav,110.092,0.092,0.120,0.080
training/moderat.wav,125.000,1.000,0.120,0.060
training/samerano.wav,122.449,0.449,0.120,-0.080
training/purelove.wav,105.263,1.263,0.300,-0.020
```

Observations:
- Tempo improves for manucho and samerano (sub‑0.5 BPM), but moderat and purelove still sit ~+1 BPM.
- Downbeat phase remains anchored at ~0.12s for the first three files; this appears driven by the earliest strong downbeat activation returned by the model rather than DBN phase selection.

### calmdad (peak-refined + deduped projection) — manucho/moderat/best

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.083,0.083,0.050,0.000,0.019,0.000,0.050,0.000,200,0.050,0.000,800,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.083
training/moderat.wav,124.102,0.102,0.048,0.015,0.015,0.000,0.048,0.015,187,0.048,0.015,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,0.254
training/best.wav,496.528,371.528,0.440,0.332,0.092,0.483,0.440,0.332,1192,0.063,0.000,4770,0.140,0.032,0.120,0.012,nan,0.049,0.108,374.849
```

### calmdad defaults (preset updates: dbn_floor=0.7, tempo_prior=1.0)

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.083,0.083,0.050,0.000,0.019,0.000,0.050,0.000,200,0.050,0.000,800,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.083
training/moderat.wav,124.102,0.102,0.048,0.015,0.015,0.000,0.048,0.015,187,0.048,0.015,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,0.254
training/samerano.wav,121.848,0.152,0.090,0.098,0.098,0.000,0.090,0.098,198,0.090,0.098,791,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.132
training/purelove.wav,105.324,1.324,0.210,0.121,0.121,0.000,0.210,0.121,118,0.210,0.121,472,0.300,0.031,0.120,0.211,nan,0.331,0.6,2.534
training/acht.wav,120.000,0.000,39.890,0.852,0.852,0.000,0.050,40.692,267,0.050,40.692,1065,0.120,40.622,0.120,40.622,nan,40.742,40.786,8.520
training/best.wav,121.516,3.484,0.063,0.000,0.014,0.000,0.063,0.000,287,0.063,0.000,1145,0.140,0.032,0.120,0.012,nan,0.049,0.108,3.484
```

### calmdad defaults (strict grid projection, no peak snap)

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.092,0.092,0.104,0.054,0.073,0.000,0.104,0.054,200,0.104,0.054,800,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.632
training/moderat.wav,125.000,1.000,0.104,0.016,0.041,0.000,0.104,0.016,187,0.104,0.016,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,1.160
training/samerano.wav,121.212,0.788,0.104,0.084,0.084,0.000,0.104,0.084,198,0.104,0.084,791,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.628
training/purelove.wav,106.195,2.195,0.284,0.047,0.047,0.000,0.284,0.047,118,0.284,0.047,472,0.300,0.031,0.120,0.211,nan,0.331,0.6,2.665
training/acht.wav,120.000,0.000,39.984,0.758,0.758,0.000,0.104,40.638,267,0.104,40.638,1065,0.120,40.622,0.120,40.622,nan,40.742,40.786,7.580
training/best.wav,121.212,3.788,0.124,0.016,0.075,0.000,0.124,0.016,287,0.124,0.016,1145,0.140,0.032,0.120,0.012,nan,0.049,0.108,3.948
```

### loose (all candidates, low floor, wide tolerance)

Flags: `--ml-dbn --ml-dbn-all-cand --ml-dbn-floor 0.01 --ml-dbn-max-cand 4096 --ml-dbn-tol 0.2`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.091,0.091,2.200,2.150,0.031,0.05,21.591
training/moderat.wav,123.645,0.355,1.500,1.412,0.063,0.088,14.475
training/samerano.wav,121.217,0.783,2.160,1.909,0.188,0.251,19.873
training/purelove.wav,104.479,0.479,2.580,1.980,0.331,0.6,20.279
training/acht.wav,117.798,2.202,60.720,19.934,40.742,40.786,201.542
```

### loose + lower tempo penalty + higher reward

Flags: `--ml-dbn --ml-dbn-all-cand --ml-dbn-floor 0.01 --ml-dbn-max-cand 4096 --ml-dbn-tol 0.1 --ml-dbn-tempo-pen 0.05 --ml-dbn-reward 0.7`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.071,0.071,2.200,2.150,0.031,0.05,21.571
training/moderat.wav,124.657,0.657,1.020,0.932,0.063,0.088,9.977
training/samerano.wav,122.038,0.038,2.160,1.909,0.188,0.251,19.128
training/purelove.wav,105.080,1.080,0.300,0.031,0.331,0.6,1.390
training/acht.wav,124.763,4.763,40.720,0.022,40.742,40.786,4.983
```

### mid (moderate floor/tolerance)

Flags: `--ml-dbn --ml-dbn-all-cand --ml-dbn-floor 0.02 --ml-dbn-max-cand 2048 --ml-dbn-tol 0.15`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.081,0.081,2.200,2.150,0.031,0.05,21.581
training/moderat.wav,124.180,0.180,1.500,1.412,0.063,0.088,14.300
training/samerano.wav,121.619,0.381,2.160,1.909,0.188,0.251,19.471
training/purelove.wav,104.421,0.421,2.580,1.980,0.331,0.6,20.221
training/acht.wav,117.653,2.347,60.720,19.934,40.742,40.786,201.687
```

### note on runtime

Attempted an even looser run with `--ml-dbn-floor 0.005 --ml-dbn-max-cand 8192`, but it timed out (>120s) for the full sweep.

## no-dbn (beatthis preset, DBN disabled)

Flags: `--ml-no-dbn`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,234.927,124.927,0.120,0.070,0.031,0.05,125.627
training/moderat.wav,248.399,124.399,0.120,0.032,0.063,0.088,124.719
training/samerano.wav,274.944,152.944,0.180,0.008,0.188,0.251,153.024
training/purelove.wav,217.963,113.963,0.300,0.031,0.331,0.6,114.273
training/acht.wav,289.483,169.483,40.720,0.022,40.742,40.786,169.703
```

## side-by-side (calmdad defaults vs no-dbn minimal)

```
file,dbn_bpm,dbn_downbeat_s,dbn_score,no_dbn_bpm,no_dbn_downbeat_s,no_dbn_score
training/manucho.wav,110.073,0.120,0.773,234.927,0.120,125.627
training/moderat.wav,124.178,0.120,0.498,248.399,0.120,124.719
training/samerano.wav,122.098,0.180,0.178,274.944,0.180,153.024
training/purelove.wav,105.095,0.300,1.405,217.963,0.300,114.273
training/acht.wav,119.597,42.220,14.743,289.483,40.720,169.703
```

## calmdad (intro/mid/outro stitch into single DBN)

Flags: `--ml-dbn --ml-dbn-mode calmdad --ml-dbn-window 30 --ml-dbn-imo --ml-dbn-stitch --ml-dbn-project-grid`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.249,2.249,1.740,1.690,0.031,0.05,19.149
training/moderat.wav,125.407,1.407,1.620,1.532,0.063,0.088,16.727
training/samerano.wav,121.247,0.753,0.180,0.008,0.188,0.251,0.833
training/purelove.wav,104.364,0.364,2.000,1.400,0.331,0.6,14.364
training/acht.wav,123.159,3.159,40.420,0.322,40.742,40.786,6.379
```

## calmdad (intro/mid/outro stitch + peak spacing clamp)

Flags: `--ml-dbn --ml-dbn-mode calmdad --ml-dbn-window 30 --ml-dbn-imo --ml-dbn-stitch --ml-dbn-project-grid`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.188,2.188,0.820,0.770,0.031,0.05,9.888
training/moderat.wav,61.500,62.500,0.540,0.452,0.063,0.088,67.020
training/samerano.wav,121.587,0.413,1.660,1.409,0.188,0.251,14.503
training/purelove.wav,104.187,0.187,1.280,0.680,0.331,0.6,6.987
training/acht.wav,284.886,164.886,40.420,0.322,40.742,40.786,168.106
```

## calmdad (best 30s window by activation)

Flags: `--ml-dbn --ml-dbn-mode calmdad --ml-dbn-window 30 --ml-dbn-project-grid`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.536,2.536,1.940,1.890,0.031,0.05,21.436
training/moderat.wav,125.656,1.656,1.020,0.932,0.063,0.088,10.976
training/samerano.wav,349.214,227.214,0.000,0.188,0.188,0.251,229.094
training/purelove.wav,108.466,4.466,0.860,0.260,0.331,0.6,7.066
training/acht.wav,646.641,526.641,40.800,0.014,40.742,40.786,526.781
```

## dbn window + grid projection (calmdad)

Flags: `--ml-dbn --ml-dbn-window 60 --ml-dbn-project-grid --ml-dbn-mode calmdad`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.536,2.536,303.280,303.230,0.031,0.05,3034.836
training/moderat.wav,125.656,1.656,2.960,2.872,0.063,0.088,30.376
training/samerano.wav,121.382,0.618,210.640,210.389,0.188,0.251,2104.508
training/purelove.wav,108.466,4.466,62.020,61.420,0.331,0.6,618.666
training/acht.wav,211.268,91.268,255.000,214.214,40.742,40.786,2233.408
```

### 3-minute window (same result)

Flags: `--ml-dbn --ml-dbn-window 180 --ml-dbn-project-grid --ml-dbn-mode calmdad`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.536,2.536,1.940,1.890,0.031,0.05,21.436
training/moderat.wav,125.656,1.656,1.020,0.932,0.063,0.088,10.976
training/samerano.wav,121.382,0.618,0.680,0.429,0.188,0.251,4.908
training/purelove.wav,108.466,4.466,0.860,0.260,0.331,0.6,7.066
training/acht.wav,211.268,91.268,40.680,0.062,40.742,40.786,91.888
```

### 30-second window (same result)

Flags: `--ml-dbn --ml-dbn-window 30 --ml-dbn-project-grid --ml-dbn-mode calmdad`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.536,2.536,1.940,1.890,0.031,0.05,21.436
training/moderat.wav,125.656,1.656,1.020,0.932,0.063,0.088,10.976
training/samerano.wav,121.382,0.618,0.680,0.429,0.188,0.251,4.908
training/purelove.wav,108.466,4.466,0.860,0.260,0.331,0.6,7.066
training/acht.wav,211.268,91.268,40.680,0.062,40.742,40.786,91.888
```

### 30-second window (intro/mid/outro selection)

Flags: `--ml-dbn --ml-dbn-window 30 --ml-dbn-project-grid --ml-dbn-mode calmdad --ml-dbn-imo`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.101,2.101,0.360,0.310,0.031,0.05,5.201
training/moderat.wav,125.656,1.656,1.020,0.932,0.063,0.088,10.976
training/samerano.wav,121.256,0.744,0.180,0.008,0.188,0.251,0.824
training/purelove.wav,104.160,0.160,2.440,1.840,0.331,0.6,18.560
training/acht.wav,122.806,2.806,40.340,0.402,40.742,40.786,6.826
```

### 30-second window (intro/mid/outro + consensus)

Flags: `--ml-dbn --ml-dbn-window 30 --ml-dbn-project-grid --ml-dbn-mode calmdad --ml-dbn-imo --ml-dbn-consensus`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.325,2.325,0.560,0.510,0.031,0.05,7.425
training/moderat.wav,126.295,2.295,0.120,0.032,0.063,0.088,2.615
training/samerano.wav,121.474,0.526,0.180,0.008,0.188,0.251,0.606
training/purelove.wav,104.438,0.438,0.300,0.031,0.331,0.6,0.748
training/acht.wav,137.594,17.594,40.420,0.322,40.742,40.786,20.814
```

## calmdad (intro/mid/outro stitch + peak spacing clamp) — rerun

file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.177,2.177,1.380,1.330,0.031,0.05,15.477
training/moderat.wav,125.658,1.658,0.540,0.452,0.063,0.088,6.178
training/samerano.wav,121.015,0.985,1.160,0.909,0.188,0.251,10.075
training/purelove.wav,168.975,64.975,0.640,0.040,0.331,0.6,65.375
training/acht.wav,175.592,55.592,40.880,0.094,40.742,40.786,56.532

**Observation:** Stitching doesn’t improve results on these canonicals — tempo errors (notably purelove/acht) and downbeat offsets remain large. So far, stitching provides no measurable improvement.

### 30-second window (intro/mid/outro selection, confidence-based)

Flags: `--ml-dbn --ml-dbn-window 30 --ml-dbn-project-grid --ml-dbn-mode calmdad --ml-dbn-imo`

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,112.526,2.526,0.120,0.070,0.031,0.05,3.226
training/moderat.wav,125.665,1.665,1.020,0.932,0.063,0.088,10.985
training/samerano.wav,121.350,0.650,1.160,0.909,0.188,0.251,9.740
training/purelove.wav,104.645,0.645,0.300,0.031,0.331,0.6,0.955
training/acht.wav,122.806,2.806,40.340,0.402,40.742,40.786,6.826
```

### calmdad defaults (peak-refined projection) — rerun

Defaults updated with peak-refined beat/downbeat frames before projection (±2 frames + parabola).

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.083,0.083,0.050,0.000,0.019,0.000,0.050,0.000,200,0.050,0.000,800,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.083
training/moderat.wav,124.102,0.102,0.048,0.015,0.015,0.000,0.048,0.015,187,0.048,0.015,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,0.254
training/samerano.wav,121.848,0.152,0.090,0.098,0.098,0.000,0.090,0.098,198,0.090,0.098,791,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.132
training/purelove.wav,104.559,0.559,0.210,0.121,0.121,0.000,0.210,0.121,152,0.210,0.121,455,0.300,0.031,0.120,0.211,nan,0.331,0.6,1.769
training/acht.wav,489.932,369.932,40.810,0.024,0.068,0.000,0.222,40.520,1478,0.000,40.742,4435,0.120,40.622,0.120,40.622,nan,40.742,40.786,370.172
training/best.wav,496.528,371.528,0.440,0.332,0.092,0.483,0.440,0.332,1192,0.063,0.000,4770,0.140,0.032,0.120,0.012,nan,0.049,0.108,374.849
```

## Canonical validation (separate run)

See `DBN_CANONICALS.md` for the command and raw output.

Summary:
- This run uses the default CLI path: full-file inference + DBN (no explicit window override). The output lists **all** beat/downbeat candidates, and the benchmark selects the candidate closest to the canonical onset/peak window.
- Across multiple files, **first beat/downbeat timestamps are consistently ~0.104s**, even when canonical downbeat onsets are earlier (0.031–0.088s) or later (0.188–0.331s). This points to a systematic phase anchor at ~0.104s rather than file-specific alignment.
- `beat_count`/`downbeat_count` show dense, full‑file grids (hundreds of events), so the issue is **phase alignment**, not missing beats.
- `downbeat_phase_err_s` is small-ish on manucho/moderat, but still outside the canonical onset/peak windows; for samerano/purelove the phase error grows, indicating the DBN is locking to a phase offset that is inconsistent with the canonical downbeat region.
- `acht.wav` shows **correct BPM** but **early downbeat estimate** (~0.64s before the canonical onset), with the first beat/downbeat still around 0.104s, confirming the same phase anchor behavior.

### Next instrumentation (needed to explain the ~0.104s anchor)

We should add explicit traces so we can explain *why* the first beat/downbeat locks at ~0.104s. Suggested logs (per file):
- **DBN input summary:** activation stats for the first 2s (min/max/mean, first N non‑zero frames, threshold used).
- **Candidate list preview:** first 8 beat candidates + first 8 downbeat candidates (frame + seconds).
- **DBN init state:** chosen tempo prior (bpm + period), beats_per_bar, phase‑prior window, and any phase offset applied.
- **Projection step:** anchor beat/downbeat frame chosen for grid projection and the resulting phase offset (seconds).
- **Window selection (if any):** window start/end timestamps and which window’s logits were used for DBN.

Once these are in place, re‑run canonicals and capture the traces alongside the CSV so we can tie the ~0.104s anchor to a concrete decision in the pipeline.

## DBN trace snapshots (manucho + moderat)

Raw logs are saved in:
- `DBN_TRACE_MANUCHO.txt`
- `DBN_TRACE_MODERAT.txt`

Key excerpts (trimmed for readability):

### manucho (trace)

```
Estimated BPM: 109.953
DBN window: start=14250 end=17250 frames=3000 (60s) selector=energy energy=phase
DBN calmdad prior: bpm=111.111 peaks=110 window_pct=0.1 clamp=[100,122.222]
DBN calmdad: frames=3000 floor=5e-06 epsilon=1e-05 tol=0.05 bpm=[100,122.222] step=1 lambda=100 use_downbeat=true beat[min,max]=[0.500001,0.731042] downbeat[min,max]=[0.5,0.731052]
DBN calmdad: first beat candidates (frame->s): 7->0.14 8->0.16 9->0.18 10->0.2 11->0.22 12->0.24 13->0.26 14->0.28 15->0.3 16->0.32
DBN calmdad: first downbeat candidates (frame->s): 0->0 1->0.02 2->0.04 3->0.06 4->0.08 5->0.1 6->0.12 7->0.14 8->0.16 9->0.18
DBN calmdad: best_bpb=4 beats=110 downbeats=28 best_score=372.837
DBN grid: bpm=114 bpm_from_peaks=109.967 bpm_from_downbeats=109.954 base_interval=27 bpm_for_grid=109.954 step_frames=27.2842 start_frame=14264
DBN grid: earliest_peak=6 earliest_downbeat_peak=6 earliest_downbeat_value=0.500001 activation_floor=0.05
DBN: beat head: 0->0 1->0 2->0 3->0 4->0 5->0 6->0.500002 7->0.500002 8->0.500002 9->0.500001 10->0.500001 11->0.500001
DBN: downbeat head: 0->0 1->0 2->0 3->0 4->0 5->0 6->0.500001 7->0.500001 8->0.500001 9->0.500001 10->0.5 11->0.5
DBN: downbeat peaks (first 100 frames): 83->0.500019 28->0.500014 97->0.500005 55->0.500003 14->0.500003
DBN: phase_window_frames=100 max_downbeat=0.500019 threshold=0.100004 best_phase=0 best_score=1.00049e+06
DBN: downbeat frames head: 6(0.12s) 115(2.3s) 224(4.48s) 333(6.66s) 443(8.86s) 552(11.04s)
DBN grid project: start_frame=6 start_time=0.12 bpm=109.954 step_frames=27.2842 total_frames=21600 latency_samples=706
DBN grid beats head: 6(0.12s) 33(0.66s) 61(1.22s) 88(1.76s) 115(2.3s) 142(2.84s)
```

### moderat (trace)

```
Estimated BPM: 124.001
DBN window: start=14250 end=17250 frames=3000 (60s) selector=energy energy=phase
DBN calmdad prior: bpm=125 peaks=124 window_pct=0.1 clamp=[112.5,137.5]
DBN calmdad: frames=3000 floor=5e-06 epsilon=1e-05 tol=0.05 bpm=[112.5,137.5] step=1 lambda=100 use_downbeat=true beat[min,max]=[0.5,0.731055] downbeat[min,max]=[0.5,0.731055]
DBN calmdad: first beat candidates (frame->s): 0->0 1->0.02 2->0.04 3->0.06 4->0.08 5->0.1 6->0.12 7->0.14 23->0.46 24->0.48
DBN calmdad: first downbeat candidates (frame->s): 0->0 1->0.02 2->0.04 3->0.06 4->0.08 5->0.1 6->0.12 7->0.14 8->0.16 9->0.18
DBN calmdad: best_bpb=4 beats=125 downbeats=32 best_score=420.522
DBN grid: bpm=119.5 bpm_from_peaks=124.008 bpm_from_downbeats=124.001 base_interval=24 bpm_for_grid=124.001 step_frames=24.1933 start_frame=14252
DBN grid: earliest_peak=6 earliest_downbeat_peak=6 earliest_downbeat_value=0.50091 activation_floor=0.05
DBN: beat head: 0->0 1->0 2->0 3->0 4->0 5->0 6->0.63407 7->0.554973 8->0.524256 9->0.512262 10->0.502989 11->0.501316
DBN: downbeat head: 0->0 1->0 2->0 3->0 4->0 5->0 6->0.50091 7->0.500283 8->0.500297 9->0.500169 10->0.500043 11->0.500034
DBN: downbeat peaks (first 100 frames): 75->0.722464 52->0.680209 99->0.523032 26->0.507152 29->0.501309
DBN: phase_window_frames=100 max_downbeat=0.722464 threshold=0.144493 best_phase=0 best_score=1.00049e+06
DBN: downbeat frames head: 6(0.12s) 103(2.06s) 200(4s) 296(5.92s) 393(7.86s) 490(9.8s)
DBN grid project: start_frame=6 start_time=0.12 bpm=124.001 step_frames=24.1933 total_frames=17950 latency_samples=706
DBN grid beats head: 6(0.12s) 30(0.6s) 54(1.08s) 79(1.58s) 103(2.06s) 127(2.54s)
```

Observation from traces:
- The **first downbeat candidate list includes 0.12s**, and the **phase search picks phase 0**, anchoring the grid at frame 6 (0.12s) for both tracks.
- Grid projection explicitly starts at `start_frame=6` → the early phase anchor propagates across the full file.

### Canonical sweep (including eureka) — 2026-01-25

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,284.924,284.874,1.047,285.940,284.924,284.874,68,284.924,284.874,270,0.120,0.070,0.120,0.070,nan,0.031,0.05,2848.787
training/moderat.wav,66.147,57.853,284.924,284.836,1.771,286.632,284.924,284.836,28,284.924,284.836,82,0.120,0.032,0.120,0.032,nan,0.063,0.088,2906.212
training/samerano.wav,121.989,0.011,269.924,269.673,0.204,269.532,269.924,269.673,64,269.924,269.673,255,0.120,0.068,0.120,0.068,nan,0.188,0.251,2696.741
training/purelove.wav,105.102,1.102,44.924,44.324,1.077,45.670,44.924,44.324,96,44.924,44.324,384,0.300,0.031,0.120,0.211,nan,0.331,0.6,444.342
training/acht.wav,120.016,0.016,194.924,154.138,0.203,153.979,194.924,154.138,169,194.924,154.138,675,0.120,40.622,0.120,40.622,nan,40.742,40.786,1541.396
training/best.wav,122.006,2.994,479.924,479.816,0.101,479.976,479.924,479.816,48,479.924,479.816,189,0.140,0.032,0.120,0.012,nan,0.049,0.108,4801.154
training/eureka.wav,119.989,0.011,179.924,179.891,0.093,180.017,179.924,179.891,91,179.924,179.891,361,0.120,0.087,0.120,0.087,nan,0.0,0.033,1798.921
```

Notes:
- `moderat.wav` BPM estimate collapsed to ~66 BPM in this run.
- Several files show **very late downbeat estimates** (hundreds of seconds), which means the phase anchor or window mapping is still off in this configuration.

### Tempo anchor diagnostics (manucho + moderat) — 2026-01-25

Added measurement-only debug to capture peak histogram top bins and peak-median BPM.

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,beat_median_bpm,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s
training/manucho.wav,111.099,1.099,111.100,107.183,111.111,111.100,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.111,0.324,0.274,0.293
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,0.284,0.196,0.221
```

Observation:
- Peaks histogram shows a **single dominant bin** (+1 BPM above canon on both files).
- Beat-median BPM matches the peak bin (beats are inheriting the peak bias).
- Autocorr is closer on moderat (120.253) but low on manucho (107.183), so it is not a universal correction.

### Tempo anchor diagnostics (all canonicals) — 2026-01-25

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,beat_median_bpm,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,111.099,1.099,111.100,107.183,111.111,111.100,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.111,0.324,0.274,0.293,0.000,0.324,0.274,200,0.324,0.274,800,0.120,0.070,0.120,0.070,nan,0.031,0.05,3.839
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.989,0.011,122.400,125.070,122.449,122.400,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),120.000,0.384,0.133,0.196,0.000,0.384,0.133,201,0.384,0.133,803,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.341
training/purelove.wav,105.102,1.102,105.250,53.571,105.263,105.250,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),103.448,0.404,0.000,0.073,0.000,0.404,0.000,116,0.404,0.000,462,0.300,0.031,0.120,0.211,nan,0.331,0.6,1.102
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,122.006,2.994,122.400,111.649,122.449,122.400,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),120.000,0.444,0.336,0.395,0.000,0.444,0.336,291,0.444,0.336,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,6.354
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721
```

Observation:
- For **most tracks**, `peak_hist_top` has a **single dominant bin** that matches `anchor_peaks` and `beat_median_bpm`, indicating the decoded beats inherit the peak bias.
- The bias is **not always +1 BPM**:  
  - moderat: +1  
  - manucho: +1  
  - best: ~‑2 (peak bin ~122.4 vs canon 125)  
  - purelove: +1 (105.25 vs 104)  
  - acht/eureka: ~0  
  This suggests track‑specific peak histogram dominance rather than a fixed quantization offset.

### Tempo anchor diagnostics (all canonicals, extended) — 2026-01-25

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,111.099,1.099,111.100,107.183,111.111,111.100,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,0.324,0.274,0.293,0.000,0.324,0.274,200,0.324,0.274,800,0.120,0.070,0.120,0.070,nan,0.031,0.05,3.839
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.989,0.011,122.400,125.070,122.449,122.400,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,0.384,0.133,0.196,0.000,0.384,0.133,201,0.384,0.133,803,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.341
training/purelove.wav,105.102,1.102,105.250,53.571,105.263,105.250,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,0.404,0.000,0.073,0.000,0.404,0.000,116,0.404,0.000,462,0.300,0.031,0.120,0.211,nan,0.331,0.6,1.102
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,122.006,2.994,122.400,111.649,122.449,122.400,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,0.444,0.336,0.395,0.000,0.444,0.336,291,0.444,0.336,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,6.354
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721
```

Observation:
- `peak_top1_ratio` ranges ~0.31–0.86; higher ratios indicate a **strong single-bin dominance** (likely stable tempo).
- `peak_top_gap` and `peak_top_range_bpm` can flag ambiguity: small gap + wide range implies **multi‑modal peaks**.

### Rule suggestion (measurement-only) — 2026-01-25

```
file,anchor_peaks,anchor_autocorr,anchor_beats,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,rule_suggest,rule_bpm,rule_delta_bpm
training/manucho.wav,111.100,107.183,111.111,0.353,309.507,4.000,autocorr,107.183,-2.817
training/moderat.wav,125.000,120.253,125.000,0.502,651.794,5.000,autocorr,120.253,-3.747
training/samerano.wav,122.400,125.070,122.449,0.375,299.137,1.200,autocorr,125.070,3.070
training/purelove.wav,105.250,53.571,105.263,0.312,47.334,0.950,autocorr,53.571,-50.429
training/acht.wav,120.000,124.753,120.000,0.805,1333.080,4.650,peaks_dom,120.000,0.000
training/best.wav,122.400,111.649,122.449,0.355,282.865,1.200,autocorr,111.649,-13.351
training/eureka.wav,120.000,59.995,120.000,0.827,1352.670,1.600,peaks_dom,120.000,0.000
```

Observation:
- This rule set **over-selects autocorr** on multiple tracks where autocorr is clearly wrong (purelove, best, manucho).
- A simple dominance rule is not sufficient; autocorr needs a **sanity gate** (e.g., reject if it differs from peaks by >5 BPM or from beats by >3 BPM).

### Canonical run (anchor selection v2 + comb stats)

Change set:
- Anchor selection now prioritizes **peaks/comb/autocorr consensus** (2% tolerance).
- Comb‑based tempo estimator logged (`comb_top*` columns).
- CLI anchor log now includes `comb` value; stream uses the same anchor policy.

Run:
```
python3 scripts/dbn_benchmark.py
```

Result:
```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,111.105,1.105,111.100,107.183,111.111,111.111,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,0.324,0.274,0.293,0.000,0.324,0.274,200,0.324,0.274,800,0.120,0.070,0.120,0.070,nan,0.031,0.05,3.845
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,0.124,0.064,0.064,0.000,0.124,0.064,200,0.124,0.064,798,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.440
training/purelove.wav,104.349,0.349,105.250,53.571,103.448,104.348,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,0.084,0.247,0.247,0.000,0.084,0.247,115,0.084,0.247,460,0.300,0.031,0.120,0.211,nan,0.331,0.6,2.819
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,121.200,3.800,122.400,111.649,120.000,121.212,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,0.224,0.116,0.175,0.000,0.224,0.116,289,0.224,0.116,1156,0.140,0.032,0.120,0.012,nan,0.049,0.108,4.960
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721
```

Observation:
- Comb estimator tracks the same dominant peak as the activation histogram for most files.
- Autocorr anchor is often pulled to half‑time (purelove, eureka) which is why `rule_suggest` tends to “autocorr.”
- The phase anchor (downbeat_est_s) remains early (~0.12s) when the first strong downbeat logits arrive at frame ~6 (0.12s), regardless of tempo anchor.

## Canonical run (DBN quality metrics)

Added qpar/qmax/qkur from DBN quality trace (60s window).

### DBN quality metrics (what they mean)

- **qpar** (peak‑to‑RMS of tempo salience): ratio of the strongest tempo peak to the RMS of the salience curve. High means one tempo dominates; low means ambiguous, many similar peaks.
- **qmax** (max salience): absolute height of the strongest tempo peak. High means strong periodic energy; low means weak periodicity overall.
- **qkur** (kurtosis of salience): how “spiky” the salience distribution is. High means a few sharp peaks (clear tempo), low means flat / smeared (uncertain).



## Canonical run (DBN quality metrics)

Added qpar/qmax/qkur from DBN quality trace (60s window).

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,111.105,1.105,111.100,107.183,111.111,111.111,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,0.324,0.274,0.293,0.000,0.324,0.274,200,0.324,0.274,800,0.120,0.070,0.120,0.070,nan,0.031,0.05,3.845
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0.124,0.064,0.064,0.000,0.124,0.064,200,0.124,0.064,798,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.440
training/purelove.wav,104.349,0.349,105.250,53.571,103.448,104.348,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0.084,0.247,0.247,0.000,0.084,0.247,115,0.084,0.247,460,0.300,0.031,0.120,0.211,nan,0.331,0.6,2.819
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,121.200,3.800,122.400,111.649,120.000,121.212,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0.224,0.116,0.175,0.000,0.224,0.116,289,0.224,0.116,1156,0.140,0.032,0.120,0.012,nan,0.049,0.108,4.960
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721


## Canonical run (DBN quality metrics)

Added qpar/qmax/qkur from DBN quality trace (60s window).

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,111.105,1.105,111.100,107.183,111.111,111.111,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,0.324,0.274,0.293,0.000,0.324,0.274,200,0.324,0.274,800,0.120,0.070,0.120,0.070,nan,0.031,0.05,3.845
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0.124,0.064,0.064,0.000,0.124,0.064,200,0.124,0.064,798,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.440
training/purelove.wav,104.349,0.349,105.250,53.571,103.448,104.348,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0.084,0.247,0.247,0.000,0.084,0.247,115,0.084,0.247,460,0.300,0.031,0.120,0.211,nan,0.331,0.6,2.819
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,121.200,3.800,122.400,111.649,120.000,121.212,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0.224,0.116,0.175,0.000,0.224,0.116,289,0.224,0.116,1156,0.140,0.032,0.120,0.012,nan,0.049,0.108,4.960
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721

```


## Canonical run (quality‑gated grid BPM selection)

Rule: treat DBN quality as low when qkur < 4.0. In low‑quality cases, skip global/linear fit and use peak‑median (fallback to peak/reg) before any downbeat override.

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,0.084,0.034,0.053,0.000,0.084,0.034,198,0.084,0.034,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.387
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0.124,0.064,0.064,0.000,0.124,0.064,200,0.124,0.064,798,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.440
training/purelove.wav,105.102,1.102,105.250,53.571,103.448,105.263,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0.404,0.000,0.073,0.000,0.404,0.000,116,0.404,0.000,462,0.300,0.031,0.120,0.211,nan,0.331,0.6,1.102
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,122.006,2.994,122.400,111.649,120.000,122.449,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0.444,0.336,0.395,0.000,0.444,0.336,291,0.444,0.336,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,6.354
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721

```


## Canonical run (quality‑gated grid BPM selection, qkur<5.0)

Rule: treat DBN quality as low when qkur < 5.0. In low‑quality cases, skip global/linear fit and use peak‑median (fallback to peak/reg) before any downbeat override.

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,0.084,0.034,0.053,0.000,0.084,0.034,198,0.084,0.034,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.387
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0.124,0.064,0.064,0.000,0.124,0.064,200,0.124,0.064,798,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.440
training/purelove.wav,105.102,1.102,105.250,53.571,103.448,105.263,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0.404,0.000,0.073,0.000,0.404,0.000,116,0.404,0.000,462,0.300,0.031,0.120,0.211,nan,0.331,0.6,1.102
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,122.006,2.994,122.400,111.649,120.000,122.449,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0.444,0.336,0.395,0.000,0.444,0.336,291,0.444,0.336,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,6.354
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721

```


## Canonical run (quality‑gated grid BPM selection, qkur<4.5 || qpar<1.015)

Rule: treat DBN quality as low when qkur < 4.5 **or** qpar < 1.015. In low‑quality cases, skip global/linear fit and use peak‑median (fallback to peak/reg) before any downbeat override.

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,0.084,0.034,0.053,0.000,0.084,0.034,198,0.084,0.034,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.387
training/moderat.wav,124.810,0.810,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0.324,0.236,0.261,0.000,0.324,0.236,249,0.324,0.236,747,0.120,0.032,0.120,0.032,nan,0.063,0.088,3.170
training/samerano.wav,121.989,0.011,122.400,125.070,120.000,122.449,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0.384,0.133,0.196,0.000,0.384,0.133,201,0.384,0.133,803,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.341
training/purelove.wav,105.102,1.102,105.250,53.571,103.448,105.263,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0.404,0.000,0.073,0.000,0.404,0.000,116,0.404,0.000,462,0.300,0.031,0.120,0.211,nan,0.331,0.6,1.102
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,122.006,2.994,122.400,111.649,120.000,122.449,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0.444,0.336,0.395,0.000,0.444,0.336,291,0.444,0.336,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,6.354
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721

```


## Canonical run (quality‑gated grid BPM selection, qkur<4.5 || qpar<1.02)

Rule: treat DBN quality as low when qkur < 4.5 **or** qpar < 1.02. In low‑quality cases, skip global/linear fit and use peak‑median (fallback to peak/reg) before any downbeat override.

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,0.084,0.034,0.053,0.000,0.084,0.034,198,0.084,0.034,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.387
training/moderat.wav,124.810,0.810,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0.324,0.236,0.261,0.000,0.324,0.236,249,0.324,0.236,747,0.120,0.032,0.120,0.032,nan,0.063,0.088,3.170
training/samerano.wav,121.989,0.011,122.400,125.070,120.000,122.449,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0.384,0.133,0.196,0.000,0.384,0.133,201,0.384,0.133,803,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.341
training/purelove.wav,105.102,1.102,105.250,53.571,103.448,105.263,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0.404,0.000,0.073,0.000,0.404,0.000,116,0.404,0.000,462,0.300,0.031,0.120,0.211,nan,0.331,0.6,1.102
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,122.006,2.994,122.400,111.649,120.000,122.449,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0.444,0.336,0.395,0.000,0.444,0.336,291,0.444,0.336,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,6.354
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721

```


## Manucho tightening (quality gate stress test)

Tightened quality gate aggressively and ran **manucho only**.

```
rule,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,dbn_qpar,dbn_qmax,dbn_qkur
qkur<8||qpar<1.05,109.953,0.047,111.100,107.183,111.111,110.092,111.106,1.016420,0.289774,3.410520
qkur<12||qpar<1.10,109.953,0.047,111.100,107.183,111.111,110.092,111.106,1.016420,0.289774,3.410520
```


## Manucho gate instrumentation

Added gate counters to understand which BPM sources are skipped and which path wins.

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,dbn_gate_used,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,1,0,1,1,downbeats_override,0.084,0.034,0.053,0.000,0.084,0.034,198,0.084,0.034,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.387

```


## Manucho gate instrumentation (pre‑override logging)

Shows the BPM source *before* downbeat override.

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,1,0,1,1,downbeats_override,peaks_median,110.175,0.084,0.034,0.053,0.000,0.084,0.034,198,0.084,0.034,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.387

```


## Canonical run (gate counters, downbeat override disabled when quality_low)

Disabled downbeat override when quality_low=1. Logged pre/post BPM selection and override counts.

```
rows=8 low=8 overrides=0
file,low,used,pre_override,pre_bpm,post_bpm
training/manucho.wav,1,peaks_median,peaks_median,110.175,110.175
training/moderat.wav,1,peaks_median,peaks_median,124.809,124.810
training/samerano.wav,1,peaks_median,peaks_median,122.218,122.218
training/purelove.wav,1,peaks_median,peaks_median,105.170,105.170
training/acht.wav,1,peaks_median,peaks_median,120.000,120.000
training/best.wav,1,peaks_median,peaks_median,121.656,121.657
training/eureka.wav,1,peaks_median,peaks_median,120.002,120.001
```

file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,dbn_gate_post_bpm,dbn_gate_override,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.175,0.175,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,1,0,1,1,peaks_median,peaks_median,110.175,110.175,0,0.104,0.054,0.073,0.000,0.104,0.054,199,0.104,0.054,794,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.715
training/moderat.wav,124.810,0.810,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,1,0,1,1,peaks_median,peaks_median,124.809,124.810,0,0.324,0.236,0.261,0.000,0.324,0.236,249,0.324,0.236,747,0.120,0.032,0.120,0.032,nan,0.063,0.088,3.170
training/samerano.wav,122.218,0.218,122.400,125.070,120.000,122.449,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,1,0,1,1,peaks_median,peaks_median,122.218,122.218,0,0.404,0.153,0.216,0.000,0.404,0.153,202,0.404,0.153,805,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.748
training/purelove.wav,105.170,1.170,105.250,53.571,103.448,105.263,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,1,0,1,1,peaks_median,peaks_median,105.170,105.170,0,0.424,0.000,0.093,0.000,0.424,0.000,116,0.424,0.000,463,0.300,0.031,0.120,0.211,nan,0.331,0.6,1.170
training/acht.wav,120.000,0.000,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,1,0,1,1,peaks_median,peaks_median,120.000,120.000,0,40.424,0.318,0.318,0.000,0.424,40.318,266,0.424,40.318,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,3.180
training/best.wav,121.657,3.343,122.400,111.649,120.000,121.212,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,1,0,1,1,peaks_median,peaks_median,121.656,121.657,0,0.044,0.005,0.005,0.000,0.044,0.005,291,0.044,0.005,1161,0.140,0.032,0.120,0.012,nan,0.049,0.108,3.393
training/eureka.wav,120.001,0.001,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,1,0,1,1,peaks_median,peaks_median,120.002,120.001,0,0.424,0.391,0.424,0.000,0.424,0.391,180,0.424,0.391,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.911

```


## Canonical run (gate retune: qkur<4.0)

Rule: treat DBN quality as low when qkur < 4.0. Downbeat override disabled only when quality_low=1.

Summary:
- low=3 of 8
- overrides=0

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,dbn_gate_post_bpm,dbn_gate_override,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.175,0.175,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,1,0,1,1,peaks_median,peaks_median,110.175,110.175,0,0.104,0.054,0.073,0.000,0.104,0.054,199,0.104,0.054,794,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.715
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0,0,0,0,global_fit,global_fit,66.147,125.000,0,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0,0,0,0,global_fit,global_fit,57.157,121.200,0,0.124,0.064,0.064,0.000,0.124,0.064,200,0.124,0.064,798,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.440
training/purelove.wav,105.170,1.170,105.250,53.571,103.448,105.263,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,1,0,1,1,peaks_median,peaks_median,105.170,105.170,0,0.424,0.000,0.093,0.000,0.424,0.000,116,0.424,0.000,463,0.300,0.031,0.120,0.211,nan,0.331,0.6,1.170
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,0,0,0,0,global_fit,global_fit,38.206,120.016,0,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,121.657,3.343,122.400,111.649,120.000,121.212,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,1,0,1,1,peaks_median,peaks_median,121.656,121.657,0,0.044,0.005,0.005,0.000,0.044,0.005,291,0.044,0.005,1161,0.140,0.032,0.120,0.012,nan,0.049,0.108,3.393
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0,0,0,0,global_fit,global_fit,64.026,119.989,0,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721

```


## Canonical run (gate retune: qkur<3.8)

Rule: treat DBN quality as low when qkur < 3.8. Downbeat override disabled only when quality_low=1.

Summary:
- low=1 of 8 (manucho only)
- overrides=0

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,dbn_gate_post_bpm,dbn_gate_override,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.175,0.175,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,1,0,1,1,peaks_median,peaks_median,110.175,110.175,0,0.104,0.054,0.073,0.000,0.104,0.054,199,0.104,0.054,794,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.715
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0,0,0,0,global_fit,global_fit,66.147,125.000,0,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0,0,0,0,global_fit,global_fit,57.157,121.200,0,0.124,0.064,0.064,0.000,0.124,0.064,200,0.124,0.064,798,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.440
training/purelove.wav,104.349,0.349,105.250,53.571,103.448,104.348,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0,0,0,0,global_fit,global_fit,87.999,104.349,0,0.084,0.247,0.247,0.000,0.084,0.247,115,0.084,0.247,460,0.300,0.031,0.120,0.211,nan,0.331,0.6,2.819
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,0,0,0,0,global_fit,global_fit,38.206,120.016,0,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,121.200,3.800,122.400,111.649,120.000,121.212,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0,0,0,0,global_fit,global_fit,36.804,121.200,0,0.224,0.116,0.175,0.000,0.224,0.116,289,0.224,0.116,1156,0.140,0.032,0.120,0.012,nan,0.049,0.108,4.960
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0,0,0,0,global_fit,global_fit,64.026,119.989,0,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721

```


## Canonical run (gate retune: qkur<3.6)

Rule: treat DBN quality as low when qkur < 3.6. Downbeat override disabled only when quality_low=1.

Summary:
- low=1 of 8 (manucho only)
- overrides=0

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,dbn_gate_post_bpm,dbn_gate_override,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.175,0.175,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,1,0,1,1,peaks_median,peaks_median,110.175,110.175,0,0.104,0.054,0.073,0.000,0.104,0.054,199,0.104,0.054,794,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.715
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0,0,0,0,global_fit,global_fit,66.147,125.000,0,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0,0,0,0,global_fit,global_fit,57.157,121.200,0,0.124,0.064,0.064,0.000,0.124,0.064,200,0.124,0.064,798,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.440
training/purelove.wav,104.349,0.349,105.250,53.571,103.448,104.348,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0,0,0,0,global_fit,global_fit,87.999,104.349,0,0.084,0.247,0.247,0.000,0.084,0.247,115,0.084,0.247,460,0.300,0.031,0.120,0.211,nan,0.331,0.6,2.819
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,0,0,0,0,global_fit,global_fit,38.206,120.016,0,40.444,0.298,0.298,0.000,0.444,40.298,266,0.444,40.298,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,2.996
training/best.wav,121.200,3.800,122.400,111.649,120.000,121.212,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0,0,0,0,global_fit,global_fit,36.804,121.200,0,0.224,0.116,0.175,0.000,0.224,0.116,289,0.224,0.116,1156,0.140,0.032,0.120,0.012,nan,0.049,0.108,4.960
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0,0,0,0,global_fit,global_fit,64.026,119.989,0,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721

```

### Experiment: Downbeat override gate (CV/count)

Added a downbeat override gate so the downbeat-based BPM override only triggers when:
- `quality_low == 0`
- `downbeat_count >= 8`
- `downbeat_cv <= 0.05`

This is logged as `downbeat_ok`, `downbeat_cv`, and `downbeat_count` in the `DBN quality gate` line.

Manucho (only):

```
file,bpm_est,bpm_err,dbn_gate_low,dbn_gate_downbeat_ok,dbn_gate_downbeat_cv,dbn_gate_downbeat_count,dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,dbn_gate_post_bpm,dbn_gate_override,downbeat_est_s,downbeat_err_s,score
training/manucho.wav,110.175,0.175,1,0,0.009486,27,peaks_median,peaks_median,110.175,110.175,0,0.104,0.054,0.715
```

Notes:
- `quality_low=1` blocks downbeat override as intended; `downbeat_ok=0`.
- `downbeat_cv` is very low (stable), but the override is still blocked because of low quality.

### Canonical run: Downbeat override gate (CV/count)

Ran full canonicals with downbeat override gate enabled. Summary:

- `quality_low=1` for **manucho** only (qkur < 3.6), so override blocked there.
- **No file used downbeat override** in this run (`override_used=0`), because the 0.5% ratio check did not pass.

Counts:

```
rows=8 quality_low=1 downbeat_ok=6 override_used=0
```

This confirms the new downbeat gate is active and logged, but it’s not yet influencing BPM selection in current canonicals.

### Grid drift math (BPM error → beat drift)

For a fixed grid over long files, even tiny BPM errors accumulate into audible drift.  
Let `ΔBPM` be the BPM error and `H` the duration in hours.

- **Drift (beats) ≈ |ΔBPM| × 60 × H**

At 120 BPM:

| ΔBPM | Drift after 1 hr | Drift after 2 hr | Drift after 3 hr |
|---:|---:|---:|---:|
| 0.10 | 6.0 beats | 12.0 beats | 18.0 beats |
| 0.05 | 3.0 beats | 6.0 beats | 9.0 beats |
| 0.02 | 1.2 beats | 2.4 beats | 3.6 beats |
| 0.01 | 0.6 beats | 1.2 beats | 1.8 beats |
| 0.005 | 0.3 beats | 0.6 beats | 0.9 beats |

**Target precision:** for ≤1 beat drift over 2 hours, `ΔBPM ≤ 0.0083`.

Implication: a single short window rarely achieves this precision. Multi‑window or global regression is required for 2‑hour grids.

### Canonical run: multi-window consensus (5 evenly spaced windows)

Implementation details:
- Consensus windows are now **five evenly spaced windows** across the usable range.
- Window start fractions: **0.20, 0.35, 0.50, 0.65, 0.80** of `(total_frames - window_frames)`.
- Duplicates are deduped; if none, fallback to the best window span.
- `dbn_window_consensus` is now **enabled by default** in the BeatThis preset.

Run results (same columns as previous canonicals):

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,dbn_gate_downbeat_ok,dbn_gate_downbeat_cv,dbn_gate_downbeat_count,dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,dbn_gate_post_bpm,dbn_gate_override,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.175,0.175,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,1,0,1,1,0,0.009486,27,peaks_median,peaks_median,110.175,110.175,0,0.104,0.054,0.073,0.000,0.104,0.054,265,0.104,0.054,794,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.715
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0,0,0,0,1,0.003177,30,global_fit,global_fit,66.147,125.000,0,0.284,0.196,0.221,0.000,0.284,0.196,250,0.284,0.196,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,2.960
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0,0,0,0,1,0.001753,29,global_fit,global_fit,57.157,121.200,0,0.124,0.064,0.064,0.000,0.124,0.064,266,0.124,0.064,798,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.440
training/purelove.wav,104.349,0.349,105.250,53.571,103.448,104.348,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0,0,0,0,0,0.095608,24,global_fit,global_fit,87.999,104.349,0,0.084,0.247,0.247,0.000,0.084,0.247,115,0.084,0.247,460,0.300,0.031,0.120,0.211,nan,0.331,0.6,2.819
training/acht.wav,120.000,0.000,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,0,0,0,0,1,0.000840,29,global_fit,global_fit,38.206,120.000,0,40.924,0.138,0.182,0.000,0.424,40.318,355,0.424,40.318,1064,0.120,40.622,0.120,40.622,nan,40.742,40.786,1.380
training/best.wav,121.200,3.800,122.400,111.649,120.000,121.212,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0,0,0,0,1,0.003680,29,global_fit,global_fit,36.804,121.200,0,0.224,0.116,0.175,0.000,0.224,0.116,289,0.224,0.116,1156,0.140,0.032,0.120,0.012,nan,0.049,0.108,4.960
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0,0,0,0,1,0.005214,29,global_fit,global_fit,64.026,119.989,0,0.404,0.371,0.404,0.000,0.404,0.371,180,0.404,0.371,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,3.721
```

### Canonical run: consensus with downbeat confidence + phase median

Consensus changes:
- Drop windows with weak downbeat confidence (downbeat_count < 4 or score < 0.6 * best).
- Compute **median BPM** from remaining windows.
- Compute **weighted median downbeat phase offset** (weights = downbeat_peak_score).
- Align grid start to that consensus phase.

Run results:

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,dbn_gate_downbeat_ok,dbn_gate_downbeat_cv,dbn_gate_downbeat_count,dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,dbn_gate_post_bpm,dbn_gate_override,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.175,0.175,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,1,0,1,1,0,0.009486,27,peaks_median,peaks_median,110.175,110.175,0,0.244,0.194,0.213,0.000,0.244,0.194,199,0.244,0.194,793,0.120,0.070,0.120,0.070,nan,0.031,0.05,2.115
training/moderat.wav,125.000,1.000,125.000,120.253,125.000,125.000,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,125.000,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0,0,0,0,1,0.003177,30,global_fit,global_fit,66.147,125.000,0,0.044,0.019,0.019,0.000,0.044,0.019,187,0.044,0.019,748,0.120,0.032,0.120,0.032,nan,0.063,0.088,1.190
training/samerano.wav,121.200,0.800,122.400,125.070,120.000,121.212,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0,0,0,0,1,0.001753,29,global_fit,global_fit,57.157,121.200,0,0.064,0.124,0.124,0.000,0.064,0.124,200,0.064,0.124,799,0.120,0.068,0.120,0.068,nan,0.188,0.251,2.040
training/purelove.wav,104.349,0.349,105.250,53.571,103.448,104.348,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0,0,0,0,0,0.095608,24,global_fit,global_fit,87.999,104.349,0,0.044,0.287,0.287,0.000,0.044,0.287,115,0.044,0.287,460,0.300,0.031,0.120,0.211,nan,0.331,0.6,3.219
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,0,0,0,0,1,0.000840,29,global_fit,global_fit,38.206,120.016,0,40.204,0.538,0.538,0.000,0.204,40.538,267,0.204,40.538,1065,0.120,40.622,0.120,40.622,nan,40.742,40.786,5.396
training/best.wav,121.200,3.800,122.400,111.649,120.000,121.212,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0,0,0,0,1,0.003680,29,global_fit,global_fit,36.804,121.200,0,0.344,0.236,0.295,0.000,0.344,0.236,289,0.344,0.236,1156,0.140,0.032,0.120,0.012,nan,0.049,0.108,6.160
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0,0,0,0,1,0.005214,29,global_fit,global_fit,64.026,119.989,0,0.464,0.431,0.464,0.000,0.464,0.431,180,0.464,0.431,719,0.120,0.087,0.120,0.087,nan,0.0,0.033,4.321
```


### Full benchmark run (post onset‑phase alignment, CPU)

```
file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,rule_suggest,rule_bpm,rule_delta_bpm,dbn_qpar,dbn_qmax,dbn_qkur,dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,dbn_gate_downbeat_ok,dbn_gate_downbeat_cv,dbn_gate_downbeat_count,dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,dbn_gate_post_bpm,dbn_gate_override,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,110.175,0.175,111.100,107.183,111.111,110.092,111.106,111.111,111.1(724.545) 110.05(415.038) 109.75(381.403) 109.05(353.603) 107.1(178.893),111.100,724.545,110.050,415.038,0.353,309.507,4.000,111.111,111.111,104.053,54.545,67.477,0.439,36.577,autocorr,107.183,-2.817,1.016420,0.289774,3.410520,1,0,0,1,0,0.009486,27,peaks_median,peaks_median,110.175,110.175,0,0.104,0.054,0.073,0.000,0.104,0.054,199,0.104,0.054,793,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.715
training/moderat.wav,92.875,31.125,125.000,120.253,125.000,93.023,125.000,125.000,125(1050.74) 123.7(398.943) 123.25(301.033) 122.4(216.977) 120(123.465),125.000,1050.740,123.700,398.943,0.502,651.794,5.000,93.750,125.000,94.174,120.000,63.302,0.432,30.872,autocorr,120.253,-3.747,1.013110,0.274369,6.879550,0,0,0,0,1,0.127400,31,downbeats_primary,downbeats_primary,92.874,92.875,0,0.344,0.256,0.281,0.000,0.344,0.256,185,0.344,0.256,555,0.120,0.032,0.120,0.032,nan,0.063,0.088,33.685
training/samerano.wav,121.989,0.011,122.400,125.070,120.000,122.449,121.200,120.000,122.4(718.743) 120(419.605) 121.6(364.579) 125(228.665) 121.2(183.696),122.400,718.743,120.000,419.605,0.375,299.137,1.200,120.000,120.000,70.853,125.000,60.897,0.377,9.955,autocorr,125.070,3.070,1.011060,0.270079,8.359100,0,0,0,0,1,0.001753,29,downbeats_primary,downbeats_primary,121.989,121.989,0,0.000,0.188,0.188,0.000,0.000,0.188,201,0.000,0.188,804,0.120,0.068,0.120,0.068,nan,0.188,0.251,1.891
training/purelove.wav,105.225,1.225,105.250,53.571,103.448,105.263,104.349,103.448,105.25(324.62) 103.4(277.286) 107.1(182.503) 104.65(159.439) 104.3(96.0967),105.250,324.620,103.400,277.286,0.312,47.334,0.950,103.448,103.448,53.460,107.143,52.219,0.346,1.241,autocorr,53.571,-50.429,1.019910,0.297014,3.879220,0,0,0,0,1,0.169291,26,downbeats_primary,downbeats_primary,105.225,105.225,0,0.104,0.227,0.227,0.000,0.104,0.227,116,0.104,0.227,463,0.300,0.031,0.120,0.211,nan,0.331,0.6,3.495
training/acht.wav,120.016,0.016,120.000,124.753,120.000,120.000,120.000,120.000,120(1422.81) 121.6(89.7344) 125(88.1934) 118.4(84.6656) 115.35(80.9973),120.000,1422.810,121.600,89.734,0.805,1333.080,4.650,120.000,120.000,108.177,60.000,73.060,0.434,35.117,peaks_dom,120.000,0.000,1.014410,0.281371,4.524260,0,0,0,0,1,0.000840,29,downbeats_primary,downbeats_primary,120.016,120.016,0,40.044,0.698,0.698,0.000,0.044,40.698,267,0.044,40.698,1065,0.120,40.622,0.120,40.622,nan,40.742,40.786,6.996
training/best.wav,122.006,2.994,122.400,111.649,120.000,122.449,121.200,120.000,122.4(851.012) 120(568.147) 121.6(423.963) 125(319.106) 121.2(236.576),122.400,851.012,120.000,568.147,0.355,282.865,1.200,120.000,120.000,163.850,125.000,156.471,0.374,7.378,autocorr,111.649,-13.351,1.017340,0.286477,3.995310,0,0,0,0,1,0.003680,29,downbeats_primary,downbeats_primary,122.005,122.006,0,0.000,0.049,0.049,0.000,0.000,0.049,291,0.000,0.049,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,3.484
training/eureka.wav,119.989,0.011,120.000,59.995,120.000,120.000,120.000,120.000,120(1432.19) 121.6(79.5122) 122.4(75.8035) 121.2(73.6716) 118.4(70.8321),120.000,1432.190,121.600,79.512,0.827,1352.670,1.600,120.000,120.000,101.297,60.000,66.686,0.451,34.611,peaks_dom,120.000,0.000,1.016740,0.282240,4.945920,0,0,0,0,1,0.005214,29,downbeats_primary,downbeats_primary,119.989,119.989,0,0.000,0.000,0.000,0.000,0.000,0.000,180,0.000,0.000,720,0.120,0.087,0.120,0.087,nan,0.0,0.033,0.011
```

### BeatThis CoreML DBN baseline (current branch, CPU-only)

Command:

```bash
python3 scripts/dbn_benchmark.py \
  --beatit build/beatit \
  --backend coreml \
  --model coreml_out_latest/BeatThis_small0.mlpackage \
  --pass-args --ml-cpu-only \
  > logs/dbn_benchmark_coreml_latest_cpu.csv
```

Per-track summary (`file,status,bpm,bpm_err,downbeat_s,downbeat_err,score`):

```csv
training/manucho.wav,ok,110.081,0.081,1.640,1.590,15.981
training/moderat.wav,ok,123.982,0.018,1.520,1.432,14.338
training/samerano.wav,ok,122.055,0.055,0.180,0.008,0.135
training/purelove.wav,ok,105.074,1.074,0.300,0.031,1.384
training/acht.wav,ok,120.000,0.000,8.220,32.522,325.220
training/best.wav,ok,122.031,2.969,0.580,0.472,7.689
training/eureka.wav,ok,120.000,0.000,1.980,1.947,19.470
```

Aggregate (`ok_rows=7`):
- mean bpm error: `0.600`
- max bpm error: `2.969`
- mean downbeat error: `5.429 s`
- max downbeat error: `32.522 s`
- mean score: `54.888`

Notes:
- `--ml-cpu-only` is currently required for stable benchmarking in this environment.
- The same run without CPU-only can fail/crash in CoreML execution on Codex.

### BeatThis preset vs DBN flags (post-fix, 2026-02-10)

Issue:
- `--ml-beatthis` applies a preset that sets `use_dbn=false`.
- CLI DBN flags were parsed before preset application, so `--ml-dbn` could be overridden.

Fix:
- In `src/cli/main.mm`, DBN-related CLI flags are now tracked as explicitly set and reapplied after preset application.
- `--ml-no-dbn` still has final precedence and forces DBN off.

Quick verification:
- `--ml-dbn` on `training/best.wav` now emits DBN trace lines (`DBN window`, `DBN calmdad`, `DBN grid`, etc.).
- `--ml-no-dbn` on the same file emits no DBN trace lines and reports a different BPM path.

Focused sweep (files: `training/acht.wav`, `training/best.wav`, CPU-only):
- Outputs are in `logs/dbn_sweeps_postfix2/`.

Per-track summary:

```csv
variant,track,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,score
baseline,training/acht.wav,120.026,0.026,40.064,0.678,6.806
baseline,training/best.wav,244.057,119.057,0.000,0.049,119.547
window60_best,training/acht.wav,120.026,0.026,40.064,0.678,6.806
window60_best,training/best.wav,244.057,119.057,0.000,0.049,119.547
strict,training/acht.wav,119.950,0.050,40.084,0.658,6.630
strict,training/best.wav,243.995,118.995,0.000,0.049,119.485
loose,training/acht.wav,119.950,0.050,40.084,0.658,6.630
loose,training/best.wav,243.995,118.995,0.000,0.049,119.485
no_dbn,training/acht.wav,120.000,0.000,8.220,32.522,325.220
no_dbn,training/best.wav,122.031,2.969,0.580,0.472,7.689
```

Variant means:

```csv
variant,mean_bpm_err,mean_downbeat_err_s,mean_score
baseline,59.541,0.364,63.176
window60_best,59.541,0.364,63.176
strict,59.523,0.354,63.057
loose,59.523,0.354,63.057
no_dbn,1.484,16.497,166.455
```

Interpretation:
- After the fix, DBN toggles and DBN parameters are no longer invariant under `--ml-beatthis`.
- On this pair, DBN greatly improves downbeat alignment but drives a severe octave/tempo doubling on `training/best.wav` (~244 BPM vs canonical 125 BPM).
- `--ml-no-dbn` avoids the tempo doubling on `training/best.wav` but regresses downbeat alignment on `training/acht.wav`.

### Post-fix sweep on remaining canonical files (2026-02-10)

Files:
- `training/manucho.wav`
- `training/moderat.wav`
- `training/samerano.wav`
- `training/purelove.wav`
- `training/eureka.wav`

Outputs:
- `logs/dbn_sweeps_postfix_rest/baseline.csv`
- `logs/dbn_sweeps_postfix_rest/window60_best.csv`
- `logs/dbn_sweeps_postfix_rest/strict.csv`
- `logs/dbn_sweeps_postfix_rest/loose.csv`
- `logs/dbn_sweeps_postfix_rest/no_dbn.csv`

Variant means:

```csv
variant,mean_bpm_err,mean_downbeat_err_s,mean_score
baseline,0.218,0.101,1.228
window60_best,0.218,0.101,1.228
strict,0.275,0.101,1.285
loose,0.275,0.101,1.285
no_dbn,0.246,1.002,10.262
```

Per-track summary:

```csv
variant,track,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,score
baseline,training/manucho.wav,109.967,0.033,0.000,0.031,0.343
baseline,training/moderat.wav,124.014,0.014,0.000,0.063,0.644
baseline,training/samerano.wav,122.017,0.017,0.004,0.184,1.857
baseline,training/purelove.wav,104.998,0.998,0.104,0.227,3.268
baseline,training/eureka.wav,120.029,0.029,0.000,0.000,0.029
window60_best,training/manucho.wav,109.967,0.033,0.000,0.031,0.343
window60_best,training/moderat.wav,124.014,0.014,0.000,0.063,0.644
window60_best,training/samerano.wav,122.017,0.017,0.004,0.184,1.857
window60_best,training/purelove.wav,104.998,0.998,0.104,0.227,3.268
window60_best,training/eureka.wav,120.029,0.029,0.000,0.000,0.029
strict,training/manucho.wav,109.975,0.025,0.000,0.031,0.335
strict,training/moderat.wav,124.056,0.056,0.000,0.063,0.686
strict,training/samerano.wav,122.220,0.220,0.004,0.184,2.060
strict,training/purelove.wav,104.998,0.998,0.104,0.227,3.268
strict,training/eureka.wav,120.076,0.076,0.000,0.000,0.076
loose,training/manucho.wav,109.975,0.025,0.000,0.031,0.335
loose,training/moderat.wav,124.056,0.056,0.000,0.063,0.686
loose,training/samerano.wav,122.220,0.220,0.004,0.184,2.060
loose,training/purelove.wav,104.998,0.998,0.104,0.227,3.268
loose,training/eureka.wav,120.076,0.076,0.000,0.000,0.076
no_dbn,training/manucho.wav,110.081,0.081,1.640,1.590,15.981
no_dbn,training/moderat.wav,123.982,0.018,1.520,1.432,14.338
no_dbn,training/samerano.wav,122.055,0.055,0.180,0.008,0.135
no_dbn,training/purelove.wav,105.074,1.074,0.300,0.031,1.384
no_dbn,training/eureka.wav,120.000,0.000,1.980,1.947,19.470
```

Interpretation:
- On these five tracks, DBN-enabled variants are stable and very similar (`baseline` ~= `window60_best`; `strict` ~= `loose`).
- Compared with `no_dbn`, DBN dramatically reduces mean downbeat error (`1.002s` to `0.101s`) without introducing the `best.wav` tempo-doubling failure mode (since `best.wav` is not in this subset).

### Consolidated comparison + candidate profile (2026-02-10)

Combined over all 7 canonical tracks (`acht/best` + remaining 5):

```csv
variant,all_n,mean_bpm_err,mean_downbeat_err_s,mean_score,bpm_fail_gt20
baseline,7,17.168,0.176,18.928,1
window60_best,7,17.168,0.176,18.928,1
strict,7,17.203,0.173,18.934,1
loose,7,17.203,0.173,18.934,1
no_dbn,7,0.600,5.429,54.888,0
```

Selection:
- Candidate profile: `baseline` (`--ml-dbn`), because it is tied best on mean score with fewer moving parts than `window60_best`, and strongly outperforms `no_dbn` on downbeat alignment.
- Known risk: one catastrophic BPM outlier (`training/best.wav` tempo doubling) remains.

### Candidate full canonical run (baseline DBN)

Command:

```bash
python3 scripts/dbn_benchmark.py \
  --beatit build/beatit \
  --backend coreml \
  --model coreml_out_latest/BeatThis_small0.mlpackage \
  --pass-args --ml-cpu-only --ml-dbn \
  > logs/dbn_candidate_baseline_full.csv
```

Per-track:

```csv
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,score
training/manucho.wav,109.967,0.033,0.000,0.031,0.343
training/moderat.wav,124.014,0.014,0.000,0.063,0.644
training/samerano.wav,122.017,0.017,0.004,0.184,1.857
training/purelove.wav,104.998,0.998,0.104,0.227,3.268
training/acht.wav,120.026,0.026,40.064,0.678,6.806
training/best.wav,244.057,119.057,0.000,0.049,119.547
training/eureka.wav,120.029,0.029,0.000,0.000,0.029
```

Aggregate (`ok_rows=7`):
- mean bpm error: `17.168`
- mean downbeat error: `0.176 s`
- mean score: `18.928`
- max bpm error: `119.057`
- max downbeat error: `0.678 s`
- bpm failures (`bpm_err > 20`): `1` (`training/best.wav`)
