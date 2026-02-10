# DBN Canonical Validation

Run:
```
python3 scripts/dbn_benchmark.py --only training/manucho.wav training/moderat.wav training/samerano.wav training/purelove.wav training/acht.wav training/best.wav training/eureka.wav
```

```
file,bpm_est,bpm_err,downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,first_downbeat_s,first_downbeat_err_s,downbeat_count,first_beat_s,first_beat_err_s,beat_count,first_beat_peak_s,first_beat_peak_err_s,first_beat_floor_s,first_beat_floor_err_s,max_beat_first2s,downbeat_onset_s,downbeat_peak_s,score
training/manucho.wav,109.953,0.047,0.104,0.054,0.073,0.000,0.104,0.054,198,0.104,0.054,792,0.120,0.070,0.120,0.070,nan,0.031,0.05,0.587
training/moderat.wav,124.001,0.001,0.104,0.016,0.041,0.000,0.104,0.016,186,0.104,0.016,742,0.120,0.032,0.120,0.032,nan,0.063,0.088,0.161
training/samerano.wav,121.989,0.011,0.104,0.084,0.084,0.000,0.104,0.084,201,0.104,0.084,804,0.120,0.068,0.120,0.068,nan,0.188,0.251,0.851
training/purelove.wav,105.040,1.040,0.104,0.227,0.227,0.000,0.104,0.227,155,0.104,0.227,463,0.300,0.031,0.120,0.211,nan,0.331,0.6,3.310
training/acht.wav,120.016,0.016,40.099,0.643,0.643,0.000,0.104,40.638,267,0.104,40.638,1065,0.120,40.622,0.120,40.622,nan,40.742,40.786,6.449
training/best.wav,122.006,2.994,0.104,0.000,0.055,0.000,0.104,0.000,291,0.104,0.000,1164,0.140,0.032,0.120,0.012,nan,0.049,0.108,2.994
```
