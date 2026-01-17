# BeatTrack integration (MIT)

This project can consume a CoreML model exported from the MIT-licensed
"TCN-based Joint Beat and Downbeat Tracking" codebase:

- https://github.com/mhrice/BeatTrack

## Recommended flow

1. Clone the BeatTrack repo separately and follow its training instructions
   using Ballroom.
2. Export the trained model to CoreML using `training/beattrack_export.py`.
3. Compile the model and place it in `models/beatit.mlmodelc`.

Example export:

```bash
python3 training/beattrack_export.py --checkpoint third_party/BeatTrack/checkpoints/best.ckpt --out models/beatit.mlpackage
xcrun coremlc compile models/beatit.mlpackage models
```

## Matching inputs/outputs

Use the CLI `--ml-*` flags to match the BeatTrack CoreML interface:

- `--ml-input` for the feature input name
- `--ml-beat` and `--ml-downbeat` for output names
- `--ml-sr`, `--ml-frame`, `--ml-hop`, `--ml-mels` for feature settings

The default CoreML settings here are:

- 16 kHz mono
- frame size 512
- hop size 160
- 64 mel bins

If BeatTrack's model expects different values, pass them explicitly.

## BeatTrack defaults

The BeatTrack model expects:

- 44.1 kHz mono
- frame size 2048 (n_fft)
- hop size 441
- 81 mel bins
- mel (not log-mel)
- input layout `[1, 1, frames, bins]`

You can pass `--ml-beattrack` to the CLI to use these defaults.
