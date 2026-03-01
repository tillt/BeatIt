# Training (Ballroom)

This folder contains a lightweight training/export pipeline for a beat-only model
trained on the Ballroom dataset. The dataset itself is not included.

## Dataset expectations

Ballroom is typically distributed with audio files and beat annotation files.
This pipeline expects a directory with:

- Audio: `*.wav` (or other formats that `librosa` can read)
- Annotations: `*.beats` text files containing beat times in seconds

Directory layout example:

```
ballroom/
  audio/
    0001.wav
    0002.wav
  annotations/
    0001.beats
    0002.beats
```

## Install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Train

```bash
python3 scripts/train.py --data /path/to/ballroom --epochs 20 --batch-size 8 --out training/outputs
```

## Export to CoreML

```bash
python3 scripts/export_coreml.py --checkpoint training/outputs/beat_model.pt --out models/beatit.mlmodel
```

Then compile the model:

```bash
xcrun coremlc compile models/beatit.mlmodel models
```

This produces `models/beatit.mlmodelc`, which the CLI expects by default.

## Notes

- The model outputs only a beat activation. Downbeat is faked by assigning the
  first detected beat as a downbeat at inference time.
- The model input is a log-mel spectrogram at 16 kHz, frame size 512, hop 160,
  mel bins 64. The CoreML wrapper matches these settings.
