# Models

Place the compiled CoreML model here:

- `beatit.mlmodelc`

You can train and export a model using the scripts in `training/`.


Torch export:

```bash
python3 -m venv .venv-beatit-export
source .venv-beatit-export/bin/activate
pip install torch pyyaml numpy einops rotary-embedding-torch

PYTHONPATH=third_party/beat_this \
python scripts/beatthis_export_torchscript.py \
  --checkpoint models/beat_this-final0.ckpt \
  --out models/beatthis.pt \
  --device cpu
```

Notes:

- `models/beatthis.pt` is exported from `models/beat_this-final0.ckpt`.
- The export script patches rotary tracing so the TorchScript graph does not hardcode CPU device creation and can run on MPS.
- Export on CPU. The resulting `models/beatthis.pt` is then usable with `--torch-device mps`.
