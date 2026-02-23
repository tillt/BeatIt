#
#  beatthis_export_torchscript.py
#  BeatIt
#
#  Created by Till Toenshoff on 2026-01-19.
#  Copyright © 2026 Till Toenshoff. All rights reserved.
#

import argparse
import inspect
from pathlib import Path

import torch

from beat_this.model.beat_tracker import BeatThis
from beat_this.utils import replace_state_dict_key


def load_model(checkpoint_path: Path, device: torch.device) -> BeatThis:
    weights_only = {"weights_only": True} if torch.__version__ >= "2" else {}
    checkpoint = torch.load(str(checkpoint_path), map_location=device, **weights_only)
    hparams = checkpoint["hyper_parameters"]
    hparams = {
        key: value
        for key, value in hparams.items()
        if key in set(inspect.signature(BeatThis).parameters)
    }
    model = BeatThis(**hparams)
    state_dict = replace_state_dict_key(checkpoint["state_dict"], "model.", "")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BeatThis to TorchScript.")
    parser.add_argument("--checkpoint", required=True, help="BeatThis checkpoint path.")
    parser.add_argument("--out", required=True, help="Output TorchScript path (.pt).")
    parser.add_argument("--frames", type=int, default=1500, help="Fixed frame length.")
    parser.add_argument("--mels", type=int, default=128, help="Mel bin count.")
    parser.add_argument("--device", default="cpu", help="Torch device for export.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.checkpoint), device)

    class BeatThisWrapper(torch.nn.Module):
        def __init__(self, inner: BeatThis):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = self.inner(x)
            beat = torch.sigmoid(outputs["beat"])
            downbeat = torch.sigmoid(outputs["downbeat"])
            return beat, downbeat

    wrapper = BeatThisWrapper(model).to(device).eval()
    example = torch.zeros(1, args.frames, args.mels, device=device)
    traced = torch.jit.trace(wrapper, example, check_trace=False)
    traced.save(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
