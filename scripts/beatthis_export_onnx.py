#!/usr/bin/env python3
#
#  beatthis_export_onnx.py
#  BeatIt
#
#  Created by Till Toenshoff on 2025-02-14.
#  Copyright © 2025 Till Toenshoff. All rights reserved.
#
import argparse
import inspect
from pathlib import Path

import torch

from beat_this.model.beat_tracker import BeatThis
from beat_this.utils import replace_state_dict_key


def load_beatthis_model(checkpoint_path: Path, device: torch.device) -> BeatThis:
    weights_only = {"weights_only": True} if torch.__version__ >= "2" else {}
    checkpoint = torch.load(checkpoint_path, map_location=device, **weights_only)

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


def export_onnx(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    model = load_beatthis_model(Path(args.checkpoint), device)

    dummy_input = torch.randn(1, args.frames, args.mels, device=device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["mel_spectrogram"],
        output_names=["beat_logits", "downbeat_logits"],
        dynamic_axes={
            "mel_spectrogram": {1: "time"},
            "beat_logits": {1: "time"},
            "downbeat_logits": {1: "time"},
        },
    )

    try:
        import onnx

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export BeatThis checkpoint to ONNX.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--out", required=True, help="Output .onnx path")
    parser.add_argument("--frames", type=int, default=1500, help="Input frame count")
    parser.add_argument("--mels", type=int, default=128, help="Mel bin count")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    export_onnx(args)


if __name__ == "__main__":
    main()
