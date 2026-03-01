#
#  beatthis_export.py
#  BeatIt
#
#  Created by Till Toenshoff on 2026-01-18.
#  Copyright © 2026 Till Toenshoff. All rights reserved.
#

import argparse
import inspect
import sys

import coremltools as ct
import torch


def load_checkpoint(path: str, device: str) -> dict:
    weights_only = {"weights_only": True} if torch.__version__ >= "2" else {}
    return torch.load(path, map_location=device, **weights_only)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to Beat This! .ckpt")
    parser.add_argument("--out", required=True, help="Output .mlpackage path")
    parser.add_argument("--frames", type=int, default=1500, help="Example frame count for tracing")
    parser.add_argument("--mels", type=int, default=0, help="Override mel bin count (0 = infer)")
    parser.add_argument("--origin", default="Beat This! (CPJKU/beat_this)",
                        help="Model origin label for metadata")
    parser.add_argument("--author", default="", help="Author label for metadata")
    parser.add_argument("--description", default="", help="Short description for metadata")
    parser.add_argument("--license", default="", help="License label for metadata")
    parser.add_argument("--version", default="", help="Version string for metadata")
    parser.add_argument("--dataset", default="", help="Training dataset label for metadata")
    parser.add_argument("--source", default="", help="Source label for metadata")
    args = parser.parse_args()

    sys.path.append("third_party/beat_this")

    from beat_this.model.beat_tracker import BeatThis
    from beat_this.utils import replace_state_dict_key

    checkpoint = load_checkpoint(args.checkpoint, "cpu")
    hparams = checkpoint.get("hyper_parameters", {})
    hparams = {
        key: value
        for key, value in hparams.items()
        if key in set(inspect.signature(BeatThis).parameters)
    }

    model = BeatThis(**hparams)
    state_dict = replace_state_dict_key(checkpoint["state_dict"], "model.", "")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    spect_dim = args.mels if args.mels > 0 else int(hparams.get("spect_dim", 128))
    example = torch.rand(1, args.frames, spect_dim)

    class BeatThisWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            out = self.base(x)
            return out["beat"], out["downbeat"]

    wrapper = BeatThisWrapper(model)
    traced = torch.jit.trace(wrapper, example, check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        outputs=[ct.TensorType(name="beat"), ct.TensorType(name="downbeat")],
    )

    if args.author:
        mlmodel.author = args.author
    if args.description:
        mlmodel.short_description = args.description
    if args.license:
        mlmodel.license = args.license
    if args.version:
        mlmodel.version = args.version

    user_defined = {"beatit:model_origin": args.origin}
    if args.dataset:
        user_defined["beatit:training_dataset"] = args.dataset
    if args.source:
        user_defined["beatit:source"] = args.source
    mlmodel.user_defined_metadata = user_defined

    mlmodel.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
