import argparse
import sys

import coremltools as ct
import torch
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to BeatTrack .ckpt")
    parser.add_argument("--out", required=True, help="Output .mlpackage path")
    parser.add_argument("--frames", type=int, default=3000, help="Example frame count for tracing")
    parser.add_argument("--origin", default="BeatTrack (mhrice/BeatTrack)",
                        help="Model origin label for metadata")
    parser.add_argument("--author", default="",
                        help="Author label for metadata")
    parser.add_argument("--description", default="",
                        help="Short description for metadata")
    parser.add_argument("--license", default="", help="License label for metadata")
    parser.add_argument("--version", default="", help="Version string for metadata")
    parser.add_argument("--dataset", default="", help="Training dataset label for metadata")
    parser.add_argument("--source", default="",
                        help="Source label for metadata")
    args = parser.parse_args()

    sys.path.append("third_party/BeatTrack")
    if "beattrack.eval" not in sys.modules:
        dummy_eval = types.ModuleType("beattrack.eval")
        def _noop_eval(*_args, **_kwargs):
            raise RuntimeError("beattrack.eval is not available during export")
        dummy_eval.eval = _noop_eval
        sys.modules["beattrack.eval"] = dummy_eval

    from beattrack.model import BeatTCN

    model = BeatTCN()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned[key[len("model."):]] = value
        else:
            cleaned[key] = value
    model.load_state_dict(cleaned, strict=False)
    model._trainer = object()
    model.eval()

    class BeatTrackWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            out = self.base(x)
            beat, downbeat = out.split(1, dim=1)
            return beat.squeeze(1), downbeat.squeeze(1)

    wrapper = BeatTrackWrapper(model)
    example = torch.rand(1, 1, args.frames, 81)
    traced = torch.jit.trace(wrapper, example)

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
