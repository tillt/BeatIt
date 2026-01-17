import argparse
import pathlib

import coremltools as ct
import torch

from model import BeatNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--origin", default="BeatNet", help="Model origin label for metadata")
    parser.add_argument("--license", default="MIT", help="License label for metadata")
    args = parser.parse_args()

    model = BeatNet()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    example = torch.rand(1, 1000, 64)
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        outputs=[ct.TensorType(name="beat"), ct.TensorType(name="downbeat")],
    )

    mlmodel.author = "Till Toenshoff"
    mlmodel.short_description = "BeatNet model export for BeatIt"
    mlmodel.license = args.license
    mlmodel.version = "1"
    mlmodel.user_defined_metadata = {
        "beatit:model_origin": args.origin,
        "beatit:training_dataset": "Unknown",
        "beatit:source": "BeatNet export_coreml.py",
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
