#!/usr/bin/env python3
#
#  beatthis_onnx_to_coreml.py
#  BeatIt
#
#  Created by Till Toenshoff on 2025-02-14.
#  Copyright © 2025 Till Toenshoff. All rights reserved.
#
import argparse
from pathlib import Path

import coremltools as ct
import onnx


def derive_model_version(out_path: Path) -> str:
    stem = out_path.stem
    if "_" not in stem:
        return "unknown"
    return stem.rsplit("_", 1)[-1] or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BeatThis ONNX model to CoreML (.mlpackage).",
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--out", required=True, help="Output .mlpackage path")
    parser.add_argument(
        "--frames",
        type=int,
        default=1500,
        help="Input frame count for fixed-shape conversion",
    )
    parser.add_argument(
        "--mels",
        type=int,
        default=128,
        help="Input mel bin count",
    )
    parser.add_argument(
        "--author",
        default="CPJKU (Converted for iOS)",
        help="Author label for metadata",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Short description for metadata",
    )
    parser.add_argument(
        "--license",
        default="MIT",
        help="License label for metadata",
    )
    parser.add_argument(
        "--version",
        default="",
        help="Version string for metadata",
    )
    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_version = args.version or derive_model_version(out_path)
    model_description = args.description or f"Beat This! ({model_version}) - Beat Tracking"

    inputs = [
        ct.TensorType(
            name="mel_spectrogram",
            shape=(1, args.frames, args.mels),
        )
    ]
    onnx_input_shape_map = {
        "mel_spectrogram": [1, args.frames, args.mels],
    }

    try:
        from coremltools.converters.onnx import convert as onnx_convert

        mlmodel = onnx_convert(
            onnx_model,
            onnx_coreml_input_shape_map=onnx_input_shape_map,
        )
    except Exception:
        minimum_target = getattr(ct.target, "macOS13", None)
        mlmodel = ct.convert(
            onnx_model,
            source="onnx",
            inputs=inputs,
            minimum_deployment_target=minimum_target,
            compute_units=ct.ComputeUnit.ALL,
        )

    if args.author:
        mlmodel.author = args.author
    if model_description:
        mlmodel.short_description = model_description
    if args.license:
        mlmodel.license = args.license
    if model_version:
        mlmodel.version = model_version

    mlmodel.save(str(out_path))


if __name__ == "__main__":
    main()
