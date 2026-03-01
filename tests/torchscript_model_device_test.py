#!/usr/bin/env python3
#
#  torchscript_model_device_test.py
#  BeatIt
#
#  Created by Till Toenshoff on 2026-03-01.
#  Copyright © 2026 Till Toenshoff. All rights reserved.
#

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    if importlib.util.find_spec("torch") is None:
        return 77

    import torch

    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/beatthis.pt")
    if not model_path.exists():
        print(f"TorchScript device test failed: missing model '{model_path}'.")
        return 1

    module = torch.jit.load(str(model_path), map_location="cpu")
    offenders: list[str] = []

    for name, child in module.named_modules():
        code = getattr(child, "code", None)
        if code is None:
            continue
        if 'device=torch.device("cpu")' in str(code):
            offenders.append(name or "<root>")

    if offenders:
        print(
            "TorchScript device test failed: exported model still hardcodes CPU "
            f"device in {', '.join(offenders)}."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
