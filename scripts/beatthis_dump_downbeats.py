#!/usr/bin/env python3
#
#  beatthis_dump_downbeats.py
#  BeatIt
#
#  Created by Till Toenshoff on 2026-01-19.
#  Copyright © 2026 Till Toenshoff. All rights reserved.
#

from __future__ import annotations

import argparse

from beat_this.inference import File2Beats


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump BeatThis downbeat anchors.")
    parser.add_argument("--checkpoint", required=True, help="BeatThis checkpoint path.")
    parser.add_argument("--dbn", action="store_true", help="Enable DBN post-processing.")
    parser.add_argument("--device", default="cpu", help="Device string (default: cpu).")
    parser.add_argument("--sample-rate", type=float, default=44100.0, help="Sample rate for frame conversion.")
    parser.add_argument("paths", nargs="+", help="Audio file paths.")

    args = parser.parse_args()

    file2beats = File2Beats(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dbn=args.dbn,
    )

    for path in args.paths:
        beats, downbeats = file2beats(path)
        if len(downbeats) == 0:
            print(f"{path} downbeat_s=none downbeat_frame=none")
            continue
        first = float(downbeats[0])
        frame = int(round(first * args.sample_rate))
        print(f"{path} downbeat_s={first:.6f} downbeat_frame={frame}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
