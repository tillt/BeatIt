#!/usr/bin/env python3
#
#  beatthis_infer.py
#  BeatIt
#
#  Created by Till Toenshoff on 2026-01-19.
#  Copyright © 2026 Till Toenshoff. All rights reserved.
#

from __future__ import annotations

import argparse

from beat_this.inference import File2Beats


def main() -> int:
    parser = argparse.ArgumentParser(description="BeatThis inference helper.")
    parser.add_argument("--input", required=True, help="Path to input audio file.")
    parser.add_argument("--checkpoint", required=True, help="BeatThis checkpoint path.")
    parser.add_argument("--dbn", action="store_true", help="Enable DBN post-processing.")
    parser.add_argument("--device", default="cpu", help="Device string (default: cpu).")

    args = parser.parse_args()

    file2beats = File2Beats(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dbn=args.dbn,
    )

    beats, downbeats = file2beats(args.input)

    beat_line = "beats " + str(len(beats))
    beat_line += "".join(f" {value:.6f}" for value in beats)
    print(beat_line)

    downbeat_line = "downbeats " + str(len(downbeats))
    downbeat_line += "".join(f" {value:.6f}" for value in downbeats)
    print(downbeat_line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
