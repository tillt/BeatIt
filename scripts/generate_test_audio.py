#!/usr/bin/env python3

import argparse
import math
import os
import wave


def build_click(sample_rate, click_ms, frequency, amplitude):
    click_samples = max(1, int(sample_rate * click_ms / 1000.0))
    window = []
    for i in range(click_samples):
        phase = 2.0 * math.pi * frequency * (i / sample_rate)
        fade = 1.0 - (i / click_samples)
        value = amplitude * math.sin(phase) * fade
        window.append(value)
    return window


def generate_click_track(out_path,
                         bpm,
                         duration,
                         sample_rate,
                         frequency,
                         click_ms,
                         amplitude,
                         accent_period,
                         accent_offset,
                         accent_multiplier,
                         trim_beats,
                         accent_frequency):
    beat_interval = 60.0 / bpm
    trim_samples = int(trim_beats * beat_interval * sample_rate)
    total_samples = int(duration * sample_rate)
    padded_samples = total_samples + trim_samples
    samples = [0.0] * padded_samples

    base_click = build_click(sample_rate, click_ms, frequency, amplitude)
    accent_freq = accent_frequency if accent_frequency > 0.0 else frequency
    accent_click = build_click(sample_rate, click_ms, accent_freq, amplitude * accent_multiplier)
    base_len = len(base_click)
    accent_len = len(accent_click)

    beat_index = 0
    while True:
        beat_start = int(beat_index * beat_interval * sample_rate)
        if beat_start >= padded_samples:
            break
        if accent_period > 0:
            offset = beat_index - accent_offset
            use_accent = (offset % accent_period == 0)
        else:
            use_accent = False
        click = accent_click if use_accent else base_click
        click_len = accent_len if use_accent else base_len
        for i in range(click_len):
            idx = beat_start + i
            if idx >= padded_samples:
                break
            samples[idx] += click[i]
        beat_index += 1

    if trim_samples > 0:
        samples = samples[trim_samples:trim_samples + total_samples]

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with wave.open(out_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for value in samples:
            clipped = max(-1.0, min(1.0, value))
            int_value = int(clipped * 32767.0)
            wav_file.writeframesraw(int_value.to_bytes(2, byteorder="little", signed=True))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic click-track WAV files.")
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("--bpm", type=float, default=120.0, help="Tempo in BPM")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate in Hz")
    parser.add_argument("--frequency", type=float, default=880.0, help="Click frequency in Hz")
    parser.add_argument("--click-ms", type=float, default=8.0, help="Click length in milliseconds")
    parser.add_argument("--amplitude", type=float, default=0.8, help="Click amplitude (0..1)")
    parser.add_argument("--accent-period", type=int, default=0, help="Accent every N beats (0 disables)")
    parser.add_argument("--accent-offset", type=int, default=0, help="Accent offset in beats")
    parser.add_argument("--accent-multiplier", type=float, default=1.0, help="Accent amplitude multiplier")
    parser.add_argument("--trim-beats", type=float, default=0.0, help="Trim this many beats from the start")
    parser.add_argument("--accent-frequency", type=float, default=0.0, help="Accent frequency in Hz")
    args = parser.parse_args()

    generate_click_track(
        out_path=args.out,
        bpm=args.bpm,
        duration=args.duration,
        sample_rate=args.sample_rate,
        frequency=args.frequency,
        click_ms=args.click_ms,
        amplitude=args.amplitude,
        accent_period=args.accent_period,
        accent_offset=args.accent_offset,
        accent_multiplier=args.accent_multiplier,
        trim_beats=args.trim_beats,
        accent_frequency=args.accent_frequency,
    )


if __name__ == "__main__":
    main()
