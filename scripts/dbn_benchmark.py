#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CanonicalTrack:
    bpm: float
    onset_s: Optional[float]
    peak_s: Optional[float]


CANONICAL: Dict[str, CanonicalTrack] = {
    "training/manucho.wav": CanonicalTrack(bpm=110.0, onset_s=0.031, peak_s=0.050),
    "training/moderat.wav": CanonicalTrack(bpm=124.0, onset_s=0.063, peak_s=0.088),
    "training/samerano.wav": CanonicalTrack(bpm=122.0, onset_s=0.188, peak_s=0.251),
    "training/purelove.wav": CanonicalTrack(bpm=104.0, onset_s=0.331, peak_s=0.600),
    "training/acht.wav": CanonicalTrack(bpm=120.0, onset_s=40.742, peak_s=40.786),
    "training/best.wav": CanonicalTrack(bpm=125.0, onset_s=0.049, peak_s=0.108),
    "training/eureka.wav": CanonicalTrack(bpm=120.0, onset_s=0.000, peak_s=0.033),
}


RE_ANCHOR = re.compile(
    r"Tempo anchor:\s*peaks=([0-9.]+)\s+autocorr=([0-9.]+)\s+comb=([0-9.]+)\s+beats=([0-9.]+)\s+chosen=([0-9.]+)"
)
RE_PEAK_MEDIAN_BPM = re.compile(r"BPM debug: peak_median_bpm=([0-9.]+)")
RE_PEAK_HIST_TOP = re.compile(r"BPM debug: peak_hist_top=([^\n]+)")
RE_PEAK_HIST_DOM = re.compile(
    r"BPM debug: peak_hist_dom "
    r"top1_bpm=([0-9.]+) top1_w=([0-9.]+) "
    r"top2_bpm=([0-9.]+) top2_w=([0-9.]+) "
    r"top1_ratio=([0-9.]+) top_gap=([0-9.]+) top_range_bpm=([0-9.]+)"
)
RE_COMB_DOM = re.compile(
    r"BPM debug: comb_dom "
    r"top1_bpm=([0-9.]+) top1_w=([0-9.]+) "
    r"top2_bpm=([0-9.]+) top2_w=([0-9.]+) "
    r"top1_ratio=([0-9.]+) top_gap=([0-9.]+)"
)
RE_DBN_QUALITY = re.compile(
    r"DBN quality:\s*qpar=([0-9.eE+-]+)\s+qmax=([0-9.eE+-]+)\s+qkur=([0-9.eE+-]+)"
)
RE_QUALITY_GATE = re.compile(
    r"DBN quality gate:\s*low=(\d+)\s+drop_ref=(\d+)\s+drop_global=(\d+)\s+drop_fit=(\d+)"
    r"\s+downbeat_ok=(\d+)\s+downbeat_cv=([0-9.eE+-]+)\s+downbeat_count=(\d+)"
    r"\s+used=([a-z_]+)\s+pre_override=([a-z_]+)\s+pre_bpm=([0-9.eE+-]+)"
)
BPM_RE = re.compile(r"Estimated BPM:\s*([0-9.]+)")
DOWNBEAT_RE = re.compile(
    r"First downbeat feature_frame:\s*(\d+)\s*\(s ([0-9.]+)\)"
    r"(?:\s*\(sample_frame ([0-9]+)\))?"
)
DOWNBEAT_LIST_RE = re.compile(r"^Downbeats \(all sample frames\):(.*)$", re.M)
DOWNBEAT_SAMPLE_RE = re.compile(r"(\d+)\(([0-9.]+)\)")
BEAT_LIST_RE = re.compile(r"^Beats \(all sample frames\):(.*)$", re.M)
BEAT_SAMPLE_RE = re.compile(r"(\d+)\(([0-9.]+)\)")
FIRST_BEAT_PEAK_RE = re.compile(r"first beat peak: .*sample_frame=(\d+)\s*\(s ([0-9.]+)\)")
FIRST_BEAT_FLOOR_RE = re.compile(r"first beat floor: .*sample_frame=(\d+)\s*\(s ([0-9.]+)\)")
MAX_BEAT_FIRST2S_RE = re.compile(r"max beat activation \\(first 2s\\):\\s*([0-9.]+)")


def run_beatit(beatit: Path,
               backend: str,
               model: Optional[Path],
               wav: str,
               extra_args: List[str]) -> Tuple[str, str, int]:
    cmd = [
        str(beatit),
        "--input",
        wav,
        "--ml-backend",
        backend,
        "--ml-verbose",
    ]

    if backend == "torch":
        if model is None:
            raise ValueError("torch backend requires --model")
        cmd += [
            "--torch-model",
            str(model),
            "--ml-preset",
            "beatthis",
        ]
    elif backend == "coreml":
        cmd += ["--ml-beatthis"]
        if model is not None:
            cmd += ["--model", str(model)]
    elif backend == "beatthis":
        if model is None:
            raise ValueError("beatthis backend requires --model checkpoint")
        cmd += ["--beatthis-checkpoint", str(model)]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    cmd += extra_args
    env = dict(**os.environ)
    env["BEATIT_DEBUG_BPM"] = "1"
    tmpdir = Path("build/coreml_tmp")
    home = Path("build/coreml_home")
    cache = Path("build/coreml_cache")
    tmpdir.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmpdir.resolve())
    env["HOME"] = str(home.resolve())
    env["XDG_CACHE_HOME"] = str(cache.resolve())
    proc = subprocess.run(cmd,
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          env=env,
                          check=False)
    output = proc.stdout or ""
    if proc.returncode == 0:
        return output, "ok", 0
    return output, "error", proc.returncode


def parse_bpm(output: str) -> float:
    match = BPM_RE.search(output)
    return float(match.group(1)) if match else float("nan")


def parse_tempo_anchor(output: str) -> Tuple[float, float, float, float, float]:
    match = RE_ANCHOR.search(output)
    if not match:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    return tuple(float(val) for val in match.groups())


def parse_peak_median_bpm(output: str) -> float:
    match = RE_PEAK_MEDIAN_BPM.search(output)
    return float(match.group(1)) if match else float("nan")


def parse_peak_hist_top(output: str) -> str:
    match = RE_PEAK_HIST_TOP.search(output)
    return match.group(1).strip() if match else ""


def parse_peak_hist_dom(output: str) -> Tuple[float, float, float, float, float, float, float]:
    match = RE_PEAK_HIST_DOM.search(output)
    if not match:
        return (float("nan"), float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"))
    return tuple(float(val) for val in match.groups())


def parse_comb_dom(output: str) -> Tuple[float, float, float, float, float, float]:
    match = RE_COMB_DOM.search(output)
    if not match:
        return (float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"), float("nan"))
    return tuple(float(val) for val in match.groups())


def parse_dbn_quality(output: str) -> Tuple[float, float, float]:
    match = RE_DBN_QUALITY.search(output)
    if not match:
        return (float("nan"), float("nan"), float("nan"))
    return tuple(float(val) for val in match.groups())


def parse_quality_gate(output: str) -> Tuple[int, int, int, int, int, float, int, str, str, float]:
    match = RE_QUALITY_GATE.search(output)
    if not match:
        return (0, 0, 0, 0, 0, float("nan"), 0, "", "", float("nan"))
    (
        low,
        drop_ref,
        drop_global,
        drop_fit,
        downbeat_ok,
        downbeat_cv,
        downbeat_count,
        used,
        pre_override,
        pre_bpm,
    ) = match.groups()
    return (int(low),
            int(drop_ref),
            int(drop_global),
            int(drop_fit),
            int(downbeat_ok),
            float(downbeat_cv),
            int(downbeat_count),
            used,
            pre_override,
            float(pre_bpm))

def parse_first_downbeat_s(output: str, sample_rate: float) -> float:
    match = DOWNBEAT_RE.search(output)
    if not match:
        return float("nan")
    sample_frame = match.group(3)
    if sample_frame is not None:
        return float(sample_frame) / sample_rate
    return float(match.group(2))


def parse_downbeat_list_s(output: str) -> List[float]:
    match = DOWNBEAT_LIST_RE.search(output)
    if not match:
        return []
    line = match.group(1)
    return [float(seconds) for _, seconds in DOWNBEAT_SAMPLE_RE.findall(line)]


def parse_beat_list_s(output: str) -> List[float]:
    match = BEAT_LIST_RE.search(output)
    if not match:
        return []
    line = match.group(1)
    return [float(seconds) for _, seconds in BEAT_SAMPLE_RE.findall(line)]


def parse_first_beat_peak_s(output: str) -> float:
    match = FIRST_BEAT_PEAK_RE.search(output)
    if not match:
        return float("nan")
    return float(match.group(2))


def parse_first_beat_floor_s(output: str) -> float:
    match = FIRST_BEAT_FLOOR_RE.search(output)
    if not match:
        return float("nan")
    return float(match.group(2))


def parse_max_beat_first2s(output: str) -> float:
    match = MAX_BEAT_FIRST2S_RE.search(output)
    if not match:
        return float("nan")
    return float(match.group(1))


def median_bpm_from_events(event_times: List[float]) -> float:
    if len(event_times) < 2:
        return float("nan")
    intervals = [
        event_times[i] - event_times[i - 1]
        for i in range(1, len(event_times))
        if event_times[i] > event_times[i - 1]
    ]
    if not intervals:
        return float("nan")
    intervals.sort()
    median = intervals[len(intervals) // 2]
    return 60.0 / median if median > 0.0 else float("nan")

def recommend_bpm_rule(peaks: float,
                       autocorr: float,
                       beats: float,
                       top1_ratio: float,
                       top_gap: float,
                       top_range_bpm: float) -> str:
    if peaks != peaks or beats != beats:
        return "insufficient"
    if top1_ratio >= 0.55 and top_gap >= 200.0:
        return "peaks_dom"
    if top1_ratio < 0.40 or top_range_bpm >= 4.0:
        if autocorr == autocorr:
            return "autocorr"
        return "peaks_fallback"
    if abs(peaks - beats) <= 0.5:
        return "peaks_consensus"
    if autocorr == autocorr and abs(autocorr - beats) <= 0.5:
        return "autocorr_consensus"
    return "median"


def recommend_bpm_value(peaks: float, autocorr: float, beats: float, rule: str) -> float:
    if rule == "peaks_dom":
        return peaks
    if rule in ("autocorr", "autocorr_consensus"):
        return autocorr
    if rule in ("peaks_consensus", "peaks_fallback"):
        return peaks
    if rule == "median":
        values = [v for v in (peaks, autocorr, beats) if v == v]
        values.sort()
        return values[len(values) // 2] if values else float("nan")
    return peaks

def downbeat_window(track: CanonicalTrack) -> Optional[Tuple[float, float]]:
    if track.onset_s is None and track.peak_s is None:
        return None
    if track.onset_s is None:
        return (track.peak_s, track.peak_s)  # type: ignore[arg-type]
    if track.peak_s is None:
        return (track.onset_s, track.onset_s)
    return (min(track.onset_s, track.peak_s), max(track.onset_s, track.peak_s))


def downbeat_err_s(downbeat_est: float, track: CanonicalTrack) -> float:
    window = downbeat_window(track)
    if window is None:
        return float("nan")
    lo, hi = window
    if lo <= downbeat_est <= hi:
        return 0.0
    return min(abs(downbeat_est - lo), abs(downbeat_est - hi))


def score(bpm_err: float, downbeat_err: float) -> float:
    if downbeat_err != downbeat_err:  # NaN check
        return bpm_err
    return bpm_err + 10.0 * downbeat_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beatit", default="build/beatit")
    parser.add_argument("--backend", default="coreml", choices=["coreml", "torch", "beatthis"])
    parser.add_argument("--model", default="")
    parser.add_argument("--sample-rate", type=float, default=44100.0)
    parser.add_argument("--flags", nargs="*", default=[])
    parser.add_argument("--pass-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--no-dbn", action="store_true")
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    beatit = Path(args.beatit)
    model = Path(args.model) if args.model else None

    print(
        "file,bpm_est,bpm_err,anchor_peaks,anchor_autocorr,anchor_comb,anchor_beats,anchor_chosen,"
        "peak_median_bpm,peak_hist_top,peak_top1_bpm,peak_top1_w,peak_top2_bpm,peak_top2_w,"
        "peak_top1_ratio,peak_top_gap,peak_top_range_bpm,beat_median_bpm,"
        "comb_top1_bpm,comb_top1_w,comb_top2_bpm,comb_top2_w,comb_top1_ratio,comb_top_gap,"
        "rule_suggest,rule_bpm,rule_delta_bpm,"
        "dbn_qpar,dbn_qmax,dbn_qkur,"
        "dbn_gate_low,dbn_gate_drop_ref,dbn_gate_drop_global,dbn_gate_drop_fit,"
        "dbn_gate_downbeat_ok,dbn_gate_downbeat_cv,dbn_gate_downbeat_count,"
        "dbn_gate_used,dbn_gate_pre_override,dbn_gate_pre_bpm,dbn_gate_post_bpm,dbn_gate_override,"
        "downbeat_est_s,downbeat_err_s,downbeat_phase_err_s,downbeat_bar_offset_s,"
        "first_downbeat_s,first_downbeat_err_s,downbeat_count,"
        "first_beat_s,first_beat_err_s,beat_count,"
        "first_beat_peak_s,first_beat_peak_err_s,"
        "first_beat_floor_s,first_beat_floor_err_s,"
        "max_beat_first2s,"
        "run_status,run_code,"
        "downbeat_onset_s,downbeat_peak_s,score"
    )
    extra_args = [arg for arg in (args.flags + args.pass_args) if arg != "--"]
    if "--ml-verbose" not in extra_args:
        extra_args.append("--ml-verbose")
    if not args.no_dbn:
        extra_args += ["--ml-dbn"]
    else:
        extra_args += ["--ml-no-dbn"]
    only = set(args.only)
    for wav, canon in CANONICAL.items():
        if only and wav not in only:
            continue
        output, run_status, run_code = run_beatit(beatit, args.backend, model, wav, extra_args)
        bpm_est = parse_bpm(output)
        anchor_peaks, anchor_autocorr, anchor_comb, anchor_beats, anchor_chosen = parse_tempo_anchor(output)
        peak_median_bpm = parse_peak_median_bpm(output)
        peak_hist_top = parse_peak_hist_top(output)
        (
            peak_top1_bpm,
            peak_top1_w,
            peak_top2_bpm,
            peak_top2_w,
            peak_top1_ratio,
            peak_top_gap,
            peak_top_range_bpm,
        ) = parse_peak_hist_dom(output)
        (
            comb_top1_bpm,
            comb_top1_w,
            comb_top2_bpm,
            comb_top2_w,
            comb_top1_ratio,
            comb_top_gap,
        ) = parse_comb_dom(output)
        dbn_qpar, dbn_qmax, dbn_qkur = parse_dbn_quality(output)
        (
            dbn_gate_low,
            dbn_drop_ref,
            dbn_drop_global,
            dbn_drop_fit,
            dbn_gate_downbeat_ok,
            dbn_gate_downbeat_cv,
            dbn_gate_downbeat_count,
            dbn_gate_used,
            dbn_gate_pre_override,
            dbn_gate_pre_bpm,
        ) = parse_quality_gate(output)
        dbn_gate_post_bpm = bpm_est
        dbn_gate_override = 1 if dbn_gate_used == "downbeats_override" else 0
        downbeat_candidates = parse_downbeat_list_s(output)
        first_downbeat_s = float("nan")
        first_downbeat_err = float("nan")
        downbeat_count = len(downbeat_candidates)
        beat_candidates = parse_beat_list_s(output)
        beat_median_bpm = median_bpm_from_events(beat_candidates)
        first_beat_s = float("nan")
        first_beat_err = float("nan")
        beat_count = len(beat_candidates)
        if beat_candidates:
            first_beat_s = beat_candidates[0]
            first_beat_err = downbeat_err_s(first_beat_s, canon)
        if downbeat_candidates:
            first_downbeat_s = downbeat_candidates[0]
            first_downbeat_err = downbeat_err_s(first_downbeat_s, canon)
            window = downbeat_window(canon)
            if window is None:
                downbeat_est = first_downbeat_s
            else:
                lo, hi = window
                downbeat_est = min(
                    downbeat_candidates,
                    key=lambda s: 0.0 if lo <= s <= hi else min(abs(s - lo), abs(s - hi)),
                )
        else:
            downbeat_est = parse_first_downbeat_s(output, args.sample_rate)
            first_downbeat_s = downbeat_est
            first_downbeat_err = downbeat_err_s(first_downbeat_s, canon)
            downbeat_count = 0

        bpm_err = abs(bpm_est - canon.bpm)
        downbeat_err = downbeat_err_s(downbeat_est, canon)
        total = score(bpm_err, downbeat_err)

        rule = recommend_bpm_rule(
            anchor_peaks,
            anchor_autocorr,
            anchor_beats,
            peak_top1_ratio,
            peak_top_gap,
            peak_top_range_bpm,
        )
        rule_bpm = recommend_bpm_value(anchor_peaks, anchor_autocorr, anchor_beats, rule)
        rule_delta = rule_bpm - canon.bpm if rule_bpm == rule_bpm else float("nan")

        bar_len_s = (240.0 / bpm_est) if bpm_est > 0.0 else float("nan")
        if bar_len_s == bar_len_s and bar_len_s > 0.0 and canon.onset_s is not None:
            delta_s = downbeat_est - canon.onset_s
            downbeat_bar_offset_s = round(delta_s / bar_len_s) * bar_len_s
            downbeat_phase_err_s = abs(delta_s - downbeat_bar_offset_s)
        else:
            downbeat_phase_err_s = float("nan")
            downbeat_bar_offset_s = float("nan")

        first_peak_s = parse_first_beat_peak_s(output)
        first_peak_err = downbeat_err_s(first_peak_s, canon) if first_peak_s == first_peak_s else float("nan")
        first_floor_s = parse_first_beat_floor_s(output)
        first_floor_err = downbeat_err_s(first_floor_s, canon) if first_floor_s == first_floor_s else float("nan")
        max_beat_first2s = parse_max_beat_first2s(output)

        print(
            f"{wav},{bpm_est:.3f},{bpm_err:.3f},"
            f"{anchor_peaks:.3f},{anchor_autocorr:.3f},{anchor_comb:.3f},{anchor_beats:.3f},{anchor_chosen:.3f},"
            f"{peak_median_bpm:.3f},{peak_hist_top},"
            f"{peak_top1_bpm:.3f},{peak_top1_w:.3f},{peak_top2_bpm:.3f},{peak_top2_w:.3f},"
            f"{peak_top1_ratio:.3f},{peak_top_gap:.3f},{peak_top_range_bpm:.3f},"
            f"{beat_median_bpm:.3f},"
            f"{comb_top1_bpm:.3f},{comb_top1_w:.3f},{comb_top2_bpm:.3f},{comb_top2_w:.3f},"
            f"{comb_top1_ratio:.3f},{comb_top_gap:.3f},"
            f"{rule},{rule_bpm:.3f},{rule_delta:.3f},"
            f"{dbn_qpar:.6f},{dbn_qmax:.6f},{dbn_qkur:.6f},"
            f"{dbn_gate_low},{dbn_drop_ref},{dbn_drop_global},{dbn_drop_fit},"
            f"{dbn_gate_downbeat_ok},{dbn_gate_downbeat_cv:.6f},{dbn_gate_downbeat_count},"
            f"{dbn_gate_used},{dbn_gate_pre_override},{dbn_gate_pre_bpm:.3f},"
            f"{dbn_gate_post_bpm:.3f},{dbn_gate_override},"
            f"{downbeat_est:.3f},{downbeat_err:.3f},"
            f"{downbeat_phase_err_s:.3f},{downbeat_bar_offset_s:.3f},"
            f"{first_downbeat_s:.3f},{first_downbeat_err:.3f},{downbeat_count},"
            f"{first_beat_s:.3f},{first_beat_err:.3f},{beat_count},"
            f"{first_peak_s:.3f},{first_peak_err:.3f},"
            f"{first_floor_s:.3f},{first_floor_err:.3f},"
            f"{max_beat_first2s:.5f},"
            f"{run_status},{run_code},"
            f"{canon.onset_s if canon.onset_s is not None else 'nan'},"
            f"{canon.peak_s if canon.peak_s is not None else 'nan'},"
            f"{total:.3f}"
        )


if __name__ == "__main__":
    main()
