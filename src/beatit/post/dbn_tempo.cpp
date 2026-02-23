//
//  dbn_tempo.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "dbn_tempo.h"

#include "beatit/post/tempo_fit.h"
#include "beatit/post/window.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace beatit::detail {

GridTempoDecision compute_grid_tempo_decision(const GridTempoDecisionInput& input) {
    GridTempoDecision decision;
    auto& diag = decision.diagnostics;
    diag.quality_qpar = input.quality_qpar;
    diag.quality_qkur = input.quality_qkur;
    if (input.decoded.beat_frames.empty()) {
        return decision;
    }

    const double min_interval_frames =
        (input.max_bpm > 1.0f && input.fps > 0.0) ? (60.0 * input.fps) / input.max_bpm : 0.0;
    const double short_interval_threshold =
        (min_interval_frames > 0.0) ? std::max(1.0, min_interval_frames * 0.5) : 0.0;
    diag.min_interval_frames = min_interval_frames;
    diag.short_interval_threshold = short_interval_threshold;
    const std::vector<std::size_t> filtered_beats =
        filter_short_intervals(input.decoded.beat_frames, short_interval_threshold);
    const std::vector<std::size_t> aligned_downbeats =
        align_downbeats_to_beats(filtered_beats, input.decoded.downbeat_frames);

    std::vector<std::size_t> bpb_candidates;
    if (input.config.dbn_mode == CoreMLConfig::DBNMode::Calmdad) {
        bpb_candidates = {3, 4};
    } else {
        bpb_candidates = {input.config.dbn_beats_per_bar};
    }
    const auto inferred_bpb_phase =
        infer_bpb_phase(filtered_beats, aligned_downbeats, bpb_candidates, input.config);
    decision.bpb = inferred_bpb_phase.first;
    decision.phase = inferred_bpb_phase.second;
    decision.base_interval = median_interval_frames(filtered_beats);

    const std::vector<float>& tempo_activation =
        input.use_window ? input.beat_slice : input.result.beat_activation;
    const float tempo_threshold =
        std::max(input.config.dbn_activation_floor, input.config.activation_threshold * 0.5f);
    const std::size_t tempo_min_interval =
        static_cast<std::size_t>(
            std::max(1.0, std::floor((60.0 * input.fps) / std::max(1.0f, input.max_bpm))));
    const std::size_t tempo_max_interval = static_cast<std::size_t>(
        std::max<double>(tempo_min_interval,
                         std::ceil((60.0 * input.fps) / std::max(1.0f, input.min_bpm))));
    const std::vector<std::size_t> tempo_peaks =
        pick_peaks(tempo_activation, tempo_threshold, tempo_min_interval, tempo_max_interval);
    const std::vector<std::size_t> tempo_peaks_full =
        input.use_window ? pick_peaks(input.result.beat_activation,
                                      tempo_threshold,
                                      tempo_min_interval,
                                      tempo_max_interval)
                         : tempo_peaks;
    double bpm_from_peaks = 0.0;
    double bpm_from_peaks_median = 0.0;
    double bpm_from_peaks_reg = 0.0;
    double bpm_from_peaks_median_full = 0.0;
    double bpm_from_peaks_reg_full = 0.0;
    if (tempo_peaks.size() >= 2) {
        const double interval_median =
            median_interval_frames_interpolated(tempo_activation, tempo_peaks);
        const double interval_reg =
            regression_interval_frames_interpolated(tempo_activation, tempo_peaks);
        if (interval_median > 0.0) {
            bpm_from_peaks_median = (60.0 * input.fps) / interval_median;
            bpm_from_peaks = bpm_from_peaks_median;
        }
        if (interval_reg > 0.0) {
            bpm_from_peaks_reg = (60.0 * input.fps) / interval_reg;
            if (bpm_from_peaks_median > 0.0) {
                const double ratio =
                    std::abs(bpm_from_peaks_reg - bpm_from_peaks_median) / bpm_from_peaks_median;
                if (ratio <= 0.02) {
                    bpm_from_peaks = bpm_from_peaks_reg;
                }
            } else {
                bpm_from_peaks = bpm_from_peaks_reg;
            }
        }
    }
    if (tempo_peaks_full.size() >= 2) {
        const double interval_median =
            median_interval_frames_interpolated(input.result.beat_activation, tempo_peaks_full);
        const double interval_reg =
            regression_interval_frames_interpolated(input.result.beat_activation, tempo_peaks_full);
        if (interval_median > 0.0) {
            bpm_from_peaks_median_full = (60.0 * input.fps) / interval_median;
        }
        if (interval_reg > 0.0) {
            bpm_from_peaks_reg_full = (60.0 * input.fps) / interval_reg;
        }
    }
    double bpm_from_downbeats = 0.0;
    double bpm_from_downbeats_median = 0.0;
    double bpm_from_downbeats_reg = 0.0;
    std::vector<std::size_t> downbeat_peaks;
    IntervalStats downbeat_stats;
    if (!input.result.downbeat_activation.empty() && decision.bpb > 0) {
        const std::vector<float>& downbeat_activation =
            input.use_window ? input.downbeat_slice : input.result.downbeat_activation;
        const float downbeat_min_bpm =
            std::max(1.0f, input.min_bpm / static_cast<float>(decision.bpb));
        const float downbeat_max_bpm =
            std::max(downbeat_min_bpm + 1.0f, input.max_bpm / static_cast<float>(decision.bpb));
        const std::size_t downbeat_min_interval = static_cast<std::size_t>(
            std::max(1.0, std::floor((60.0 * input.fps) / downbeat_max_bpm)));
        const std::size_t downbeat_max_interval = static_cast<std::size_t>(
            std::max<double>(downbeat_min_interval,
                             std::ceil((60.0 * input.fps) / downbeat_min_bpm)));
        downbeat_peaks = pick_peaks(downbeat_activation,
                                    tempo_threshold,
                                    downbeat_min_interval,
                                    downbeat_max_interval);
        if (downbeat_peaks.size() >= 2) {
            const double interval_median =
                median_interval_frames_interpolated(downbeat_activation, downbeat_peaks);
            const double interval_reg =
                regression_interval_frames_interpolated(downbeat_activation, downbeat_peaks);
            if (interval_median > 0.0) {
                const double downbeat_bpm = (60.0 * input.fps) / interval_median;
                bpm_from_downbeats_median = downbeat_bpm * static_cast<double>(decision.bpb);
                bpm_from_downbeats = bpm_from_downbeats_median;
            }
            if (interval_reg > 0.0) {
                const double downbeat_bpm = (60.0 * input.fps) / interval_reg;
                bpm_from_downbeats_reg = downbeat_bpm * static_cast<double>(decision.bpb);
                if (bpm_from_downbeats_median > 0.0) {
                    const double ratio = std::abs(bpm_from_downbeats_reg - bpm_from_downbeats_median) /
                        bpm_from_downbeats_median;
                    if (ratio <= 0.02) {
                        bpm_from_downbeats = bpm_from_downbeats_reg;
                    }
                } else {
                    bpm_from_downbeats = bpm_from_downbeats_reg;
                }
            }
        }
    }
    if (!downbeat_peaks.empty()) {
        downbeat_stats = interval_stats_interpolated(input.use_window ? input.downbeat_slice
                                                                      : input.result.downbeat_activation,
                                                     downbeat_peaks,
                                                     input.fps,
                                                     0.2);
        diag.has_downbeat_stats = true;
    }
    if (input.config.dbn_trace) {
        diag.stats_computed = true;
        diag.tempo_stats = interval_stats_interpolated(tempo_activation, tempo_peaks, input.fps, 0.2);
        diag.decoded_stats = interval_stats_frames(input.decoded.beat_frames, input.fps, 0.2);
        diag.decoded_filtered_stats = interval_stats_frames(filtered_beats, input.fps, 0.2);
        if (diag.has_downbeat_stats) {
            diag.downbeat_stats = downbeat_stats;
        }
    }

    const double bpm_from_fit = bpm_from_linear_fit(filtered_beats, input.fps);
    const double bpm_from_global_fit = detail::bpm_from_global_fit(input.result,
                                                                   input.config,
                                                                   input.calmdad_decoder,
                                                                   input.fps,
                                                                   input.min_bpm,
                                                                   input.max_bpm,
                                                                   input.used_frames);
    diag.bpm_from_fit = bpm_from_fit;
    diag.bpm_from_global_fit = bpm_from_global_fit;
    decision.quality_low = input.quality_valid && (input.quality_qkur < 3.6);
    const bool drop_fit = decision.quality_low && bpm_from_fit > 0.0;
    diag.drop_fit = drop_fit;
    const std::size_t downbeat_count = downbeat_stats.count;
    const double downbeat_cv = (downbeat_count > 0 && downbeat_stats.mean_interval > 0.0)
        ? (downbeat_stats.stdev_interval / downbeat_stats.mean_interval)
        : 0.0;
    diag.downbeat_count = downbeat_count;
    diag.downbeat_cv = downbeat_cv;
    decision.downbeat_override_ok = !decision.quality_low && downbeat_count >= 6 && downbeat_cv <= 0.25;
    const double ref_downbeat_ratio =
        (decision.downbeat_override_ok && bpm_from_downbeats > 0.0)
            ? (std::abs(input.reference_bpm - bpm_from_downbeats) / bpm_from_downbeats)
            : 0.0;
    const bool ref_mismatch =
        decision.downbeat_override_ok && bpm_from_downbeats > 0.0 && ref_downbeat_ratio > 0.005;
    const bool drop_ref = (decision.quality_low || ref_mismatch) && input.reference_bpm > 0.0f;
    diag.drop_ref = drop_ref;
    const bool allow_reference_grid_bpm =
        input.reference_bpm > 0.0f &&
        ((static_cast<double>(input.max_bpm) - static_cast<double>(input.min_bpm)) <=
         std::max(2.0, static_cast<double>(input.reference_bpm) * 0.05));
    bool global_fit_plausible = false;
    if (bpm_from_global_fit > 0.0 && bpm_from_fit > 0.0) {
        const double diff = std::abs(bpm_from_global_fit - bpm_from_fit);
        const double rel_diff = diff / bpm_from_fit;
        global_fit_plausible = rel_diff <= 0.08;
    }
    diag.global_fit_plausible = global_fit_plausible;
    std::string bpm_source = "none";
    if (global_fit_plausible) {
        decision.bpm_for_grid = bpm_from_global_fit;
        bpm_source = "global_fit";
    } else if (allow_reference_grid_bpm && !decision.quality_low && !ref_mismatch) {
        decision.bpm_for_grid = input.reference_bpm;
        bpm_source = "reference";
    } else if (decision.downbeat_override_ok && bpm_from_downbeats > 0.0) {
        if (bpm_from_fit > 0.0) {
            decision.bpm_for_grid = bpm_from_fit;
            bpm_source = "fit_primary";
        } else {
            decision.bpm_for_grid = bpm_from_downbeats;
            bpm_source = "downbeats_primary";
        }
    } else if (!decision.quality_low && bpm_from_peaks_reg_full > 0.0) {
        decision.bpm_for_grid = bpm_from_peaks_reg_full;
        bpm_source = "peaks_reg_full";
    } else if (!decision.quality_low && bpm_from_fit > 0.0) {
        decision.bpm_for_grid = bpm_from_fit;
        bpm_source = "fit";
    } else if (bpm_from_peaks_median > 0.0) {
        decision.bpm_for_grid = bpm_from_peaks_median;
        bpm_source = "peaks_median";
    } else if (bpm_from_peaks > 0.0) {
        decision.bpm_for_grid = bpm_from_peaks;
        bpm_source = "peaks";
    }
    if (decision.bpm_for_grid <= 0.0 && allow_reference_grid_bpm) {
        decision.bpm_for_grid = input.reference_bpm;
        bpm_source = "reference_fallback";
    }
    const double bpm_before_downbeat = decision.bpm_for_grid;
    const std::string bpm_source_before_downbeat = bpm_source;
    diag.bpm_before_downbeat = bpm_before_downbeat;
    diag.bpm_source_before_downbeat = bpm_source_before_downbeat;
    if (decision.downbeat_override_ok && bpm_from_downbeats > 0.0 && decision.bpm_for_grid > 0.0 &&
        bpm_source != "peaks_reg_full" && bpm_source != "downbeats_primary" &&
        bpm_source != "fit_primary") {
        const double ratio = std::abs(bpm_from_downbeats - decision.bpm_for_grid) / decision.bpm_for_grid;
        if (ratio <= 0.005) {
            decision.bpm_for_grid = bpm_from_downbeats;
            bpm_source = "downbeats_override";
        }
    }
    if (decision.bpm_for_grid <= 0.0 && input.decoded.bpm > 0.0) {
        decision.bpm_for_grid = input.decoded.bpm;
        bpm_source = "decoded";
    }
    if (decision.bpm_for_grid <= 0.0 && decision.base_interval > 0.0) {
        decision.bpm_for_grid = (60.0 * input.fps) / decision.base_interval;
        bpm_source = "base_interval";
    }
    diag.bpm_source = bpm_source;
    decision.step_frames = (decision.bpm_for_grid > 0.0)
        ? (60.0 * input.fps) / decision.bpm_for_grid
        : decision.base_interval;
    diag.bpm_from_peaks = bpm_from_peaks;
    diag.bpm_from_peaks_median = bpm_from_peaks_median;
    diag.bpm_from_peaks_reg = bpm_from_peaks_reg;
    diag.bpm_from_peaks_median_full = bpm_from_peaks_median_full;
    diag.bpm_from_peaks_reg_full = bpm_from_peaks_reg_full;
    diag.bpm_from_downbeats = bpm_from_downbeats;
    diag.bpm_from_downbeats_median = bpm_from_downbeats_median;
    diag.bpm_from_downbeats_reg = bpm_from_downbeats_reg;

    return decision;
}

void log_grid_tempo_decision(const GridTempoDecision& decision,
                             const GridTempoDecisionInput& input) {
    const auto& d = decision.diagnostics;
    if (input.config.dbn_trace && d.stats_computed) {
        auto print_stats = [&](const char* label, const IntervalStats& stats) {
            if (stats.count == 0 || stats.median_interval <= 0.0) {
                std::cerr << "DBN stats: " << label << " empty\n";
                return;
            }
            const double bpm_median = (60.0 * input.fps) / stats.median_interval;
            const double bpm_mean = (60.0 * input.fps) / stats.mean_interval;
            const double interval_cv =
                stats.mean_interval > 0.0 ? (stats.stdev_interval / stats.mean_interval) : 0.0;
            std::cerr << "DBN stats: " << label
                      << " count=" << stats.count
                      << " bpm_median=" << bpm_median
                      << " bpm_mean=" << bpm_mean
                      << " interval_cv=" << interval_cv
                      << " interval_range=[" << stats.min_interval
                      << "," << stats.max_interval << "]";
            if (!stats.top_bpm_bins.empty()) {
                std::cerr << " bpm_bins:";
                for (const auto& bin : stats.top_bpm_bins) {
                    std::cerr << " " << bin.first << "(" << bin.second << ")";
                }
            }
            std::cerr << "\n";
        };
        print_stats("tempo_peaks", d.tempo_stats);
        if (d.has_downbeat_stats) {
            print_stats("downbeat_peaks", d.downbeat_stats);
        }
        print_stats("decoded_beats", d.decoded_stats);
        print_stats("decoded_beats_filtered", d.decoded_filtered_stats);
        if (d.short_interval_threshold > 0.0) {
            std::cerr << "DBN stats: filter_threshold=" << d.short_interval_threshold
                      << " min_interval=" << d.min_interval_frames << "\n";
        }
    }
    if (input.config.verbose) {
        std::cerr << "DBN grid: bpm=" << input.decoded.bpm
                  << " bpm_from_fit=" << d.bpm_from_fit
                  << " bpm_from_global_fit=" << d.bpm_from_global_fit
                  << " bpm_from_peaks=" << d.bpm_from_peaks
                  << " bpm_from_peaks_median=" << d.bpm_from_peaks_median
                  << " bpm_from_peaks_reg=" << d.bpm_from_peaks_reg
                  << " bpm_from_peaks_median_full=" << d.bpm_from_peaks_median_full
                  << " bpm_from_peaks_reg_full=" << d.bpm_from_peaks_reg_full
                  << " bpm_from_downbeats=" << d.bpm_from_downbeats
                  << " bpm_from_downbeats_median=" << d.bpm_from_downbeats_median
                  << " bpm_from_downbeats_reg=" << d.bpm_from_downbeats_reg
                  << " base_interval=" << decision.base_interval
                  << " bpm_reference=" << input.reference_bpm
                  << " quality_qpar=" << d.quality_qpar
                  << " quality_qkur=" << d.quality_qkur
                  << " quality_low=" << (decision.quality_low ? 1 : 0)
                  << " bpm_for_grid=" << decision.bpm_for_grid
                  << " step_frames=" << decision.step_frames
                  << " start_frame=" << input.decoded.beat_frames.front()
                  << "\n";
    }
    if (input.config.dbn_trace) {
        std::cerr << "DBN quality gate: low=" << (decision.quality_low ? 1 : 0)
                  << " drop_ref=" << (d.drop_ref ? 1 : 0)
                  << " drop_fit=" << (d.drop_fit ? 1 : 0)
                  << " downbeat_ok=" << (decision.downbeat_override_ok ? 1 : 0)
                  << " downbeat_cv=" << d.downbeat_cv
                  << " downbeat_count=" << d.downbeat_count
                  << " used=" << d.bpm_source
                  << " pre_override=" << d.bpm_source_before_downbeat
                  << " pre_bpm=" << d.bpm_before_downbeat
                  << "\n";
    }
}

} // namespace beatit::detail
