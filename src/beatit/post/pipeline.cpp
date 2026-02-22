//
//  pipeline.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/coreml.h"
#include "beatit/dbn_beatit.h"
#include "beatit/dbn_calmdad.h"

#include "beatit/pp_helpers.h"
#include "beatit/pp_dbn.h"
#include "beatit/pp_result_ops.h"
#include "beatit/pp_logits.h"
#include "beatit/pp_tempo_fit.h"
#include "beatit/pp_window.h"
#include "beatit/logging.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace beatit {
namespace {

using detail::IntervalStats;
using detail::fill_peaks_with_gaps;
using detail::fill_peaks_with_grid;
using detail::filter_short_intervals;
using detail::guard_projected_downbeat_phase;
using detail::interval_stats_frames;
using detail::interval_stats_interpolated;
using detail::median_interval_frames;
using detail::median_interval_frames_interpolated;
using detail::pick_peaks;
using detail::regression_interval_frames_interpolated;
using detail::score_peaks;
using detail::trace_grid_peak_alignment;
using detail::WindowSummary;
using detail::align_downbeats_to_beats;
using detail::compute_minimal_peaks;
using detail::infer_bpb_phase;
using detail::project_downbeats_from_beats;
using detail::select_dbn_window;
using detail::select_dbn_window_energy;
using detail::summarize_window;
using detail::window_tempo_score;

} // namespace

CoreMLResult postprocess_coreml_activations(const std::vector<float>& beat_activation,
                                            const std::vector<float>& downbeat_activation,
                                            const std::vector<float>* phase_energy,
                                            const CoreMLConfig& config,
                                            double sample_rate,
                                            float reference_bpm,
                                            std::size_t last_active_frame,
                                            std::size_t total_frames_full) {
    set_log_verbosity_from_config(config);

    double dbn_ms = 0.0;
    double peaks_ms = 0.0;

    CoreMLResult result;
    result.beat_activation = beat_activation;
    result.downbeat_activation = downbeat_activation;

    const std::size_t used_frames = result.beat_activation.size();
    if (used_frames == 0) {
        return result;
    }
    const std::size_t grid_total_frames =
        total_frames_full > used_frames ? total_frames_full : used_frames;

    const float hard_min_bpm = std::max(1.0f, config.min_bpm);
    const float hard_max_bpm = std::max(hard_min_bpm + 1.0f, config.max_bpm);
    auto clamp_bpm_range = [&](float* min_value, float* max_value) {
        *min_value = std::max(hard_min_bpm, *min_value);
        *max_value = std::min(hard_max_bpm, *max_value);
        if (*max_value <= *min_value) {
            *min_value = hard_min_bpm;
            *max_value = hard_max_bpm;
        }
    };

    float min_bpm = hard_min_bpm;
    float max_bpm = hard_max_bpm;
    float min_bpm_alt = min_bpm;
    float max_bpm_alt = max_bpm;
    bool has_window = false;
    if (config.tempo_window_percent > 0.0f && reference_bpm > 0.0f) {
        const float window = config.tempo_window_percent / 100.0f;
        min_bpm = reference_bpm * (1.0f - window);
        max_bpm = reference_bpm * (1.0f + window);
        if (config.prefer_double_time) {
            const float doubled = reference_bpm * 2.0f;
            min_bpm_alt = doubled * (1.0f - window);
            max_bpm_alt = doubled * (1.0f + window);
            has_window = true;
        }
        clamp_bpm_range(&min_bpm, &max_bpm);
        clamp_bpm_range(&min_bpm_alt, &max_bpm_alt);
    }

    const double fps = static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size);
    const double hop_scale = sample_rate / static_cast<double>(config.sample_rate);
    const bool windowed_inference =
        config.fixed_frames > 0 && used_frames > config.fixed_frames;
    const std::size_t analysis_latency_frames =
        windowed_inference ? std::min(config.window_border_frames,
                                      config.fixed_frames / 2)
                           : 0;
    const double analysis_latency_frames_f =
        static_cast<double>(analysis_latency_frames);

    if (config.debug_activations_start_s >= 0.0 &&
        config.debug_activations_end_s > config.debug_activations_start_s) {
        const std::size_t start_frame = static_cast<std::size_t>(
            std::max(0.0, std::floor(config.debug_activations_start_s * fps)));
        const std::size_t end_frame = static_cast<std::size_t>(
            std::min<double>(used_frames - 1,
                             std::ceil(config.debug_activations_end_s * fps)));
        const double epsilon = 1e-5;
        const double floor_value = epsilon / 2.0;
        std::size_t emitted = 0;

        std::cerr << "Activation window: start=" << config.debug_activations_start_s
                  << "s end=" << config.debug_activations_end_s
                  << "s fps=" << fps
                  << " hop=" << config.hop_size
                  << " hop_scale=" << hop_scale
                  << " frames=[" << start_frame << "," << end_frame << "]\n";
        std::cerr << "Activations (frame,time_s,sample_frame,"
                  << "beat_raw,downbeat_raw,beat_prob,downbeat_prob,combined)\n";
        std::cerr << std::fixed << std::setprecision(6);
        for (std::size_t frame = start_frame; frame <= end_frame; ++frame) {
            const float beat_raw = result.beat_activation[frame];
            const float downbeat_raw =
                (frame < result.downbeat_activation.size()) ? result.downbeat_activation[frame] : 0.0f;
            double beat_prob = beat_raw;
            double downbeat_prob = downbeat_raw;
            double combined = beat_raw;
            if (config.dbn_mode == CoreMLConfig::DBNMode::Calmdad) {
                beat_prob = static_cast<double>(beat_raw) * (1.0 - epsilon) + floor_value;
                downbeat_prob = static_cast<double>(downbeat_raw) * (1.0 - epsilon) + floor_value;
                combined = std::max(floor_value, beat_prob - downbeat_prob);
            }
            const double time_s = static_cast<double>(frame) / fps;
            const double sample_pos = static_cast<double>(frame * config.hop_size) * hop_scale;
            std::cerr << frame << "," << time_s << "," << static_cast<unsigned long long>(std::llround(sample_pos))
                      << "," << beat_raw << "," << downbeat_raw
                      << "," << beat_prob << "," << downbeat_prob << "," << combined << "\n";
            if (config.debug_activations_max > 0 &&
                ++emitted >= config.debug_activations_max) {
                std::cerr << "Activations truncated at " << emitted << " rows\n";
                break;
            }
        }
        std::cerr << std::defaultfloat;
    }

    constexpr std::size_t kRefineWindow = 2;

    auto fill_beats_from_frames = [&](const std::vector<std::size_t>& frames) {
        detail::fill_beats_from_frames(result,
                                       frames,
                                       config,
                                       sample_rate,
                                       hop_scale,
                                       analysis_latency_frames,
                                       analysis_latency_frames_f,
                                       kRefineWindow);
    };


    if (!config.use_dbn) {
        detail::run_logit_consensus_postprocess(result,
                                                phase_energy,
                                                config,
                                                sample_rate,
                                                min_bpm,
                                                max_bpm,
                                                grid_total_frames,
                                                fps,
                                                hop_scale,
                                                analysis_latency_frames,
                                                analysis_latency_frames_f,
                                                dbn_ms,
                                                peaks_ms);
        return result;
    }


    if (config.use_dbn) {
        float dbn_min_bpm = min_bpm;
        float dbn_max_bpm = max_bpm;
        if (config.prefer_double_time && has_window) {
            dbn_min_bpm = std::min(dbn_min_bpm, min_bpm_alt);
            dbn_max_bpm = std::max(dbn_max_bpm, max_bpm_alt);
        }
        clamp_bpm_range(&dbn_min_bpm, &dbn_max_bpm);
        if (detail::run_dbn_postprocess(result,
                                        phase_energy,
                                        config,
                                        sample_rate,
                                        reference_bpm,
                                        grid_total_frames,
                                        dbn_min_bpm,
                                        dbn_max_bpm,
                                        fps,
                                        hop_scale,
                                        analysis_latency_frames,
                                        analysis_latency_frames_f,
                                        dbn_ms,
                                        peaks_ms)) {
            return result;
        }
    }

    if (!config.use_dbn && config.use_minimal_postprocess) {
        const std::vector<std::size_t> beat_peaks = compute_minimal_peaks(result.beat_activation);
        const std::vector<std::size_t> downbeat_peaks =
            compute_minimal_peaks(result.downbeat_activation);
        fill_beats_from_frames(beat_peaks);
        const std::vector<std::size_t> aligned_downbeats =
            align_downbeats_to_beats(beat_peaks, downbeat_peaks);
        result.downbeat_feature_frames.clear();
        result.downbeat_feature_frames.reserve(aligned_downbeats.size());
        for (std::size_t frame : aligned_downbeats) {
            result.downbeat_feature_frames.push_back(static_cast<unsigned long long>(frame));
        }
        if (config.profile) {
            BEATIT_LOG_INFO("Timing(postprocess): dbn=" << dbn_ms
                            << "ms peaks=" << peaks_ms << "ms");
        }
        return result;
    }

    auto compute_peaks = [&](const std::vector<float>& activation,
                             float local_min_bpm,
                             float local_max_bpm,
                             float threshold) {
        const auto peaks_start = std::chrono::steady_clock::now();
        const double max_bpm_local = std::max(local_min_bpm + 1.0f, local_max_bpm);
        const double min_bpm_local = std::max(1.0f, local_min_bpm);
        const std::size_t min_interval =
            static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm_local)));
        const std::size_t max_interval =
            static_cast<std::size_t>(std::ceil((60.0 * fps) / min_bpm_local));
        auto peaks = pick_peaks(activation, threshold, min_interval, max_interval);
        const auto peaks_end = std::chrono::steady_clock::now();
        peaks_ms += std::chrono::duration<double, std::milli>(peaks_end - peaks_start).count();
        return peaks;
    };

    auto adjust_threshold = [&](const std::vector<float>& activation,
                                float local_min_bpm,
                                float local_max_bpm,
                                std::vector<std::size_t>* peaks) {
        if (!peaks) {
            return;
        }
        if (config.synthetic_fill) {
            return;
        }
        const double duration = static_cast<double>(used_frames) / fps;
        const double min_expected = duration * (std::max(1.0f, local_min_bpm) / 60.0);
        if (min_expected <= 0.0) {
            return;
        }
        if (peaks->size() < static_cast<std::size_t>(min_expected * 0.5)) {
            const float lowered = std::max(0.1f, config.activation_threshold * 0.5f);
            *peaks = compute_peaks(activation, local_min_bpm, local_max_bpm, lowered);
        }
    };

    std::vector<std::size_t> peaks =
        compute_peaks(result.beat_activation, min_bpm, max_bpm, config.activation_threshold);
    adjust_threshold(result.beat_activation, min_bpm, max_bpm, &peaks);
    if (config.prefer_double_time && has_window) {
        std::vector<std::size_t> peaks_alt =
            compute_peaks(result.beat_activation, min_bpm_alt, max_bpm_alt, config.activation_threshold);
        adjust_threshold(result.beat_activation, min_bpm_alt, max_bpm_alt, &peaks_alt);
        if (score_peaks(result.beat_activation, peaks_alt) >
            score_peaks(result.beat_activation, peaks)) {
            peaks.swap(peaks_alt);
        }
    }

    const float activation_floor = std::max(0.05f, config.activation_threshold * 0.2f);
    if (config.synthetic_fill) {
        double base_interval_frames = 0.0;
        if (reference_bpm > 0.0f) {
            base_interval_frames = (60.0 * fps) / reference_bpm;
        }
        if (base_interval_frames <= 1.0) {
            base_interval_frames = median_interval_frames(peaks);
        }
        std::size_t active_end = last_active_frame;
        if (active_end == 0 && used_frames > 0) {
            active_end = used_frames - 1;
        }
        std::vector<std::size_t> filled =
            fill_peaks_with_gaps(result.beat_activation,
                                 peaks,
                                 fps,
                                 activation_floor,
                                 last_active_frame,
                                 base_interval_frames,
                                 config.gap_tolerance,
                                 config.offbeat_tolerance,
                                 config.tempo_window_beats);
        if (base_interval_frames > 1.0 && !peaks.empty()) {
            std::vector<std::size_t> grid =
                fill_peaks_with_grid(result.beat_activation,
                                     peaks.front(),
                                     active_end,
                                     base_interval_frames,
                                     activation_floor);
            if (grid.size() > filled.size()) {
                filled.swap(grid);
            }
        }
        if (filled.size() > peaks.size()) {
            peaks.swap(filled);
        }
    }

    fill_beats_from_frames(peaks);

    if (!result.downbeat_activation.empty()) {
        std::vector<std::size_t> down_peaks =
            compute_peaks(result.downbeat_activation, min_bpm, max_bpm, config.activation_threshold);
        adjust_threshold(result.downbeat_activation, min_bpm, max_bpm, &down_peaks);
        if (config.prefer_double_time && has_window) {
            std::vector<std::size_t> peaks_alt =
                compute_peaks(result.downbeat_activation, min_bpm_alt, max_bpm_alt, config.activation_threshold);
            adjust_threshold(result.downbeat_activation, min_bpm_alt, max_bpm_alt, &peaks_alt);
            if (score_peaks(result.downbeat_activation, peaks_alt) >
                score_peaks(result.downbeat_activation, down_peaks)) {
                down_peaks.swap(peaks_alt);
            }
        }
        result.downbeat_feature_frames.clear();
        result.downbeat_feature_frames.reserve(down_peaks.size());
        for (std::size_t frame : down_peaks) {
            result.downbeat_feature_frames.push_back(static_cast<unsigned long long>(frame));
        }
    } else if (!result.beat_feature_frames.empty()) {
        result.downbeat_feature_frames.push_back(result.beat_feature_frames.front());
    }

    if (config.profile) {
        BEATIT_LOG_INFO("Timing(postprocess): dbn=" << dbn_ms
                        << "ms peaks=" << peaks_ms << "ms");
    }

    return result;
}

} // namespace beatit
