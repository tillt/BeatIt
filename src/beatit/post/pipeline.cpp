//
//  pipeline.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/config.h"

#include "beatit/post/dbn_run.h"
#include "beatit/post/helpers.h"
#include "beatit/post/result_ops.h"
#include "beatit/post/logits.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <vector>

namespace beatit {
namespace {

using detail::fill_peaks_with_gaps;
using detail::fill_peaks_with_grid;
using detail::median_interval_frames;
using detail::pick_peaks;
using detail::score_peaks;

struct TempoWindowConfig {
    float hard_min_bpm = 1.0f;
    float hard_max_bpm = 2.0f;
    float min_bpm = 1.0f;
    float max_bpm = 2.0f;
    float min_bpm_alt = 1.0f;
    float max_bpm_alt = 2.0f;
    bool has_alt_window = false;
};

void clamp_tempo_window(float& min_bpm, float& max_bpm, float hard_min_bpm, float hard_max_bpm) {
    min_bpm = std::max(hard_min_bpm, min_bpm);
    max_bpm = std::min(hard_max_bpm, max_bpm);
    if (max_bpm <= min_bpm) {
        min_bpm = hard_min_bpm;
        max_bpm = hard_max_bpm;
    }
}

TempoWindowConfig build_tempo_window_config(const BeatitConfig& config, float reference_bpm) {
    TempoWindowConfig out;
    out.hard_min_bpm = std::max(1.0f, config.min_bpm);
    out.hard_max_bpm = std::max(out.hard_min_bpm + 1.0f, config.max_bpm);
    out.min_bpm = out.hard_min_bpm;
    out.max_bpm = out.hard_max_bpm;
    out.min_bpm_alt = out.min_bpm;
    out.max_bpm_alt = out.max_bpm;

    if (config.tempo_window_percent <= 0.0f || reference_bpm <= 0.0f) {
        return out;
    }

    const float window = config.tempo_window_percent / 100.0f;
    out.min_bpm = reference_bpm * (1.0f - window);
    out.max_bpm = reference_bpm * (1.0f + window);
    if (config.prefer_double_time) {
        const float doubled = reference_bpm * 2.0f;
        out.min_bpm_alt = doubled * (1.0f - window);
        out.max_bpm_alt = doubled * (1.0f + window);
        out.has_alt_window = true;
    }

    clamp_tempo_window(out.min_bpm, out.max_bpm, out.hard_min_bpm, out.hard_max_bpm);
    clamp_tempo_window(out.min_bpm_alt, out.max_bpm_alt, out.hard_min_bpm, out.hard_max_bpm);
    return out;
}

std::vector<std::size_t> compute_post_peaks(const std::vector<float>& activation,
                                            double fps,
                                            float min_bpm,
                                            float max_bpm,
                                            float threshold,
                                            double& peaks_ms) {
    const auto peaks_start = std::chrono::steady_clock::now();
    const double max_bpm_local = std::max(min_bpm + 1.0f, max_bpm);
    const double min_bpm_local = std::max(1.0f, min_bpm);
    const std::size_t min_interval =
        static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm_local)));
    const std::size_t max_interval =
        static_cast<std::size_t>(std::ceil((60.0 * fps) / min_bpm_local));
    std::vector<std::size_t> peaks = pick_peaks(activation, threshold, min_interval, max_interval);
    const auto peaks_end = std::chrono::steady_clock::now();
    peaks_ms += std::chrono::duration<double, std::milli>(peaks_end - peaks_start).count();
    return peaks;
}

void relax_post_peak_threshold(std::vector<std::size_t>& peaks,
                               const std::vector<float>& activation,
                               const BeatitConfig& config,
                               double fps,
                               std::size_t used_frames,
                               float min_bpm,
                               float max_bpm,
                               double& peaks_ms) {
    if (config.synthetic_fill) {
        return;
    }

    const double duration = static_cast<double>(used_frames) / fps;
    const double min_expected = duration * (std::max(1.0f, min_bpm) / 60.0);
    if (min_expected <= 0.0) {
        return;
    }

    if (peaks.size() < static_cast<std::size_t>(min_expected * 0.5)) {
        const float lowered = std::max(0.1f, config.activation_threshold * 0.5f);
        peaks = compute_post_peaks(activation, fps, min_bpm, max_bpm, lowered, peaks_ms);
    }
}

std::vector<std::size_t> select_post_peaks(const std::vector<float>& activation,
                                           const BeatitConfig& config,
                                           double fps,
                                           std::size_t used_frames,
                                           float min_bpm,
                                           float max_bpm,
                                           bool use_alt_window,
                                           float min_bpm_alt,
                                           float max_bpm_alt,
                                           double& peaks_ms) {
    std::vector<std::size_t> peaks =
        compute_post_peaks(activation, fps, min_bpm, max_bpm, config.activation_threshold, peaks_ms);
    relax_post_peak_threshold(peaks,
                              activation,
                              config,
                              fps,
                              used_frames,
                              min_bpm,
                              max_bpm,
                              peaks_ms);
    if (!use_alt_window) {
        return peaks;
    }

    std::vector<std::size_t> peaks_alt =
        compute_post_peaks(activation,
                           fps,
                           min_bpm_alt,
                           max_bpm_alt,
                           config.activation_threshold,
                           peaks_ms);
    relax_post_peak_threshold(peaks_alt,
                              activation,
                              config,
                              fps,
                              used_frames,
                              min_bpm_alt,
                              max_bpm_alt,
                              peaks_ms);
    if (score_peaks(activation, peaks_alt) > score_peaks(activation, peaks)) {
        peaks.swap(peaks_alt);
    }
    return peaks;
}

void dump_activation_window(const CoreMLResult& result,
                            const BeatitConfig& config,
                            double fps,
                            double hop_scale) {
    const std::size_t used_frames = result.beat_activation.size();
    if (used_frames == 0 ||
        config.debug_activations_start_s < 0.0 ||
        config.debug_activations_end_s <= config.debug_activations_start_s) {
        return;
    }

    const std::size_t start_frame = static_cast<std::size_t>(
        std::max(0.0, std::floor(config.debug_activations_start_s * fps)));
    const std::size_t end_frame = static_cast<std::size_t>(
        std::min<double>(used_frames - 1, std::ceil(config.debug_activations_end_s * fps)));
    const double epsilon = 1e-5;
    const double floor_value = epsilon / 2.0;
    std::size_t emitted = 0;

    auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
    debug_stream << std::fixed << std::setprecision(6);
    debug_stream << "Activation window: start=" << config.debug_activations_start_s
                 << "s end=" << config.debug_activations_end_s
                 << "s fps=" << fps
                 << " hop=" << config.hop_size
                 << " hop_scale=" << hop_scale
                 << " frames=[" << start_frame << "," << end_frame << "]\n";
    debug_stream << "Activations (frame,time_s,sample_frame,"
                 << "beat_raw,downbeat_raw,beat_prob,downbeat_prob,combined)\n";

    for (std::size_t frame = start_frame; frame <= end_frame; ++frame) {
        const float beat_raw = result.beat_activation[frame];
        const float downbeat_raw =
            (frame < result.downbeat_activation.size()) ? result.downbeat_activation[frame] : 0.0f;
        double beat_prob = beat_raw;
        double downbeat_prob = downbeat_raw;
        double combined = beat_raw;
        if (config.dbn_mode == BeatitConfig::DBNMode::Calmdad) {
            beat_prob = static_cast<double>(beat_raw) * (1.0 - epsilon) + floor_value;
            downbeat_prob = static_cast<double>(downbeat_raw) * (1.0 - epsilon) + floor_value;
            combined = std::max(floor_value, beat_prob - downbeat_prob);
        }
        const double time_s = static_cast<double>(frame) / fps;
        const double sample_pos = static_cast<double>(frame * config.hop_size) * hop_scale;
        debug_stream << frame << "," << time_s << ","
                     << static_cast<unsigned long long>(std::llround(sample_pos))
                     << "," << beat_raw << "," << downbeat_raw
                     << "," << beat_prob << "," << downbeat_prob << "," << combined << "\n";
        if (config.debug_activations_max > 0 && ++emitted >= config.debug_activations_max) {
            debug_stream << "Activations truncated at " << emitted << " rows\n";
            break;
        }
    }

    debug_stream << std::defaultfloat;
}

void apply_synthetic_fill(std::vector<std::size_t>& peaks,
                          const std::vector<float>& activation,
                          const BeatitConfig& config,
                          double fps,
                          float reference_bpm,
                          std::size_t last_active_frame,
                          std::size_t used_frames,
                          float activation_floor) {
    if (!config.synthetic_fill) {
        return;
    }

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
        fill_peaks_with_gaps(activation,
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
            fill_peaks_with_grid(activation,
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

} // namespace

CoreMLResult postprocess_coreml_activations(const std::vector<float>& beat_activation,
                                            const std::vector<float>& downbeat_activation,
                                            const std::vector<float>* phase_energy,
                                            const BeatitConfig& config,
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
    const TempoWindowConfig tempo_window = build_tempo_window_config(config, reference_bpm);

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

    dump_activation_window(result, config, fps, hop_scale);

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
                                                tempo_window.min_bpm,
                                                tempo_window.max_bpm,
                                                grid_total_frames,
                                                fps,
                                                hop_scale,
                                                analysis_latency_frames,
                                                analysis_latency_frames_f,
                                                dbn_ms,
                                                peaks_ms);
        return result;
    }
    float dbn_min_bpm = tempo_window.min_bpm;
    float dbn_max_bpm = tempo_window.max_bpm;
    if (config.prefer_double_time && tempo_window.has_alt_window) {
        dbn_min_bpm = std::min(dbn_min_bpm, tempo_window.min_bpm_alt);
        dbn_max_bpm = std::max(dbn_max_bpm, tempo_window.max_bpm_alt);
    }
    clamp_tempo_window(dbn_min_bpm,
                       dbn_max_bpm,
                       tempo_window.hard_min_bpm,
                       tempo_window.hard_max_bpm);
    const detail::DBNRunRequest dbn_request{
        result,
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
        peaks_ms,
        dbn_ms,
    };
    if (detail::run_dbn_postprocess(dbn_request)) {
        return result;
    }

    std::vector<std::size_t> peaks =
        select_post_peaks(result.beat_activation,
                          config,
                          fps,
                          used_frames,
                          tempo_window.min_bpm,
                          tempo_window.max_bpm,
                          config.prefer_double_time && tempo_window.has_alt_window,
                          tempo_window.min_bpm_alt,
                          tempo_window.max_bpm_alt,
                          peaks_ms);

    const float activation_floor = std::max(0.05f, config.activation_threshold * 0.2f);
    apply_synthetic_fill(peaks,
                         result.beat_activation,
                         config,
                         fps,
                         reference_bpm,
                         last_active_frame,
                         used_frames,
                         activation_floor);

    fill_beats_from_frames(peaks);

    if (!result.downbeat_activation.empty()) {
        std::vector<std::size_t> down_peaks =
            select_post_peaks(result.downbeat_activation,
                              config,
                              fps,
                              used_frames,
                              tempo_window.min_bpm,
                              tempo_window.max_bpm,
                              config.prefer_double_time && tempo_window.has_alt_window,
                              tempo_window.min_bpm_alt,
                              tempo_window.max_bpm_alt,
                              peaks_ms);
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
