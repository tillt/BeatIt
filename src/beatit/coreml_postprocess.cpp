//
//  coreml_postprocess.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/coreml.h"
#include "beatit/dbn_beatit.h"
#include "beatit/dbn_calmdad.h"

#include "beatit/coreml_postprocess_helpers.h"
#include "beatit/coreml_postprocess_tempo_fit.h"
#include "beatit/coreml_postprocess_window.h"
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

constexpr float kPi = 3.14159265358979323846f;

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
    auto refine_frame_to_peak = [&](std::size_t frame,
                                    const std::vector<float>& activation) -> std::size_t {
        if (activation.empty()) {
            return frame;
        }
        const std::size_t start = frame > kRefineWindow ? frame - kRefineWindow : 0;
        const std::size_t end = std::min(frame + kRefineWindow, activation.size() - 1);
        std::size_t best_index = frame;
        float best_value = activation[frame];
        for (std::size_t i = start; i <= end; ++i) {
            const float value = activation[i];
            if (value > best_value) {
                best_value = value;
                best_index = i;
            }
        }
        return best_index;
    };

    auto fill_beats_from_frames = [&](const std::vector<std::size_t>& frames) {
        result.beat_feature_frames.clear();
        result.beat_feature_frames.reserve(frames.size());
        result.beat_sample_frames.clear();
        result.beat_sample_frames.reserve(frames.size());
        result.beat_strengths.clear();
        result.beat_strengths.reserve(frames.size());

        const long long latency_samples = static_cast<long long>(
            std::llround(config.output_latency_seconds * sample_rate));

        for (std::size_t frame : frames) {
            const std::size_t output_frame =
                (analysis_latency_frames > 0 && frame > analysis_latency_frames)
                    ? (frame - analysis_latency_frames)
                    : (analysis_latency_frames > 0 ? 0 : frame);
            result.beat_feature_frames.push_back(static_cast<unsigned long long>(output_frame));
            const std::size_t peak_frame = refine_frame_to_peak(frame, result.beat_activation);

            double frame_pos = static_cast<double>(peak_frame);
            if (peak_frame > 0 && peak_frame + 1 < result.beat_activation.size()) {
                const double prev = result.beat_activation[peak_frame - 1];
                const double curr = result.beat_activation[peak_frame];
                const double next = result.beat_activation[peak_frame + 1];
                const double denom = prev - 2.0 * curr + next;
                if (std::abs(denom) > 1e-9) {
                    double offset = 0.5 * (prev - next) / denom;
                    offset = std::max(-0.5, std::min(0.5, offset));
                    frame_pos += offset;
                }
            }
            if (analysis_latency_frames > 0) {
                frame_pos = std::max(0.0, frame_pos - analysis_latency_frames_f);
            }
            const double sample_pos =
                (frame_pos * static_cast<double>(config.hop_size)) * hop_scale;
            long long sample_frame = static_cast<long long>(std::llround(sample_pos)) - latency_samples;
            if (sample_frame < 0) {
                sample_frame = 0;
            }
            result.beat_sample_frames.push_back(static_cast<unsigned long long>(sample_frame));
            if (!result.beat_activation.empty()) {
                result.beat_strengths.push_back(result.beat_activation[peak_frame]);
            } else {
                result.beat_strengths.push_back(0.0f);
            }
        }

        if (!result.beat_sample_frames.empty()) {
            std::size_t write = 1;
            unsigned long long last = result.beat_sample_frames[0];
            for (std::size_t i = 1; i < result.beat_sample_frames.size(); ++i) {
                const unsigned long long current = result.beat_sample_frames[i];
                if (current <= last) {
                    continue;
                }
                result.beat_sample_frames[write] = current;
                result.beat_feature_frames[write] = result.beat_feature_frames[i];
                result.beat_strengths[write] = result.beat_strengths[i];
                last = current;
                ++write;
            }
            result.beat_sample_frames.resize(write);
            result.beat_feature_frames.resize(write);
            result.beat_strengths.resize(write);
        }
    };

    [[maybe_unused]] auto fill_beats_from_frames_raw = [&](const std::vector<std::size_t>& frames) {
        result.beat_feature_frames.clear();
        result.beat_feature_frames.reserve(frames.size());
        result.beat_sample_frames.clear();
        result.beat_sample_frames.reserve(frames.size());
        result.beat_strengths.clear();
        result.beat_strengths.reserve(frames.size());

        const long long latency_samples = static_cast<long long>(
            std::llround(config.output_latency_seconds * sample_rate));

        for (std::size_t frame : frames) {
            const std::size_t output_frame =
                (analysis_latency_frames > 0 && frame > analysis_latency_frames)
                    ? (frame - analysis_latency_frames)
                    : (analysis_latency_frames > 0 ? 0 : frame);
            result.beat_feature_frames.push_back(static_cast<unsigned long long>(output_frame));
            double frame_pos = static_cast<double>(frame);
            if (analysis_latency_frames > 0) {
                frame_pos = std::max(0.0, frame_pos - analysis_latency_frames_f);
            }
            const double sample_pos =
                (frame_pos * static_cast<double>(config.hop_size)) * hop_scale;
            long long sample_frame = static_cast<long long>(std::llround(sample_pos)) - latency_samples;
            if (sample_frame < 0) {
                sample_frame = 0;
            }
            result.beat_sample_frames.push_back(static_cast<unsigned long long>(sample_frame));
            if (!result.beat_activation.empty() && frame < result.beat_activation.size()) {
                result.beat_strengths.push_back(result.beat_activation[frame]);
            } else {
                result.beat_strengths.push_back(0.0f);
            }
        }

        if (!result.beat_sample_frames.empty()) {
            std::size_t write = 1;
            unsigned long long last = result.beat_sample_frames[0];
            for (std::size_t i = 1; i < result.beat_sample_frames.size(); ++i) {
                const unsigned long long current = result.beat_sample_frames[i];
                if (current <= last) {
                    continue;
                }
                result.beat_sample_frames[write] = current;
                result.beat_feature_frames[write] = result.beat_feature_frames[i];
                result.beat_strengths[write] = result.beat_strengths[i];
                last = current;
                ++write;
            }
            result.beat_sample_frames.resize(write);
            result.beat_feature_frames.resize(write);
            result.beat_strengths.resize(write);
        }
    };

    auto fill_beats_from_bpm_grid_into = [&](std::size_t start_frame,
                                             double bpm,
                                             std::size_t total_frames,
                                             std::vector<unsigned long long>& out_feature_frames,
                                             std::vector<unsigned long long>& out_sample_frames,
                                             std::vector<float>& out_strengths) {
        out_feature_frames.clear();
        out_sample_frames.clear();
        out_strengths.clear();

        if (bpm <= 0.0 || fps <= 0.0 || total_frames == 0) {
            return;
        }

        const double step_frames = (60.0 * fps) / bpm;
        if (step_frames <= 0.0) {
            return;
        }

        // Projected DBN grid is already in timeline-aligned feature frames.
        // Applying border latency here shifts the entire grid early.
        const double start_frame_adjusted = static_cast<double>(start_frame);

        if (config.dbn_trace) {
            const double start_time = start_frame_adjusted / fps;
            const double start_sample_pos =
                (start_frame_adjusted * static_cast<double>(config.hop_size)) * hop_scale;
            const long long start_sample_frame =
                static_cast<long long>(std::llround(start_sample_pos));
            const double start_time_after_latency =
                sample_rate > 0.0
                    ? static_cast<double>(std::max<long long>(0, start_sample_frame)) /
                        sample_rate
                    : 0.0;
            std::cerr << "DBN grid project: start_frame=" << start_frame
                      << " start_time=" << start_time
                      << " bpm=" << bpm
                      << " step_frames=" << step_frames
                      << " total_frames=" << total_frames
                      << " hop_size=" << config.hop_size
                      << " hop_scale=" << hop_scale
                      << " start_sample_frame=" << start_sample_frame
                      << " start_time_adj=" << start_time_after_latency
                      << "\n";
        }

        const double step_samples = (60.0 * sample_rate) / bpm;
        if (step_samples <= 0.0) {
            return;
        }
        const double start_sample_pos =
            (start_frame_adjusted * static_cast<double>(config.hop_size)) * hop_scale;
        const long long start_sample_frame = static_cast<long long>(std::llround(start_sample_pos));
        std::vector<unsigned long long> grid_samples;
        grid_samples.reserve(static_cast<std::size_t>(
            std::ceil(static_cast<double>(total_frames) / step_frames)) + 4);
        std::size_t backward_count = 0;
        std::size_t forward_count = 0;

        double sample_pos = static_cast<double>(start_sample_frame);
        while (sample_pos >= step_samples) {
            sample_pos -= step_samples;
            grid_samples.push_back(static_cast<unsigned long long>(
                std::llround(sample_pos)));
        }
        std::reverse(grid_samples.begin(), grid_samples.end());
        backward_count = grid_samples.size();
        grid_samples.push_back(static_cast<unsigned long long>(start_sample_frame));

        sample_pos = static_cast<double>(start_sample_frame) + step_samples;
        const double total_samples = static_cast<double>(total_frames) *
            static_cast<double>(config.hop_size) * hop_scale;
        while (sample_pos < total_samples) {
            grid_samples.push_back(static_cast<unsigned long long>(
                std::llround(sample_pos)));
            ++forward_count;
            sample_pos += step_samples;
        }

        out_feature_frames.reserve(grid_samples.size());
        out_sample_frames.reserve(grid_samples.size());
        out_strengths.reserve(grid_samples.size());

        for (unsigned long long sample_frame : grid_samples) {
            out_sample_frames.push_back(sample_frame);
            const double feature_pos =
                (static_cast<double>(sample_frame) / hop_scale) /
                static_cast<double>(config.hop_size);
            const std::size_t frame = static_cast<std::size_t>(std::llround(feature_pos));
            out_feature_frames.push_back(static_cast<unsigned long long>(frame));

            if (!result.beat_activation.empty() && frame < result.beat_activation.size()) {
                out_strengths.push_back(result.beat_activation[frame]);
            } else {
                out_strengths.push_back(0.0f);
            }
        }

        if (!out_sample_frames.empty()) {
            std::size_t write = 1;
            unsigned long long last = out_sample_frames[0];
            for (std::size_t i = 1; i < out_sample_frames.size(); ++i) {
                const unsigned long long current = out_sample_frames[i];
                if (current <= last) {
                    continue;
                }
                out_sample_frames[write] = current;
                out_feature_frames[write] = out_feature_frames[i];
                out_strengths[write] = out_strengths[i];
                last = current;
                ++write;
            }
            out_sample_frames.resize(write);
            out_feature_frames.resize(write);
            out_strengths.resize(write);
        }

        if (config.dbn_trace) {
            const std::size_t preview = std::min<std::size_t>(6, out_feature_frames.size());
            std::cerr << "DBN grid beats head:";
            for (std::size_t i = 0; i < preview; ++i) {
                const std::size_t frame = static_cast<std::size_t>(out_feature_frames[i]);
                const double time_s = static_cast<double>(frame) / fps;
                std::cerr << " " << frame << "(" << time_s << "s)";
            }
            std::cerr << "\n";
            std::cerr << "DBN grid beats total=" << out_feature_frames.size()
                      << " backward=" << backward_count
                      << " forward=" << forward_count
                      << "\n";
        }
    };

    auto dedupe_frames = [&](std::vector<std::size_t>& frames) {
        if (frames.empty()) {
            return;
        }
        std::size_t write = 1;
        std::size_t last = frames[0];
        for (std::size_t i = 1; i < frames.size(); ++i) {
            const std::size_t current = frames[i];
            if (current <= last) {
                continue;
            }
            frames[write++] = current;
            last = current;
        }
        frames.resize(write);
    };

    auto dedupe_frames_tolerant = [&](std::vector<std::size_t>& frames,
                                      std::size_t tolerance) {
        if (frames.empty()) {
            return;
        }
        if (tolerance == 0) {
            dedupe_frames(frames);
            return;
        }
        std::size_t write = 1;
        std::size_t last = frames[0];
        for (std::size_t i = 1; i < frames.size(); ++i) {
            const std::size_t current = frames[i];
            if (current <= last + tolerance) {
                continue;
            }
            frames[write++] = current;
            last = current;
        }
        frames.resize(write);
    };

    auto apply_latency_to_frames = [&](const std::vector<std::size_t>& frames) {
        if (analysis_latency_frames == 0 || frames.empty()) {
            return frames;
        }
        std::vector<std::size_t> adjusted;
        adjusted.reserve(frames.size());
        for (std::size_t frame : frames) {
            if (frame > analysis_latency_frames) {
                adjusted.push_back(frame - analysis_latency_frames);
            } else {
                adjusted.push_back(0);
            }
        }
        dedupe_frames(adjusted);
        return adjusted;
    };

    auto select_dbn_window = [&](const std::vector<float>& activation,
                                 double window_seconds,
                                 bool intro_mid_outro,
                                 float local_min_bpm,
                                 float local_max_bpm,
                                 float peak_threshold) -> std::pair<std::size_t, std::size_t> {
        return detail::select_dbn_window(activation,
                                         window_seconds,
                                         intro_mid_outro,
                                         local_min_bpm,
                                         local_max_bpm,
                                         peak_threshold,
                                         fps);
    };

    auto select_dbn_window_energy = [&](const std::vector<float>& energy,
                                        double window_seconds,
                                        bool intro_mid_outro) -> std::pair<std::size_t, std::size_t> {
        return detail::select_dbn_window_energy(energy,
                                                window_seconds,
                                                intro_mid_outro,
                                                fps);
    };

    auto infer_bpb_phase = [&](const std::vector<std::size_t>& beats,
                               const std::vector<std::size_t>& downbeats,
                               const std::vector<std::size_t>& candidates) {
        return detail::infer_bpb_phase(beats, downbeats, candidates, config);
    };

    if (!config.use_dbn) {
        auto refine_peak_position = [&](std::size_t frame,
                                        const std::vector<float>& activation) -> double {
            if (activation.empty()) {
                return static_cast<double>(frame);
            }
            const std::size_t peak_frame = refine_frame_to_peak(frame, activation);
            double pos = static_cast<double>(peak_frame);
            if (peak_frame > 0 && peak_frame + 1 < activation.size()) {
                const double prev = activation[peak_frame - 1];
                const double curr = activation[peak_frame];
                const double next = activation[peak_frame + 1];
                const double denom = prev - 2.0 * curr + next;
                if (std::abs(denom) > 1e-9) {
                    double offset = 0.5 * (prev - next) / denom;
                    offset = std::max(-0.5, std::min(0.5, offset));
                    pos += offset;
                }
            }
            return pos;
        };

        auto compute_peaks_for_range = [&](const std::vector<float>& activation,
                                           float local_min_bpm,
                                           float local_max_bpm,
                                           float threshold) {
            const double max_bpm_local = std::max(local_min_bpm + 1.0f, local_max_bpm);
            const double min_bpm_local = std::max(1.0f, local_min_bpm);
            const std::size_t min_interval =
                static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm_local)));
            const std::size_t max_interval =
                static_cast<std::size_t>(std::ceil((60.0 * fps) / min_bpm_local));
            return pick_peaks(activation, threshold, min_interval, max_interval);
        };

        std::vector<float> onset_activation;
        onset_activation.reserve(result.beat_activation.size());
        const bool phase_energy_ok =
            phase_energy && !phase_energy->empty() && phase_energy->size() >= used_frames;
        const std::vector<float>* onset_source =
            phase_energy_ok ? phase_energy : &result.beat_activation;
        if (!onset_source->empty()) {
            onset_activation.push_back(0.0f);
            for (std::size_t i = 1; i < onset_source->size(); ++i) {
                const float delta = (*onset_source)[i] - (*onset_source)[i - 1];
                const float onset = std::max(0.0f, delta);
                onset_activation.push_back(onset);
            }
        }

        float max_activation = 0.0f;
        for (float v : result.beat_activation) {
            if (v > max_activation) {
                max_activation = v;
            }
        }

        float peak_threshold = std::max(0.05f, config.activation_threshold);
        if (max_activation > 0.0f) {
            const float adaptive = std::max(0.1f, max_activation * 0.5f);
            peak_threshold = std::min(peak_threshold, adaptive);
        }

        std::vector<std::size_t> beat_peaks =
            compute_peaks_for_range(result.beat_activation, min_bpm, max_bpm, peak_threshold);
        if (beat_peaks.size() < config.logit_min_peaks) {
            float lowered = std::max(0.05f, peak_threshold * 0.5f);
            if (max_activation > 0.0f) {
                lowered = std::min(lowered, std::max(0.05f, max_activation * 0.25f));
            }
            beat_peaks =
                compute_peaks_for_range(result.beat_activation, min_bpm, max_bpm, lowered);
        }

        double interval_frames =
            median_interval_frames_interpolated(result.beat_activation, beat_peaks);
        if (config.verbose) {
            std::cerr << "Logit consensus: max_activation=" << max_activation
                      << " peak_threshold=" << peak_threshold
                      << " peaks=" << beat_peaks.size()
                      << " interval_frames=" << interval_frames
                      << " fps=" << fps << "\n";
        }

        double bpm = 0.0;
        double sweep_phase = 0.0;
        double sweep_score = -1.0;
        if (fps > 0.0 && min_bpm > 0.0f && max_bpm > min_bpm) {
            const double step = 0.05;
            for (double candidate = min_bpm; candidate <= max_bpm + 1e-6; candidate += step) {
                const double period = (60.0 * fps) / candidate;
                if (period <= 1.0) {
                    continue;
                }
                const double omega = (2.0 * kPi) / period;
                double sum_cos = 0.0;
                double sum_sin = 0.0;
                double sum_weight = 0.0;
                for (std::size_t i = 0; i < used_frames; ++i) {
                    const double weight = result.beat_activation[i];
                    if (weight <= 0.0) {
                        continue;
                    }
                    const double angle = omega * static_cast<double>(i);
                    sum_cos += weight * std::cos(angle);
                    sum_sin += weight * std::sin(angle);
                    sum_weight += weight;
                }
                if (sum_weight <= 0.0) {
                    continue;
                }
                const double magnitude =
                    std::hypot(sum_cos, sum_sin) / sum_weight;
                if (magnitude > sweep_score) {
                    sweep_score = magnitude;
                    bpm = candidate;
                    const double phase_angle = std::atan2(sum_sin, sum_cos);
                    sweep_phase = (phase_angle / omega);
                    sweep_phase = std::fmod(sweep_phase, period);
                    if (sweep_phase < 0.0) {
                        sweep_phase += period;
                    }
                }
            }
            if (config.verbose) {
                std::cerr << "Logit sweep: bpm=" << bpm
                          << " phase=" << sweep_phase
                          << " score=" << sweep_score << "\n";
            }
        }

        if (bpm <= 0.0) {
            if (interval_frames <= 0.0) {
                const std::vector<std::size_t> fallback_peaks =
                    compute_minimal_peaks(result.beat_activation);
                fill_beats_from_frames(fallback_peaks);
                const std::vector<std::size_t> aligned_downbeats =
                    align_downbeats_to_beats(fallback_peaks,
                                             compute_minimal_peaks(result.downbeat_activation));
                result.downbeat_feature_frames.clear();
                result.downbeat_feature_frames.reserve(aligned_downbeats.size());
                for (std::size_t frame : aligned_downbeats) {
                    result.downbeat_feature_frames.push_back(
                        static_cast<unsigned long long>(frame));
                }
                result.beat_projected_feature_frames.clear();
                result.beat_projected_sample_frames.clear();
                result.beat_projected_strengths.clear();
                result.downbeat_projected_feature_frames.clear();
                if (config.profile) {
                    std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                              << "ms peaks=" << peaks_ms << "ms\n";
                }
                return result;
            }
            bpm = (60.0 * fps) / interval_frames;
        }

        if (bpm > 0.0) {
            while (bpm < min_bpm && bpm * 2.0 <= max_bpm) {
                bpm *= 2.0;
            }
            while (bpm > max_bpm && bpm / 2.0 >= min_bpm) {
                bpm /= 2.0;
            }
            if (bpm < min_bpm) {
                bpm = min_bpm;
            } else if (bpm > max_bpm) {
                bpm = max_bpm;
            }
        }

        const double step_frames = (60.0 * fps) / std::max(1.0, bpm);
        if (config.verbose) {
            std::cerr << "Logit consensus: bpm=" << bpm
                      << " step_frames=" << step_frames
                      << " used_frames=" << used_frames
                      << " max_shift_s=" << config.logit_phase_max_shift_seconds
                      << "\n";
        }
        if (step_frames <= 0.0) {
            return result;
        }

        std::size_t strongest_peak = 0;
        float strongest_value = 0.0f;
        for (std::size_t i = 0; i < result.beat_activation.size(); ++i) {
            const float value = result.beat_activation[i];
            if (value > strongest_value) {
                strongest_value = value;
                strongest_peak = i;
            }
        }

        const std::vector<float>& phase_signal =
            onset_activation.empty() ? result.beat_activation : onset_activation;
        float max_phase_signal = 0.0f;
        for (float value : phase_signal) {
            if (value > max_phase_signal) {
                max_phase_signal = value;
            }
        }

        // Shift peaks backward to the attack onset (earliest rise), to avoid late-phase bias.
        std::vector<float> phase_onsets;
        phase_onsets.assign(phase_signal.size(), 0.0f);
        const float onset_ratio = 0.35f;
        const std::size_t onset_max_back = static_cast<std::size_t>(
            std::max(1.0, std::round(0.2 * fps)));
        const float onset_peak_threshold = std::max(0.02f, max_phase_signal * 0.25f);

        if (phase_signal.size() >= 3) {
            const std::size_t limit = std::min(used_frames, phase_signal.size());
            for (std::size_t i = 1; i + 1 < limit; ++i) {
                const float curr = phase_signal[i];
                if (curr < onset_peak_threshold) {
                    continue;
                }
                if (curr < phase_signal[i - 1] || curr < phase_signal[i + 1]) {
                    continue;
                }
                const float threshold = curr * onset_ratio;
                std::size_t frame = i;
                std::size_t steps = 0;
                while (frame > 0 && steps < onset_max_back &&
                       phase_signal[frame] > threshold) {
                    --frame;
                    ++steps;
                }
                if (frame < phase_onsets.size()) {
                    phase_onsets[frame] = std::max(phase_onsets[frame], curr);
                }
            }
        }

        const std::vector<float>& phase_score_signal =
            phase_onsets.empty() ? phase_signal : phase_onsets;

        auto score_phase_global = [&](double phase_frame) -> double {
            if (phase_signal.empty() || step_frames <= 0.0) {
                return -1.0;
            }

            double cursor = phase_frame;
            if (cursor < 0.0) {
                const double k = std::ceil((-cursor) / step_frames);
                cursor += k * step_frames;
            }

            double sum = 0.0;
            std::size_t count = 0;
            while (cursor < static_cast<double>(used_frames) &&
                   cursor < static_cast<double>(phase_score_signal.size())) {
                const std::size_t idx = static_cast<std::size_t>(std::llround(cursor));
                if (idx < phase_score_signal.size()) {
                    sum += phase_score_signal[idx];
                    ++count;
                }
                cursor += step_frames;
            }

            return count > 0 ? (sum / static_cast<double>(count)) : -1.0;
        };

        double global_phase = sweep_phase;
        double best_score = -1.0;
        if (step_frames > 0.0) {
            const double phase_step = 1.0;
            const double max_phase = std::max(1.0, step_frames);
            for (double phase = 0.0; phase < max_phase; phase += phase_step) {
                const double score = score_phase_global(phase);
                if (score > best_score) {
                    best_score = score;
                    global_phase = phase;
                }
            }
        }

        if (best_score < 0.0) {
            global_phase = refine_peak_position(strongest_peak, result.beat_activation);
        }

        if (config.verbose) {
            std::cerr << "Logit consensus: global_phase=" << global_phase
                      << " best_score=" << best_score << "\n";
        }

        auto build_grid_frames = [&](double phase_seed) {
            std::vector<std::size_t> grid_frames;
            grid_frames.reserve(static_cast<std::size_t>(
                std::ceil(static_cast<double>(used_frames) / step_frames)) + 8);

            double cursor = phase_seed;
            if (cursor < 0.0) {
                const double k = std::ceil((-cursor) / step_frames);
                cursor += k * step_frames;
            }
            while (cursor < static_cast<double>(used_frames)) {
                grid_frames.push_back(static_cast<std::size_t>(std::llround(cursor)));
                cursor += step_frames;
            }

            std::vector<std::size_t> projected_frames;
            if (step_frames > 0.0 && grid_total_frames > 0) {
                const double total_frames = static_cast<double>(grid_total_frames);
                const double k = std::floor((0.0 - phase_seed) / step_frames);
                double cursor = phase_seed + k * step_frames;
                while (cursor < 0.0) {
                    cursor += step_frames;
                }
                while (cursor < total_frames) {
                    projected_frames.push_back(static_cast<std::size_t>(std::llround(cursor)));
                    cursor += step_frames;
                }
            }

            if (!projected_frames.empty()) {
                grid_frames.insert(grid_frames.end(),
                                   projected_frames.begin(),
                                   projected_frames.end());
            }

            return grid_frames;
        };

        std::vector<std::size_t> grid_frames = build_grid_frames(global_phase);

        if (!grid_frames.empty()) {
            std::sort(grid_frames.begin(), grid_frames.end());
            const std::size_t tolerance_frames = static_cast<std::size_t>(
                std::max(0.0, std::round(0.025 * fps)));
            dedupe_frames_tolerant(grid_frames, tolerance_frames);
        }
        if (config.verbose) {
            std::cerr << "Logit consensus: grid_frames=" << grid_frames.size();
            if (!grid_frames.empty()) {
                std::cerr << " first=" << grid_frames.front()
                          << " last=" << grid_frames.back();
            }
            std::cerr << "\n";
        }

        fill_beats_from_frames(grid_frames);
        if (config.verbose) {
            std::cerr << "Logit consensus: beats_out=" << result.beat_feature_frames.size()
                      << " samples_out=" << result.beat_sample_frames.size() << "\n";
        }

        const std::size_t bpb = std::max<std::size_t>(1, config.dbn_beats_per_bar);
        if (!result.downbeat_activation.empty()) {
            const float activation_floor = std::max(0.05f, config.activation_threshold * 0.2f);
            const float min_db_bpm = std::max(1.0f, min_bpm / static_cast<float>(bpb));
            const float max_db_bpm = std::max(min_db_bpm + 1.0f,
                                              max_bpm / static_cast<float>(bpb));
            const std::size_t min_interval =
                static_cast<std::size_t>(std::max(1.0,
                                                  std::floor((60.0 * fps) / max_db_bpm)));
            const std::size_t max_interval =
                static_cast<std::size_t>(std::ceil((60.0 * fps) / min_db_bpm));
            std::vector<std::size_t> downbeat_peaks =
                pick_peaks(result.downbeat_activation,
                           activation_floor,
                           min_interval,
                           max_interval);
            const std::vector<std::size_t> aligned_downbeats =
                align_downbeats_to_beats(grid_frames, downbeat_peaks);
            result.downbeat_feature_frames.clear();
            result.downbeat_feature_frames.reserve(aligned_downbeats.size());
            for (std::size_t frame : aligned_downbeats) {
                result.downbeat_feature_frames.push_back(
                    static_cast<unsigned long long>(frame));
            }
        } else if (!result.beat_feature_frames.empty()) {
            result.downbeat_feature_frames.push_back(result.beat_feature_frames.front());
        }

        result.beat_projected_feature_frames.clear();
        result.beat_projected_sample_frames.clear();
        result.beat_projected_strengths.clear();
        result.downbeat_projected_feature_frames.clear();

        if (config.profile) {
            std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                      << "ms peaks=" << peaks_ms << "ms\n";
        }
        return result;
    }

    if (config.use_dbn) {
        if (config.prefer_double_time && has_window) {
            min_bpm = std::min(min_bpm, min_bpm_alt);
            max_bpm = std::max(max_bpm, max_bpm_alt);
        }
        clamp_bpm_range(&min_bpm, &max_bpm);
        const float window_peak_threshold =
            std::max(config.activation_threshold, config.dbn_activation_floor);
        std::pair<std::size_t, std::size_t> window{0, used_frames};
        bool window_energy = false;
        const bool phase_energy_ok =
            phase_energy && !phase_energy->empty() && phase_energy->size() >= used_frames;
        if (phase_energy_ok) {
            window = select_dbn_window_energy(*phase_energy,
                                              config.dbn_window_seconds,
                                              false);
            window_energy = true;
        } else {
            window = select_dbn_window(result.beat_activation,
                                       config.dbn_window_seconds,
                                       true,
                                       min_bpm,
                                       max_bpm,
                                       window_peak_threshold);
        }
        const std::size_t window_start = window.first;
        const std::size_t window_end = window.second;
        const bool use_window = (window_start > 0 || window_end < used_frames);
        std::vector<float> beat_slice;
        std::vector<float> downbeat_slice;
        if (use_window) {
            beat_slice.assign(result.beat_activation.begin() + window_start,
                              result.beat_activation.begin() + window_end);
            if (!result.downbeat_activation.empty()) {
                downbeat_slice.assign(result.downbeat_activation.begin() + window_start,
                                      result.downbeat_activation.begin() + window_end);
            }
            if (config.verbose) {
                std::cerr << "DBN window: start=" << window_start
                          << " end=" << window_end
                          << " frames=" << (window_end - window_start)
                          << " (" << ((window_end - window_start) / fps) << "s)"
                          << " selector=" << (window_energy ? "best-energy-phase" : "tempo")
                          << " energy=" << (window_energy ? "phase" : "beat")
                          << "\n";
            }
        }
        double quality_qpar = 0.0;
        double quality_qmax = 0.0;
        double quality_qkur = 0.0;
        bool quality_valid = false;
        if (fps > 0.0) {
            const std::vector<float>& quality_src =
                use_window ? beat_slice : result.beat_activation;
            if (quality_src.size() >= 16) {
                const double min_bpm_q = std::max(1.0, static_cast<double>(config.min_bpm));
                const double max_bpm_q = std::max(min_bpm_q + 1.0,
                                                  static_cast<double>(config.max_bpm));
                const std::size_t min_lag =
                    static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm_q)));
                const std::size_t max_lag =
                    static_cast<std::size_t>(std::max<double>(min_lag + 1,
                                                              std::ceil((60.0 * fps) / min_bpm_q)));
                const std::size_t max_lag_clamped =
                    std::min<std::size_t>(max_lag, quality_src.size() - 1);
                if (max_lag_clamped > min_lag) {
                    std::vector<double> salience;
                    salience.reserve(max_lag_clamped - min_lag + 1);
                    for (std::size_t lag = min_lag; lag <= max_lag_clamped; ++lag) {
                        double sum = 0.0;
                        std::size_t count = 0;
                        for (std::size_t i = lag; i < quality_src.size(); ++i) {
                            sum += static_cast<double>(quality_src[i]) *
                                   static_cast<double>(quality_src[i - lag]);
                            ++count;
                        }
                        const double value = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
                        salience.push_back(value);
                    }
                    double mean = 0.0;
                    for (double v : salience) {
                        mean += v;
                    }
                    mean /= static_cast<double>(salience.size());
                    double var = 0.0;
                    for (double v : salience) {
                        const double d = v - mean;
                        var += d * d;
                    }
                    var /= static_cast<double>(salience.size());
                    const double rms = std::sqrt(var + mean * mean);
                    double max_val = 0.0;
                    for (double v : salience) {
                        if (v > max_val) {
                            max_val = v;
                        }
                    }
                    double kurtosis = 0.0;
                    if (var > 1e-12) {
                        double m4 = 0.0;
                        for (double v : salience) {
                            const double d = v - mean;
                            m4 += d * d * d * d;
                        }
                        m4 /= static_cast<double>(salience.size());
                        kurtosis = m4 / (var * var);
                    }
                    quality_qpar = (rms > 1e-12) ? (max_val / rms) : 0.0;
                    quality_qmax = max_val;
                    quality_qkur = kurtosis;
                    quality_valid = true;
                    if (config.dbn_trace) {
                        std::cerr << "DBN quality: qpar=" << quality_qpar
                                  << " qmax=" << quality_qmax
                                  << " qkur=" << quality_qkur
                                  << " lags=[" << min_lag << "," << max_lag_clamped << "]"
                                  << " frames=" << quality_src.size()
                                  << "\n";
                    }
                }
            }
        }
        const auto dbn_start = std::chrono::steady_clock::now();
        DBNDecodeResult decoded;
        const CalmdadDecoder calmdad_decoder(config);
        if (config.dbn_mode == CoreMLConfig::DBNMode::Calmdad) {
            if (config.dbn_tempo_prior_weight > 0.0f) {
                const double tolerance =
                    std::max(0.0, static_cast<double>(config.dbn_interval_tolerance));
                const double min_interval_frames =
                    std::max(1.0, (60.0 * fps) / std::max(1.0f, max_bpm)) * (1.0 - tolerance);
                const double max_interval_frames =
                    std::max(1.0, (60.0 * fps) / std::max(1.0f, min_bpm)) * (1.0 + tolerance);
                const std::size_t peak_min_interval =
                    static_cast<std::size_t>(std::max(1.0, std::floor(min_interval_frames)));
                const std::size_t peak_max_interval =
                    static_cast<std::size_t>(std::max<double>(peak_min_interval,
                                                              std::ceil(max_interval_frames)));
                const float peak_threshold =
                    std::max(config.activation_threshold, config.dbn_activation_floor);

                const std::vector<float>& prior_src =
                    use_window ? beat_slice : result.beat_activation;
                std::vector<std::size_t> prior_peaks =
                    pick_peaks(prior_src, peak_threshold, peak_min_interval, peak_max_interval);
                const double prior_interval = median_interval_frames(prior_peaks);
                if (prior_interval > 1.0) {
                    const double prior_bpm = (60.0 * fps) / prior_interval;
                    const double window_pct = config.tempo_window_percent > 0.0f
                        ? (static_cast<double>(config.tempo_window_percent) / 100.0)
                        : 0.10;
                    min_bpm = static_cast<float>(prior_bpm * (1.0 - window_pct));
                    max_bpm = static_cast<float>(prior_bpm * (1.0 + window_pct));
                    clamp_bpm_range(&min_bpm, &max_bpm);
                    if (config.verbose) {
                        std::cerr << "DBN calmdad prior: bpm=" << prior_bpm
                                  << " peaks=" << prior_peaks.size()
                                  << " window_pct=" << window_pct
                                  << " clamp=[" << min_bpm << "," << max_bpm << "]\n";
                    }
                } else if (config.verbose) {
                    std::cerr << "DBN calmdad prior: insufficient peaks for clamp\n";
                }
            }
                decoded = calmdad_decoder.decode({
                    use_window ? beat_slice : result.beat_activation,
                    use_window ? downbeat_slice : result.downbeat_activation,
                    fps,
                    min_bpm,
                    max_bpm,
                    config.dbn_bpm_step,
                });
        } else {
            decoded = decode_dbn_beats_beatit(use_window ? beat_slice : result.beat_activation,
                                              use_window ? downbeat_slice : result.downbeat_activation,
                                              fps,
                                              min_bpm,
                                              max_bpm,
                                              config,
                                              reference_bpm);
        }
        const auto dbn_end = std::chrono::steady_clock::now();
        dbn_ms += std::chrono::duration<double, std::milli>(dbn_end - dbn_start).count();
        if (!decoded.beat_frames.empty()) {
            double projected_bpm = 0.0;
            if (use_window) {
                for (std::size_t& frame : decoded.beat_frames) {
                    frame += window_start;
                }
                for (std::size_t& frame : decoded.downbeat_frames) {
                    frame += window_start;
                }
            }
            if (config.dbn_project_grid) {
                std::vector<std::size_t> refined_beats;
                refined_beats.reserve(decoded.beat_frames.size());
                for (std::size_t frame : decoded.beat_frames) {
                    refined_beats.push_back(refine_frame_to_peak(frame, result.beat_activation));
                }
                decoded.beat_frames = std::move(refined_beats);
                dedupe_frames(decoded.beat_frames);

                if (!decoded.downbeat_frames.empty()) {
                    const std::vector<float>& downbeat_source =
                        result.downbeat_activation.empty() ? result.beat_activation
                                                           : result.downbeat_activation;
                    std::vector<std::size_t> refined_downbeats;
                    refined_downbeats.reserve(decoded.downbeat_frames.size());
                    for (std::size_t frame : decoded.downbeat_frames) {
                        refined_downbeats.push_back(refine_frame_to_peak(frame, downbeat_source));
                    }
                    decoded.downbeat_frames = std::move(refined_downbeats);
                    dedupe_frames(decoded.downbeat_frames);
                }

                const double min_interval_frames =
                    (max_bpm > 1.0f && fps > 0.0) ? (60.0 * fps) / max_bpm : 0.0;
                const double short_interval_threshold =
                    (min_interval_frames > 0.0) ? std::max(1.0, min_interval_frames * 0.5) : 0.0;
                const std::vector<std::size_t> filtered_beats =
                    filter_short_intervals(decoded.beat_frames, short_interval_threshold);
                const std::vector<std::size_t> aligned_downbeats =
                    align_downbeats_to_beats(filtered_beats, decoded.downbeat_frames);

                std::vector<std::size_t> bpb_candidates;
                if (config.dbn_mode == CoreMLConfig::DBNMode::Calmdad) {
                    bpb_candidates = {3, 4};
                } else {
                    bpb_candidates = {config.dbn_beats_per_bar};
                }
                const auto [bpb, phase] =
                    infer_bpb_phase(filtered_beats, aligned_downbeats, bpb_candidates);
                const double base_interval = median_interval_frames(filtered_beats);
                const std::vector<float>& tempo_activation =
                    use_window ? beat_slice : result.beat_activation;
                const float tempo_threshold =
                    std::max(config.dbn_activation_floor, config.activation_threshold * 0.5f);
                const std::size_t tempo_min_interval =
                    static_cast<std::size_t>(std::max(1.0,
                                                      std::floor((60.0 * fps) /
                                                                 std::max(1.0f, max_bpm))));
                const std::size_t tempo_max_interval =
                    static_cast<std::size_t>(std::max<double>(tempo_min_interval,
                                                              std::ceil((60.0 * fps) /
                                                                        std::max(1.0f, min_bpm))));
                const std::vector<std::size_t> tempo_peaks =
                    pick_peaks(tempo_activation,
                               tempo_threshold,
                               tempo_min_interval,
                               tempo_max_interval);
                const std::vector<std::size_t> tempo_peaks_full =
                    use_window
                        ? pick_peaks(result.beat_activation,
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
                        bpm_from_peaks_median = (60.0 * fps) / interval_median;
                        bpm_from_peaks = bpm_from_peaks_median;
                    }
                    if (interval_reg > 0.0) {
                        bpm_from_peaks_reg = (60.0 * fps) / interval_reg;
                        if (bpm_from_peaks_median > 0.0) {
                            const double ratio =
                                std::abs(bpm_from_peaks_reg - bpm_from_peaks_median) /
                                bpm_from_peaks_median;
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
                        median_interval_frames_interpolated(result.beat_activation, tempo_peaks_full);
                    const double interval_reg =
                        regression_interval_frames_interpolated(result.beat_activation, tempo_peaks_full);
                    if (interval_median > 0.0) {
                        bpm_from_peaks_median_full = (60.0 * fps) / interval_median;
                    }
                    if (interval_reg > 0.0) {
                        bpm_from_peaks_reg_full = (60.0 * fps) / interval_reg;
                    }
                }
                double bpm_from_downbeats = 0.0;
                double bpm_from_downbeats_median = 0.0;
                double bpm_from_downbeats_reg = 0.0;
                std::vector<std::size_t> downbeat_peaks;
                IntervalStats downbeat_stats;
                if (!result.downbeat_activation.empty() && bpb > 0) {
                    const std::vector<float>& downbeat_activation =
                        use_window ? downbeat_slice : result.downbeat_activation;
                    const float downbeat_min_bpm =
                        std::max(1.0f, min_bpm / static_cast<float>(bpb));
                    const float downbeat_max_bpm =
                        std::max(downbeat_min_bpm + 1.0f, max_bpm / static_cast<float>(bpb));
                    const std::size_t downbeat_min_interval =
                        static_cast<std::size_t>(std::max(1.0,
                                                          std::floor((60.0 * fps) /
                                                                     downbeat_max_bpm)));
                    const std::size_t downbeat_max_interval =
                        static_cast<std::size_t>(std::max<double>(downbeat_min_interval,
                                                                  std::ceil((60.0 * fps) /
                                                                            downbeat_min_bpm)));
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
                            const double downbeat_bpm = (60.0 * fps) / interval_median;
                            bpm_from_downbeats_median = downbeat_bpm * static_cast<double>(bpb);
                            bpm_from_downbeats = bpm_from_downbeats_median;
                        }
                        if (interval_reg > 0.0) {
                            const double downbeat_bpm = (60.0 * fps) / interval_reg;
                            bpm_from_downbeats_reg = downbeat_bpm * static_cast<double>(bpb);
                            if (bpm_from_downbeats_median > 0.0) {
                                const double ratio =
                                    std::abs(bpm_from_downbeats_reg - bpm_from_downbeats_median) /
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
                    downbeat_stats = interval_stats_interpolated(
                        use_window ? downbeat_slice : result.downbeat_activation,
                        downbeat_peaks,
                        fps,
                        0.2);
                }
                if (config.dbn_trace) {
                    const IntervalStats tempo_stats =
                        interval_stats_interpolated(tempo_activation, tempo_peaks, fps, 0.2);
                    const IntervalStats decoded_stats =
                        interval_stats_frames(decoded.beat_frames, fps, 0.2);
                    const IntervalStats decoded_filtered_stats =
                        interval_stats_frames(filtered_beats, fps, 0.2);
                    auto print_stats = [&](const char* label, const IntervalStats& stats) {
                        if (stats.count == 0 || stats.median_interval <= 0.0) {
                            std::cerr << "DBN stats: " << label << " empty\n";
                            return;
                        }
                        const double bpm_median = (60.0 * fps) / stats.median_interval;
                        const double bpm_mean = (60.0 * fps) / stats.mean_interval;
                        const double interval_cv = stats.mean_interval > 0.0
                            ? (stats.stdev_interval / stats.mean_interval)
                            : 0.0;
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
                    print_stats("tempo_peaks", tempo_stats);
                    if (!downbeat_peaks.empty()) {
                        print_stats("downbeat_peaks", downbeat_stats);
                    }
                    print_stats("decoded_beats", decoded_stats);
                    print_stats("decoded_beats_filtered", decoded_filtered_stats);
                    if (short_interval_threshold > 0.0) {
                        std::cerr << "DBN stats: filter_threshold=" << short_interval_threshold
                                  << " min_interval=" << min_interval_frames << "\n";
                    }
                }
                const double bpm_from_fit =
                    detail::bpm_from_linear_fit(filtered_beats, fps);
                const double bpm_from_global_fit =
                    detail::bpm_from_global_fit(result,
                                                config,
                                                calmdad_decoder,
                                                fps,
                                                min_bpm,
                                                max_bpm,
                                                used_frames);
                const bool quality_low =
                    quality_valid && (quality_qkur < 3.6);
                const bool drop_global = false;
                const bool drop_fit = quality_low && bpm_from_fit > 0.0;
                const std::size_t downbeat_count = downbeat_stats.count;
                const double downbeat_cv = (downbeat_count > 0 && downbeat_stats.mean_interval > 0.0)
                    ? (downbeat_stats.stdev_interval / downbeat_stats.mean_interval)
                    : 0.0;
                const bool downbeat_override_ok =
                    !quality_low && downbeat_count >= 6 && downbeat_cv <= 0.25;
                const double ref_downbeat_ratio =
                    (downbeat_override_ok && bpm_from_downbeats > 0.0)
                        ? (std::abs(reference_bpm - bpm_from_downbeats) / bpm_from_downbeats)
                        : 0.0;
                const bool ref_mismatch =
                    downbeat_override_ok && bpm_from_downbeats > 0.0 && ref_downbeat_ratio > 0.005;
                const bool drop_ref = (quality_low || ref_mismatch) && reference_bpm > 0.0f;
                const bool allow_reference_grid_bpm =
                    reference_bpm > 0.0f &&
                    ((static_cast<double>(max_bpm) - static_cast<double>(min_bpm)) <=
                     std::max(2.0, static_cast<double>(reference_bpm) * 0.05));
                bool global_fit_plausible = false;
                if (bpm_from_global_fit > 0.0 && bpm_from_fit > 0.0) {
                    const double diff = std::abs(bpm_from_global_fit - bpm_from_fit);
                    const double rel_diff = diff / bpm_from_fit;
                    global_fit_plausible = rel_diff <= 0.08;
                    if (!global_fit_plausible && config.verbose) {
                        std::cerr << "DBN grid: rejecting global_fit bpm=" << bpm_from_global_fit
                                  << " fit_bpm=" << bpm_from_fit
                                  << " rel_diff=" << rel_diff << "\n";
                    }
                }
                double bpm_for_grid = 0.0;
                std::string bpm_source = "none";
                if (global_fit_plausible) {
                    bpm_for_grid = bpm_from_global_fit;
                    bpm_source = "global_fit";
                } else if (allow_reference_grid_bpm && !quality_low && !ref_mismatch) {
                    bpm_for_grid = reference_bpm;
                    bpm_source = "reference";
                } else if (downbeat_override_ok && bpm_from_downbeats > 0.0) {
                    if (bpm_from_fit > 0.0) {
                        bpm_for_grid = bpm_from_fit;
                        bpm_source = "fit_primary";
                    } else {
                        bpm_for_grid = bpm_from_downbeats;
                        bpm_source = "downbeats_primary";
                    }
                } else if (!quality_low && bpm_from_peaks_reg_full > 0.0) {
                    bpm_for_grid = bpm_from_peaks_reg_full;
                    bpm_source = "peaks_reg_full";
                } else if (!quality_low && bpm_from_fit > 0.0) {
                    bpm_for_grid = bpm_from_fit;
                    bpm_source = "fit";
                } else if (bpm_from_peaks_median > 0.0) {
                    bpm_for_grid = bpm_from_peaks_median;
                    bpm_source = "peaks_median";
                } else if (bpm_from_peaks > 0.0) {
                    bpm_for_grid = bpm_from_peaks;
                    bpm_source = "peaks";
                }
                if (bpm_for_grid <= 0.0 && allow_reference_grid_bpm) {
                    bpm_for_grid = reference_bpm;
                    bpm_source = "reference_fallback";
                }
                const double bpm_before_downbeat = bpm_for_grid;
                const std::string bpm_source_before_downbeat = bpm_source;
                if (downbeat_override_ok && bpm_from_downbeats > 0.0 && bpm_for_grid > 0.0 &&
                    bpm_source != "peaks_reg_full" &&
                    bpm_source != "downbeats_primary" &&
                    bpm_source != "fit_primary") {
                    const double ratio =
                        std::abs(bpm_from_downbeats - bpm_for_grid) / bpm_for_grid;
                    if (ratio <= 0.005) {
                        bpm_for_grid = bpm_from_downbeats;
                        bpm_source = "downbeats_override";
                    }
                }
                if (bpm_for_grid <= 0.0 && decoded.bpm > 0.0) {
                    bpm_for_grid = decoded.bpm;
                    bpm_source = "decoded";
                }
                if (bpm_for_grid <= 0.0 && base_interval > 0.0) {
                    bpm_for_grid = (60.0 * fps) / base_interval;
                    bpm_source = "base_interval";
                }
                projected_bpm = bpm_for_grid;
                double step_frames =
                    (bpm_for_grid > 0.0) ? (60.0 * fps) / bpm_for_grid : base_interval;
                if (config.verbose) {
                    std::cerr << "DBN grid: bpm=" << decoded.bpm
                              << " bpm_from_fit=" << bpm_from_fit
                              << " bpm_from_global_fit=" << bpm_from_global_fit
                              << " bpm_from_peaks=" << bpm_from_peaks
                              << " bpm_from_peaks_median=" << bpm_from_peaks_median
                              << " bpm_from_peaks_reg=" << bpm_from_peaks_reg
                              << " bpm_from_peaks_median_full=" << bpm_from_peaks_median_full
                              << " bpm_from_peaks_reg_full=" << bpm_from_peaks_reg_full
                              << " bpm_from_downbeats=" << bpm_from_downbeats
                              << " bpm_from_downbeats_median=" << bpm_from_downbeats_median
                              << " bpm_from_downbeats_reg=" << bpm_from_downbeats_reg
                              << " base_interval=" << base_interval
                              << " bpm_reference=" << reference_bpm
                              << " quality_qpar=" << quality_qpar
                              << " quality_qkur=" << quality_qkur
                              << " quality_low=" << (quality_low ? 1 : 0)
                              << " bpm_for_grid=" << bpm_for_grid
                              << " step_frames=" << step_frames
                              << " start_frame=" << decoded.beat_frames.front()
                              << "\n";
                }
                if (config.dbn_trace) {
                    std::cerr << "DBN quality gate: low=" << (quality_low ? 1 : 0)
                              << " drop_ref=" << (drop_ref ? 1 : 0)
                              << " drop_global=" << (drop_global ? 1 : 0)
                              << " drop_fit=" << (drop_fit ? 1 : 0)
                              << " downbeat_ok=" << (downbeat_override_ok ? 1 : 0)
                              << " downbeat_cv=" << downbeat_cv
                              << " downbeat_count=" << downbeat_count
                              << " used=" << bpm_source
                              << " pre_override=" << bpm_source_before_downbeat
                              << " pre_bpm=" << bpm_before_downbeat
                              << "\n";
                }
                std::size_t earliest_peak = decoded.beat_frames.front();
                std::size_t earliest_downbeat_peak = decoded.beat_frames.front();
                float earliest_downbeat_value = 0.0f;
                std::size_t strongest_peak = decoded.beat_frames.front();
                float strongest_peak_value = -1.0f;
                const float activation_floor =
                    std::max(0.01f, config.activation_threshold * 0.1f);
                if (base_interval > 1.0 && !decoded.beat_frames.empty()) {
                    const std::size_t peak_search_start = use_window ? window_start : 0;
                    const std::size_t peak_search_end = use_window
                        ? std::min<std::size_t>(
                              used_frames - 1,
                              window_start + static_cast<std::size_t>(
                                  std::llround(base_interval)))
                        : std::min<std::size_t>(
                              used_frames - 1,
                              static_cast<std::size_t>(std::llround(base_interval)));
                    if (!result.beat_activation.empty()) {
                        if (peak_search_start + 1 <= peak_search_end) {
                            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                                const float prev = result.beat_activation[i - 1];
                                const float curr = result.beat_activation[i];
                                const float next = result.beat_activation[i + 1];
                                if (curr >= activation_floor && curr >= prev && curr >= next) {
                                    earliest_peak = i;
                                    break;
                                }
                            }
                        }
                        if (peak_search_start + 1 <= peak_search_end) {
                            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                                const float prev = result.beat_activation[i - 1];
                                const float curr = result.beat_activation[i];
                                const float next = result.beat_activation[i + 1];
                                if (curr >= activation_floor && curr >= prev && curr >= next) {
                                    if (curr > strongest_peak_value) {
                                        strongest_peak_value = curr;
                                        strongest_peak = i;
                                    }
                                }
                            }
                        }
                        if (strongest_peak_value < 0.0f &&
                            earliest_peak < result.beat_activation.size()) {
                            strongest_peak = earliest_peak;
                            strongest_peak_value = result.beat_activation[earliest_peak];
                        }
                    }
                    if (!result.downbeat_activation.empty()) {
                        float max_downbeat = 0.0f;
                        for (std::size_t i = peak_search_start; i <= peak_search_end; ++i) {
                            max_downbeat = std::max(max_downbeat, result.downbeat_activation[i]);
                        }
                        const float onset_threshold =
                            std::max(activation_floor,
                                     max_downbeat * config.dbn_downbeat_phase_peak_ratio);
                        for (std::size_t i = peak_search_start; i <= peak_search_end; ++i) {
                            const float curr = result.downbeat_activation[i];
                            if (curr >= onset_threshold) {
                                earliest_downbeat_peak = i;
                                earliest_downbeat_value = curr;
                                break;
                            }
                        }
                    }
                    if (config.verbose) {
                        std::cerr << "DBN grid: earliest_peak=" << earliest_peak
                                  << " earliest_downbeat_peak=" << earliest_downbeat_peak
                                  << " earliest_downbeat_value=" << earliest_downbeat_value
                                  << " strongest_peak=" << strongest_peak
                                  << " strongest_peak_value=" << strongest_peak_value
                                  << " activation_floor=" << activation_floor
                                  << "\n";
                    }
                    std::size_t start_peak = decoded.beat_frames.front();
                    if (!result.beat_activation.empty()) {
                        std::size_t earliest = start_peak;
                        if (peak_search_start + 1 <= peak_search_end) {
                            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                                const float prev = result.beat_activation[i - 1];
                                const float curr = result.beat_activation[i];
                                const float next = result.beat_activation[i + 1];
                                if (curr >= activation_floor && curr >= prev && curr >= next) {
                                    earliest = i;
                                    break;
                                }
                            }
                        }
                        if (earliest < start_peak) {
                            start_peak = earliest;
                        }
                        if (config.dbn_grid_start_strong_peak &&
                            strongest_peak_value >= activation_floor) {
                            start_peak = strongest_peak;
                        }
                    }
                    std::vector<std::size_t> forward =
                        fill_peaks_with_grid(result.beat_activation,
                                             start_peak,
                                             used_frames - 1,
                                             base_interval,
                                             activation_floor);
                    std::vector<std::size_t> backward;
                    double cursor = static_cast<double>(start_peak) - base_interval;
                    const std::size_t window = static_cast<std::size_t>(
                        std::max(1.0, std::round(base_interval * 0.25)));
                    while (cursor >= 0.0) {
                        const std::size_t center = static_cast<std::size_t>(std::llround(cursor));
                        const std::size_t start = center > window ? center - window : 0;
                        const std::size_t end = std::min(used_frames - 1, center + window);
                        float best_value = -1.0f;
                        std::size_t best_index = center;
                        for (std::size_t k = start; k <= end; ++k) {
                            const float value = result.beat_activation[k];
                            if (value > best_value) {
                                best_value = value;
                                best_index = k;
                            }
                        }
                        std::size_t chosen = best_index;
                        if (best_value < activation_floor) {
                            chosen = center;
                        }
                        backward.push_back(chosen);
                        if (cursor < base_interval) {
                            break;
                        }
                        cursor -= base_interval;
                    }
                    std::sort(backward.begin(), backward.end());
                    std::vector<std::size_t> combined;
                    combined.reserve(backward.size() + forward.size());
                    combined.insert(combined.end(), backward.begin(), backward.end());
                    if (!forward.empty() && (combined.empty() || combined.back() != forward.front())) {
                        combined.insert(combined.end(), forward.begin(), forward.end());
                    } else if (forward.size() > 1) {
                        combined.insert(combined.end(), forward.begin() + 1, forward.end());
                    }
                    decoded.beat_frames = std::move(combined);
                }

                std::size_t best_phase = phase;
                double best_score = -std::numeric_limits<double>::infinity();
                const double local_frame_rate =
                    config.hop_size > 0 ? static_cast<double>(config.sample_rate) /
                        static_cast<double>(config.hop_size) : 0.0;
                const std::size_t phase_window_frames =
                    (config.dbn_downbeat_phase_window_seconds > 0.0 && local_frame_rate > 0.0)
                        ? static_cast<std::size_t>(std::llround(
                            config.dbn_downbeat_phase_window_seconds * local_frame_rate))
                        : 0;
                const std::size_t phase_window_start = use_window ? window_start : 0;
                const std::size_t phase_window_end =
                    (phase_window_frames > 0)
                        ? std::min(used_frames, phase_window_start + phase_window_frames)
                        : phase_window_start;
                const bool allow_downbeat_phase =
                    !result.downbeat_activation.empty() && !quality_low && downbeat_override_ok;
                const std::size_t max_delay_frames =
                    (config.dbn_downbeat_phase_max_delay_seconds > 0.0 && local_frame_rate > 0.0)
                        ? static_cast<std::size_t>(std::llround(
                            config.dbn_downbeat_phase_max_delay_seconds * local_frame_rate))
                        : 0;
                const float onset_ratio = 0.35f;
                const std::size_t onset_max_back =
                    max_delay_frames > 0 ? max_delay_frames : 8;
                auto onset_from_peak = [&](const std::vector<float>& activation,
                                           std::size_t peak_frame) -> std::size_t {
                    if (activation.empty() || peak_frame >= activation.size()) {
                        return peak_frame;
                    }
                    const float peak_value = activation[peak_frame];
                    if (peak_value <= 0.0f) {
                        return peak_frame;
                    }
                    const float threshold = peak_value * onset_ratio;
                    std::size_t frame = peak_frame;
                    std::size_t steps = 0;
                    while (frame > 0 && steps < onset_max_back) {
                        if (activation[frame] < threshold) {
                            return frame + 1;
                        }
                        --frame;
                        ++steps;
                    }
                    return frame;
                };
                auto build_onset_frames = [&](const std::vector<std::size_t>& frames,
                                              const std::vector<float>& activation) {
                    std::vector<std::size_t> out;
                    out.reserve(frames.size());
                    for (std::size_t frame : frames) {
                        out.push_back(onset_from_peak(activation, frame));
                    }
                    dedupe_frames(out);
                    return out;
                };

                float max_downbeat = 0.0f;
                float max_beat = 0.0f;
                std::vector<uint8_t> phase_peak_mask;
                bool has_phase_peaks = false;
                if (phase_window_frames > 0) {
                    for (std::size_t i = phase_window_start; i < phase_window_end; ++i) {
                        if (allow_downbeat_phase) {
                            max_downbeat = std::max(max_downbeat, result.downbeat_activation[i]);
                        }
                        if (i < result.beat_activation.size()) {
                            max_beat = std::max(max_beat, result.beat_activation[i]);
                        }
                    }
                    phase_peak_mask.assign(used_frames, 0);
                    if (max_beat > 0.0f && phase_window_end > phase_window_start + 2) {
                        const float beat_threshold =
                            static_cast<float>(max_beat * config.dbn_downbeat_phase_peak_ratio);
                        const float peak_eps = std::max(1e-6f, max_beat * 0.01f);
                        for (std::size_t i = phase_window_start + 1; i + 1 < phase_window_end; ++i) {
                            const float prev = result.beat_activation[i - 1];
                            const float curr = result.beat_activation[i];
                            const float next = result.beat_activation[i + 1];
                            if (curr >= beat_threshold &&
                                curr >= prev + peak_eps &&
                                curr >= next + peak_eps) {
                                const std::size_t onset_frame =
                                    onset_from_peak(result.beat_activation, i);
                                if (onset_frame < phase_peak_mask.size()) {
                                    phase_peak_mask[onset_frame] = 1;
                                }
                                has_phase_peaks = true;
                            }
                        }
                    }
                    if (config.dbn_trace && allow_downbeat_phase) {
                        const std::size_t limit =
                            phase_window_end > phase_window_start
                                ? (phase_window_end - phase_window_start)
                                : 0;
                        const std::size_t beat_preview =
                            std::min<std::size_t>(12, result.beat_activation.size());
                        std::cerr << "DBN: beat head:";
                        for (std::size_t i = 0; i < beat_preview; ++i) {
                            std::cerr << " " << i << "->" << result.beat_activation[i];
                        }
                        std::cerr << "\n";
                        const std::size_t preview =
                            std::min<std::size_t>(12, result.downbeat_activation.size());
                        std::cerr << "DBN: downbeat head:";
                        for (std::size_t i = 0; i < preview; ++i) {
                            std::cerr << " " << i << "->" << result.downbeat_activation[i];
                        }
                        std::cerr << "\n";
                        std::cerr << "DBN: downbeat max=" << max_downbeat
                                  << " beat max=" << max_beat
                                  << " activation_floor=" << activation_floor
                                  << "\n";
                        struct Peak {
                            float value;
                            std::size_t frame;
                        };
                        std::vector<Peak> peaks;
                        const std::size_t start =
                            (phase_window_start + 1 < phase_window_end)
                                ? (phase_window_start + 1)
                                : phase_window_start;
                        const std::size_t end =
                            phase_window_end > 1 ? phase_window_end - 1 : phase_window_end;
                        for (std::size_t i = start; i <= end; ++i) {
                            const float prev = result.downbeat_activation[i - 1];
                            const float curr = result.downbeat_activation[i];
                            const float next = result.downbeat_activation[i + 1];
                            if (curr >= prev && curr >= next) {
                                peaks.push_back({curr, i});
                            }
                        }
                        std::sort(peaks.begin(), peaks.end(),
                                  [](const Peak& a, const Peak& b) { return a.value > b.value; });
                        std::cerr << "DBN: downbeat peaks (phase window "
                                  << phase_window_start << "-" << phase_window_end << "):";
                        const std::size_t top = std::min<std::size_t>(5, peaks.size());
                        for (std::size_t i = 0; i < top; ++i) {
                            std::cerr << " " << peaks[i].frame << "->" << peaks[i].value;
                        }
                        std::cerr << "\n";
                        std::vector<Peak> global_peaks;
                        const std::size_t global_end =
                            result.downbeat_activation.empty() ? 0 : used_frames > 1
                                ? used_frames - 1 : 0;
                        for (std::size_t i = start; i <= global_end; ++i) {
                            const float prev = result.downbeat_activation[i - 1];
                            const float curr = result.downbeat_activation[i];
                            const float next = result.downbeat_activation[i + 1];
                            if (curr >= prev && curr >= next) {
                                global_peaks.push_back({curr, i});
                            }
                        }
                        std::sort(global_peaks.begin(), global_peaks.end(),
                                  [](const Peak& a, const Peak& b) { return a.value > b.value; });
                        std::cerr << "DBN: downbeat peaks (global top):";
                        const std::size_t global_top = std::min<std::size_t>(6, global_peaks.size());
                        for (std::size_t i = 0; i < global_top; ++i) {
                            const std::size_t frame = global_peaks[i].frame;
                            const double time_s = fps > 0.0 ? static_cast<double>(frame) / fps : 0.0;
                            std::cerr << " " << frame << "(" << time_s << "s)"
                                      << "->" << global_peaks[i].value;
                        }
                        std::cerr << "\n";
                        std::cerr << "DBN: phase peaks for selection (beat-only, strict): "
                                  << (has_phase_peaks ? "picked" : "none")
                                  << " (limit=" << limit << ")\n";
                    }
                }
                const float downbeat_threshold =
                    max_downbeat > 0.0f
                        ? static_cast<float>(max_downbeat * config.dbn_downbeat_phase_peak_ratio)
                        : 0.0f;
                struct PhaseDebug {
                    std::size_t phase = 0;
                    double score = -std::numeric_limits<double>::infinity();
                    std::size_t first_frame = 0;
                    std::size_t hits = 0;
                    double mean = 0.0;
                    double delay_penalty = 0.0;
                    const char* source = "none";
                };
                std::vector<PhaseDebug> phase_debug;
                if (config.dbn_trace) {
                    phase_debug.reserve(bpb);
                }
                for (std::size_t candidate_phase = 0; candidate_phase < bpb; ++candidate_phase) {
                    const auto projected =
                        project_downbeats_from_beats(decoded.beat_frames, bpb, candidate_phase);
                    if (projected.empty()) {
                        continue;
                    }
                    const std::vector<float>& onset_activation =
                        allow_downbeat_phase
                            ? result.downbeat_activation
                            : result.beat_activation;
                    const auto projected_onsets =
                        build_onset_frames(projected, onset_activation);
                    const auto& phase_frames =
                        projected_onsets.empty() ? projected : projected_onsets;
                    double score = -std::numeric_limits<double>::infinity();
                    std::size_t hits = 0;
                    double mean = 0.0;
                    const char* source = "none";
                    bool score_set = false;

                    if (phase_window_frames > 0 && !allow_downbeat_phase) {
                        if (has_phase_peaks) {
                            double sum = 0.0;
                            double weight = 0.0;
                            for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                                const std::size_t frame = phase_frames[i];
                                if (frame >= phase_window_frames) {
                                    break;
                                }
                                if (frame < phase_peak_mask.size() && phase_peak_mask[frame]) {
                                    const double value =
                                        (frame < result.beat_activation.size())
                                            ? static_cast<double>(result.beat_activation[frame])
                                            : 0.0;
                                    sum += value;
                                    weight += 1.0;
                                }
                            }
                            if (weight > 0.0) {
                                score = (sum / weight);
                                hits = static_cast<std::size_t>(weight);
                                mean = score;
                                source = "beat_peak_mask";
                            }
                        } else {
                            double sum = 0.0;
                            double weight = 0.0;
                            const float threshold = max_beat > 0.0f
                                ? static_cast<float>(
                                    max_beat * config.dbn_downbeat_phase_peak_ratio)
                                : 0.0f;
                            for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                                const std::size_t frame = phase_frames[i];
                                if (frame >= phase_window_frames) {
                                    break;
                                }
                                if (frame < result.beat_activation.size()) {
                                    const float value = result.beat_activation[frame];
                                    if (value >= threshold) {
                                        sum += static_cast<double>(value);
                                        weight += 1.0;
                                    }
                                }
                            }
                            if (weight > 0.0) {
                                score = (sum / weight);
                                hits = static_cast<std::size_t>(weight);
                                mean = score;
                                source = "beat_threshold";
                            }
                        }
                    } else if (!score_set && phase_window_frames > 0 && allow_downbeat_phase) {
                        double sum = 0.0;
                        double weight = 0.0;
                        const float threshold = max_downbeat > 0.0f
                            ? static_cast<float>(
                                max_downbeat * config.dbn_downbeat_phase_peak_ratio)
                            : 0.0f;
                        const std::size_t end =
                            std::min(phase_window_end, result.downbeat_activation.size());
                        const double decay = std::max(1.0, phase_window_frames * 0.2);
                        for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                            const std::size_t frame = phase_frames[i];
                            if (frame >= end) {
                                break;
                            }
                            const float value = result.downbeat_activation[frame];
                            if (value >= threshold) {
                                const double w = std::exp(-static_cast<double>(frame) / decay);
                                sum += static_cast<double>(value) * w;
                                weight += w;
                            }
                        }
                        if (weight > 0.0) {
                            score = (sum / weight);
                            hits = static_cast<std::size_t>(weight + 0.5);
                            mean = score;
                            source = "downbeat_window_decay";
                            score_set = true;
                        }
                    }
                    if (!score_set && !std::isfinite(score)) {
                        const std::size_t max_checks = std::min<std::size_t>(phase_frames.size(), 3);
                        double sum = 0.0;
                        if (allow_downbeat_phase) {
                            const float threshold = max_downbeat > 0.0f
                                ? static_cast<float>(
                                    max_downbeat * config.dbn_downbeat_phase_peak_ratio)
                                : 0.0f;
                            for (std::size_t i = 0; i < max_checks; ++i) {
                                const std::size_t frame = phase_frames[i];
                                if (frame < result.downbeat_activation.size()) {
                                    const float value = result.downbeat_activation[frame];
                                    if (value >= threshold) {
                                        sum += value;
                                        hits += 1;
                                    }
                                }
                            }
                            if (hits == 0) {
                                for (std::size_t i = 0; i < max_checks; ++i) {
                                    const std::size_t frame = phase_frames[i];
                                    if (frame < result.downbeat_activation.size()) {
                                        sum += result.downbeat_activation[frame];
                                    }
                                }
                                hits = max_checks;
                            }
                        } else {
                            for (std::size_t i = 0; i < max_checks; ++i) {
                                const std::size_t frame = phase_frames[i];
                                if (frame < result.beat_activation.size()) {
                                    sum += result.beat_activation[frame];
                                }
                            }
                            hits = max_checks;
                        }
                        const double penalty = 0.01 * static_cast<double>(phase_frames.front());
                        score = sum - penalty;
                        mean = (max_checks > 0) ? (sum / static_cast<double>(max_checks)) : 0.0;
                        source = allow_downbeat_phase ? "fallback_downbeat" : "fallback_beat";
                    }
                    if (!score_set && !std::isfinite(score) && allow_downbeat_phase) {
                        const std::size_t early_limit = std::min<std::size_t>(
                            phase_window_end,
                            static_cast<std::size_t>(std::max(1.0, local_frame_rate * 2.0)));
                        double sum = 0.0;
                        std::size_t count = 0;
                        for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                            const std::size_t frame = phase_frames[i];
                            if (frame >= early_limit) {
                                break;
                            }
                            if (frame < result.downbeat_activation.size()) {
                                sum += result.downbeat_activation[frame];
                                count += 1;
                            }
                        }
                        if (count > 0) {
                            score = sum / static_cast<double>(count);
                            hits = count;
                            mean = score;
                            source = "downbeat_early_energy";
                            score_set = true;
                        }
                    }
                    double delay_penalty = 0.0;
                    if (max_delay_frames > 0 && phase_frames.front() > max_delay_frames &&
                        source != nullptr && std::strncmp(source, "fallback", 8) != 0 &&
                        !(quality_low &&
                          (std::strncmp(source, "beat_peak_mask", 14) == 0 ||
                           std::strncmp(source, "beat_threshold", 14) == 0))) {
                        const double delay = static_cast<double>(phase_frames.front() - max_delay_frames);
                        delay_penalty = delay * 1000.0;
                        score -= delay_penalty;
                    }
                    if (score > best_score) {
                        best_score = score;
                        best_phase = candidate_phase;
                    }
                    if (config.dbn_trace) {
                        PhaseDebug entry;
                        entry.phase = candidate_phase;
                        entry.score = score;
                        entry.first_frame = phase_frames.front();
                        entry.hits = hits;
                        entry.mean = mean;
                        entry.delay_penalty = delay_penalty;
                        entry.source = source;
                        phase_debug.push_back(entry);
                    }
                }
                if (config.verbose) {
                    std::cerr << "DBN: phase_window_frames=" << phase_window_frames
                              << " max_downbeat=" << max_downbeat
                              << " threshold=" << downbeat_threshold
                              << " best_phase=" << best_phase
                              << " best_score=" << best_score
                              << "\n";
                }
                if (config.dbn_trace && !phase_debug.empty()) {
                    std::vector<PhaseDebug> sorted = phase_debug;
                    std::sort(sorted.begin(), sorted.end(),
                              [](const PhaseDebug& a, const PhaseDebug& b) {
                                  return a.score > b.score;
                              });
                    const PhaseDebug& top = sorted.front();
                    const PhaseDebug* runner = (sorted.size() > 1) ? &sorted[1] : nullptr;
                    std::cerr << "DBN: phase winner="
                              << top.phase
                              << " score=" << top.score
                              << " first=" << top.first_frame
                              << " hits=" << top.hits
                              << " mean=" << top.mean
                              << " penalty=" << top.delay_penalty
                              << " src=" << top.source;
                    if (runner) {
                        std::cerr << " runner=" << runner->phase
                                  << " score=" << runner->score
                                  << " first=" << runner->first_frame
                                  << " hits=" << runner->hits
                                  << " mean=" << runner->mean
                                  << " penalty=" << runner->delay_penalty
                                  << " src=" << runner->source;
                    }
                    std::cerr << "\n";
                    std::cerr << "DBN: phase candidates:";
                    for (const auto& entry : phase_debug) {
                        std::cerr << " p" << entry.phase
                                  << " score=" << entry.score
                                  << " first=" << entry.first_frame
                                  << " hits=" << entry.hits
                                  << " mean=" << entry.mean
                                  << " penalty=" << entry.delay_penalty
                                  << " src=" << entry.source;
                    }
                    std::cerr << "\n";
                }
                decoded.downbeat_frames =
                    project_downbeats_from_beats(decoded.beat_frames, bpb, best_phase);
                if (config.dbn_trace) {
                    const std::size_t preview =
                        std::min<std::size_t>(6, decoded.downbeat_frames.size());
                    std::cerr << "DBN: downbeat frames head:";
                    for (std::size_t i = 0; i < preview; ++i) {
                        const std::size_t frame = decoded.downbeat_frames[i];
                        const double time_s = fps > 0.0 ? static_cast<double>(frame) / fps : 0.0;
                        std::cerr << " " << frame << "(" << time_s << "s)";
                    }
                    std::cerr << "\n";
                    if (!decoded.downbeat_frames.empty()) {
                        const std::size_t first_frame = decoded.downbeat_frames.front();
                        const double first_time =
                            fps > 0.0 ? static_cast<double>(first_frame) / fps : 0.0;
                        std::cerr << "DBN: downbeat selection start="
                                  << first_frame << " (" << first_time << "s)"
                                  << " bpb=" << bpb
                                  << " phase=" << best_phase
                                  << " score=" << best_score
                                  << "\n";
                    }
                }
                // Force a uniform grid so projection yields evenly spaced beats.
                if (decoded.beat_frames.size() >= 2 && step_frames > 1.0) {
                    if (config.dbn_grid_global_fit && decoded.beat_frames.size() >= 8) {
                        const double n = static_cast<double>(decoded.beat_frames.size());
                        double sx = 0.0;
                        double sy = 0.0;
                        double sxx = 0.0;
                        double sxy = 0.0;
                        for (std::size_t i = 0; i < decoded.beat_frames.size(); ++i) {
                            const double x = static_cast<double>(i);
                            const double y = static_cast<double>(decoded.beat_frames[i]);
                            sx += x;
                            sy += y;
                            sxx += x * x;
                            sxy += x * y;
                        }
                        const double den = n * sxx - sx * sx;
                        if (std::abs(den) > 1e-9) {
                            const double fit_step = (n * sxy - sx * sy) / den;
                            if (fit_step > 1.0) {
                                if (config.verbose) {
                                    std::cerr << "DBN grid fit: step_frames(raw)=" << step_frames
                                              << " step_frames(fit)=" << fit_step
                                              << " beats=" << decoded.beat_frames.size()
                                              << "\n";
                                }
                                step_frames = fit_step;
                            }
                        }
                    }
                    const bool have_downbeat_start = !decoded.downbeat_frames.empty();
                    const bool reliable_downbeat_start =
                        have_downbeat_start &&
                        max_downbeat > activation_floor;
                    const std::size_t start = reliable_downbeat_start
                        ? decoded.downbeat_frames.front()
                        : std::min(decoded.beat_frames.front(),
                                   std::min(earliest_peak, earliest_downbeat_peak));
                    double grid_start = static_cast<double>(start);
                    if (earliest_downbeat_peak > 0 && earliest_downbeat_peak < start) {
                        grid_start = static_cast<double>(earliest_downbeat_peak);
                    }
                    if (!reliable_downbeat_start &&
                        config.dbn_grid_start_strong_peak &&
                        strongest_peak_value >= activation_floor) {
                        grid_start = static_cast<double>(strongest_peak);
                    }
                    if (!reliable_downbeat_start &&
                        config.dbn_grid_align_downbeat_peak &&
                        earliest_downbeat_peak > 0 &&
                        step_frames > 1.0) {
                        const double offset =
                            static_cast<double>(earliest_downbeat_peak) -
                            grid_start;
                        const double half_step = step_frames * 0.5;
                        if (std::abs(offset) <= half_step) {
                            const double adjusted = grid_start + offset;
                            if (adjusted >= 0.0) {
                                grid_start = adjusted;
                            }
                        } else if (earliest_downbeat_peak < grid_start) {
                            // If the first downbeat peak is clearly earlier, bias the grid to it.
                            grid_start = static_cast<double>(earliest_downbeat_peak);
                        }
                    }
                    if (config.dbn_grid_start_advance_seconds > 0.0f &&
                        fps > 0.0) {
                        const double frames_per_second = fps;
                        grid_start -= static_cast<double>(config.dbn_grid_start_advance_seconds) *
                            frames_per_second;
                    }
                    if (!reliable_downbeat_start && step_frames > 1.0 &&
                        result.beat_activation.size() >= 8) {
                        const auto phase_score = [&](double start_frame) -> double {
                            if (start_frame < 0.0) {
                                return -1.0;
                            }
                            double score = 0.0;
                            std::size_t hits = 0;
                            double cursor_local = start_frame;
                            while (cursor_local >= step_frames) {
                                cursor_local -= step_frames;
                            }
                            while (cursor_local < static_cast<double>(used_frames) && hits < 128) {
                                const long long idx_ll = static_cast<long long>(std::llround(cursor_local));
                                if (idx_ll >= 0 &&
                                    static_cast<std::size_t>(idx_ll) < result.beat_activation.size()) {
                                    const std::size_t idx = static_cast<std::size_t>(idx_ll);
                                    float value = result.beat_activation[idx];
                                    if (idx > 0) {
                                        value = std::max(value, result.beat_activation[idx - 1]);
                                    }
                                    if (idx + 1 < result.beat_activation.size()) {
                                        value = std::max(value, result.beat_activation[idx + 1]);
                                    }
                                    score += static_cast<double>(value);
                                    hits += 1;
                                }
                                cursor_local += step_frames;
                            }
                            return hits > 0 ? (score / static_cast<double>(hits)) : -1.0;
                        };
                        const double alt_start = grid_start + (0.5 * step_frames);
                        const double base_score = phase_score(grid_start);
                        const double alt_score = phase_score(alt_start);
                        if (alt_score > base_score) {
                            grid_start = alt_start;
                            if (config.verbose) {
                                std::cerr << "DBN grid: half-step phase shift selected"
                                          << " base_score=" << base_score
                                          << " alt_score=" << alt_score
                                          << "\n";
                            }
                        }
                    }
                    if (grid_start < 0.0) {
                        grid_start = 0.0;
                    }
                    if (step_frames > 1.0 && result.beat_activation.size() >= 64) {
                        std::vector<std::size_t> beat_peaks;
                        beat_peaks.reserve(result.beat_activation.size() / 8);
                        if (result.beat_activation.size() >= 3) {
                            const std::size_t end = result.beat_activation.size() - 1;
                            for (std::size_t i = 1; i < end; ++i) {
                                const float prev = result.beat_activation[i - 1];
                                const float curr = result.beat_activation[i];
                                const float next = result.beat_activation[i + 1];
                                if (curr >= activation_floor && curr >= prev && curr >= next) {
                                    beat_peaks.push_back(i);
                                }
                            }
                        }
                        if (beat_peaks.size() >= 16) {
                            struct OffsetSample {
                                std::size_t beat_index = 0;
                                double offset = 0.0;
                            };
                            std::vector<OffsetSample> samples;
                            samples.reserve(256);
                            double cursor_fit = grid_start;
                            std::size_t beat_index = 0;
                            while (cursor_fit < static_cast<double>(used_frames) &&
                                   beat_index < 512) {
                                const long long frame_ll =
                                    static_cast<long long>(std::llround(cursor_fit));
                                if (frame_ll >= 0) {
                                    const std::size_t frame = static_cast<std::size_t>(frame_ll);
                                    auto it = std::lower_bound(beat_peaks.begin(), beat_peaks.end(), frame);
                                    std::size_t nearest = beat_peaks.front();
                                    if (it == beat_peaks.end()) {
                                        nearest = beat_peaks.back();
                                    } else {
                                        nearest = *it;
                                        if (it != beat_peaks.begin()) {
                                            const std::size_t prev = *(it - 1);
                                            if (frame - prev < nearest - frame) {
                                                nearest = prev;
                                            }
                                        }
                                    }
                                    const double max_dist = step_frames * 0.45;
                                    const double dist =
                                        std::abs(static_cast<double>(nearest) -
                                                 static_cast<double>(frame));
                                    if (dist <= max_dist) {
                                        OffsetSample sample;
                                        sample.beat_index = beat_index;
                                        sample.offset =
                                            static_cast<double>(nearest) -
                                            static_cast<double>(frame);
                                        samples.push_back(sample);
                                    }
                                }
                                cursor_fit += step_frames;
                                beat_index += 1;
                            }
                            if (samples.size() >= 32) {
                                auto median_of_offsets = [](const std::vector<OffsetSample>& src,
                                                            std::size_t begin,
                                                            std::size_t count) {
                                    std::vector<double> values;
                                    values.reserve(count);
                                    const std::size_t end = std::min(src.size(), begin + count);
                                    for (std::size_t i = begin; i < end; ++i) {
                                        values.push_back(src[i].offset);
                                    }
                                    if (values.empty()) {
                                        return 0.0;
                                    }
                                    auto mid =
                                        values.begin() + static_cast<long>(values.size() / 2);
                                    std::nth_element(values.begin(), mid, values.end());
                                    return *mid;
                                };
                                const std::size_t edge =
                                    std::min<std::size_t>(32, samples.size() / 2);
                                const double start_offset = median_of_offsets(samples, 0, edge);
                                const std::size_t tail_begin = samples.size() - edge;
                                const double end_offset =
                                    median_of_offsets(samples, tail_begin, edge);
                                const std::size_t start_index = samples.front().beat_index;
                                const std::size_t end_index = samples.back().beat_index;
                                const double index_delta =
                                    static_cast<double>(end_index - start_index);
                                if (index_delta > 0.0) {
                                    const double step_correction =
                                        (end_offset - start_offset) / index_delta;
                                    const double max_step_correction = step_frames * 0.01;
                                    const double clamped_correction =
                                        std::clamp(step_correction,
                                                   -max_step_correction,
                                                   max_step_correction);
                                    if (std::abs(clamped_correction) > 1e-6) {
                                        step_frames += clamped_correction;
                                        grid_start += start_offset;
                                        if (grid_start < 0.0) {
                                            grid_start = 0.0;
                                        }
                                        if (config.verbose) {
                                            std::cerr << "DBN grid drift-correct:"
                                                      << " start_offset=" << start_offset
                                                      << " end_offset=" << end_offset
                                                      << " start_index=" << start_index
                                                      << " end_index=" << end_index
                                                      << " step_correction=" << step_correction
                                                      << " step_applied=" << clamped_correction
                                                      << " step_frames=" << step_frames
                                                      << " samples=" << samples.size()
                                                      << "\n";
                                        }
                                    }
                                }
                            }
                        }
                    }
                    std::vector<std::size_t> uniform_beats;
                    double cursor = grid_start;
                    while (cursor >= step_frames) {
                        cursor -= step_frames;
                        uniform_beats.push_back(
                            static_cast<std::size_t>(std::llround(cursor)));
                    }
                    std::reverse(uniform_beats.begin(), uniform_beats.end());
                    cursor = grid_start;
                    while (cursor < static_cast<double>(used_frames)) {
                        uniform_beats.push_back(
                            static_cast<std::size_t>(std::llround(cursor)));
                        cursor += step_frames;
                    }
                    dedupe_frames(uniform_beats);
                    decoded.beat_frames = std::move(uniform_beats);
                    decoded.downbeat_frames =
                        project_downbeats_from_beats(decoded.beat_frames, bpb, best_phase);
                    if (config.dbn_trace && fps > 0.0) {
                        trace_grid_peak_alignment(decoded.beat_frames,
                                                  decoded.downbeat_frames,
                                                  result.beat_activation,
                                                  result.downbeat_activation,
                                                  activation_floor,
                                                  fps);
                    }
                    if (config.verbose) {
                        std::cerr << "DBN grid: start=" << start
                                  << " grid_start=" << grid_start
                                  << " strongest_peak=" << strongest_peak
                                  << " strongest_peak_value=" << strongest_peak_value
                                  << " earliest_downbeat_peak=" << earliest_downbeat_peak
                                  << " advance_s=" << config.dbn_grid_start_advance_seconds
                                  << "\n";
                    }
                }
            }

            // Use the refined peak interpolation path for decoded DBN beats as well,
            // so sample-frame timing is not quantized to integer feature frames.
            fill_beats_from_frames(decoded.beat_frames);
            if (config.dbn_project_grid && decoded.beat_frames.size() >= 2 && projected_bpm > 0.0) {
                fill_beats_from_bpm_grid_into(decoded.beat_frames.front(),
                                              projected_bpm,
                                              grid_total_frames,
                                              result.beat_projected_feature_frames,
                                              result.beat_projected_sample_frames,
                                              result.beat_projected_strengths);
                std::vector<std::size_t> projected_frames;
                projected_frames.reserve(result.beat_projected_feature_frames.size());
                for (unsigned long long frame : result.beat_projected_feature_frames) {
                    projected_frames.push_back(static_cast<std::size_t>(frame));
                }
                std::size_t projected_bpb = std::max<std::size_t>(1, config.dbn_beats_per_bar);
                std::size_t projected_phase = 0;
                if (!decoded.downbeat_frames.empty()) {
                    const auto inferred =
                        infer_bpb_phase(decoded.beat_frames, decoded.downbeat_frames, {3, 4});
                    projected_bpb = std::max<std::size_t>(1, inferred.first);
                    projected_phase = inferred.second % projected_bpb;
                }
                projected_phase = guard_projected_downbeat_phase(projected_frames,
                                                                 result.downbeat_activation,
                                                                 projected_bpb,
                                                                 projected_phase,
                                                                 config.verbose);
                // Preserve DBN-selected bar phase on the projected beat grid.
                // Nearest-neighbor remapping can flip bars by one beat when
                // projection tempo differs slightly from the decoded beat list.
                const std::vector<std::size_t> projected_downbeats =
                    project_downbeats_from_beats(projected_frames, projected_bpb, projected_phase);
                result.downbeat_projected_feature_frames.clear();
                result.downbeat_projected_feature_frames.reserve(projected_downbeats.size());
                for (std::size_t frame : projected_downbeats) {
                    result.downbeat_projected_feature_frames.push_back(
                        static_cast<unsigned long long>(frame));
                }
            } else {
                result.beat_projected_feature_frames.clear();
                result.beat_projected_sample_frames.clear();
                result.beat_projected_strengths.clear();
                result.downbeat_projected_feature_frames.clear();
            }
            result.downbeat_feature_frames.clear();
            const std::vector<std::size_t> adjusted_downbeats =
                apply_latency_to_frames(decoded.downbeat_frames);
            result.downbeat_feature_frames.reserve(adjusted_downbeats.size());
            for (std::size_t frame : adjusted_downbeats) {
                result.downbeat_feature_frames.push_back(static_cast<unsigned long long>(frame));
            }
            if (config.profile) {
                std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                          << "ms peaks=" << peaks_ms << "ms\n";
            }
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
            std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                      << "ms peaks=" << peaks_ms << "ms\n";
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
        std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                  << "ms peaks=" << peaks_ms << "ms\n";
    }

    return result;
}

} // namespace beatit
