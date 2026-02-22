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
#include "beatit/coreml_postprocess_dbn.h"
#include "beatit/coreml_postprocess_result_ops.h"
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
        return detail::refine_frame_to_peak(frame, activation, kRefineWindow);
    };

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
                detail::dedupe_frames_tolerant(grid_frames, tolerance_frames);
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
