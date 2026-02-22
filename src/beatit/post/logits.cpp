//
//  logits.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/logits.h"

#include "beatit/logging.hpp"
#include "beatit/post/helpers.h"
#include "beatit/post/result_ops.h"
#include "beatit/post/window.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace beatit::detail {

void run_logit_consensus_postprocess(CoreMLResult& result,
                                     const std::vector<float>* phase_energy,
                                     const CoreMLConfig& config,
                                     double sample_rate,
                                     float min_bpm,
                                     float max_bpm,
                                     std::size_t grid_total_frames,
                                     double fps,
                                     double hop_scale,
                                     std::size_t analysis_latency_frames,
                                     double analysis_latency_frames_f,
                                     double dbn_ms,
                                     double peaks_ms) {
    constexpr float kPi = 3.14159265358979323846f;
    constexpr std::size_t kRefineWindow = 2;
    const std::size_t used_frames = result.beat_activation.size();

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

    auto clear_projected_grid = [&] {
        result.beat_projected_feature_frames.clear();
        result.beat_projected_sample_frames.clear();
        result.beat_projected_strengths.clear();
        result.downbeat_projected_feature_frames.clear();
    };

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
            BEATIT_LOG_DEBUG("Logit consensus: max_activation=" << max_activation
                             << " peak_threshold=" << peak_threshold
                             << " peaks=" << beat_peaks.size()
                             << " interval_frames=" << interval_frames
                             << " fps=" << fps);
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
                BEATIT_LOG_DEBUG("Logit sweep: bpm=" << bpm
                                 << " phase=" << sweep_phase
                                 << " score=" << sweep_score);
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
                clear_projected_grid();
                if (config.profile) {
                    BEATIT_LOG_INFO("Timing(postprocess): dbn=" << dbn_ms
                                    << "ms peaks=" << peaks_ms << "ms");
                }
                return;
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
            BEATIT_LOG_DEBUG("Logit consensus: bpm=" << bpm
                             << " step_frames=" << step_frames
                             << " used_frames=" << used_frames
                             << " max_shift_s=" << config.logit_phase_max_shift_seconds);
        }
        if (step_frames <= 0.0) {
            return;
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
            BEATIT_LOG_DEBUG("Logit consensus: global_phase=" << global_phase
                             << " best_score=" << best_score);
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
            if (!grid_frames.empty()) {
                BEATIT_LOG_DEBUG("Logit consensus: grid_frames=" << grid_frames.size()
                                 << " first=" << grid_frames.front()
                                 << " last=" << grid_frames.back());
            } else {
                BEATIT_LOG_DEBUG("Logit consensus: grid_frames=0");
            }
        }

        fill_beats_from_frames(grid_frames);
        if (config.verbose) {
            BEATIT_LOG_DEBUG("Logit consensus: beats_out=" << result.beat_feature_frames.size()
                             << " samples_out=" << result.beat_sample_frames.size());
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

        clear_projected_grid();

        if (config.profile) {
            BEATIT_LOG_INFO("Timing(postprocess): dbn=" << dbn_ms
                            << "ms peaks=" << peaks_ms << "ms");
        }
        return;
}

} // namespace beatit::detail
