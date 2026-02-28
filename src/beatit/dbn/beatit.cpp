//
//  beatit.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//
#include "beatit/dbn/beatit.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace beatit {

DBNDecodeResult decode_dbn_beats_beatit(const std::vector<float>& beat_activation,
                                        const std::vector<float>& downbeat_activation,
                                        double fps,
                                        float min_bpm,
                                        float max_bpm,
                                        const BeatitConfig& config,
                                        float reference_bpm) {
    DBNDecodeResult result;
    if (beat_activation.empty() || fps <= 0.0) {
        return result;
    }

    const float bpm_step = std::max(0.1f, config.dbn_bpm_step);
    const float activation_floor = std::max(1e-6f, config.dbn_activation_floor);
    const std::size_t beats_per_bar = std::max<std::size_t>(1, config.dbn_beats_per_bar);
    const float downbeat_weight = std::max(0.0f, config.dbn_downbeat_weight);
    const float tempo_change_penalty = std::max(0.0f, config.dbn_tempo_change_penalty);
    const float tempo_prior_weight = std::max(0.0f, config.dbn_tempo_prior_weight);
    const float transition_reward = config.dbn_transition_reward;
    const std::size_t max_candidates = std::max<std::size_t>(4, config.dbn_max_candidates);
    const double floor_value = std::max(1e-6f, activation_floor);
    const double tolerance = std::max(0.0, static_cast<double>(config.dbn_interval_tolerance));

    const bool use_all_candidates = config.dbn_use_all_candidates;

    std::vector<std::size_t> candidates;
    std::size_t candidate_count = beat_activation.size();
    if (!use_all_candidates) {
        candidates.reserve(beat_activation.size());
        for (std::size_t i = 0; i < beat_activation.size(); ++i) {
            if (beat_activation[i] >= activation_floor) {
                candidates.push_back(i);
            }
        }
        candidate_count = candidates.size();
    }

    if (!use_all_candidates && candidates.size() > max_candidates) {
        std::vector<std::size_t> sorted = candidates;
        std::nth_element(sorted.begin(),
                         sorted.begin() + static_cast<std::ptrdiff_t>(max_candidates),
                         sorted.end(),
                         [&](std::size_t a, std::size_t b) {
                             return beat_activation[a] > beat_activation[b];
                         });
        sorted.resize(max_candidates);
        std::sort(sorted.begin(), sorted.end());
        candidates.swap(sorted);
    }

    if (config.dbn_trace && beatit_should_log("debug")) {
        const std::size_t first_window =
            std::min<std::size_t>(beat_activation.size(),
                                  static_cast<std::size_t>(std::llround(2.0 * fps)));
        auto summarize_activation = [&](const std::vector<float>& activation,
                                        const char* label) {
            if (activation.empty() || first_window == 0) {
                BEATIT_LOG_DEBUG("DBN: " << label << " first2s: empty");
                return;
            }
            float min_val = activation[0];
            float max_val = activation[0];
            double sum = 0.0;
            std::size_t above = 0;
            for (std::size_t i = 0; i < first_window; ++i) {
                const float v = activation[i];
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
                sum += v;
                if (v >= activation_floor) {
                    ++above;
                }
            }
            const double mean = sum / static_cast<double>(first_window);
            BEATIT_LOG_DEBUG("DBN: " << label << " first2s"
                             << " frames=" << first_window
                             << " min=" << min_val
                             << " max=" << max_val
                             << " mean=" << mean
                             << " above_floor=" << above
                             << " floor=" << activation_floor);
        };
        summarize_activation(beat_activation, "beat");
        summarize_activation(downbeat_activation, "downbeat");

    }

    if (candidate_count < 2) {
        result.beat_frames = CalmdadDecoder::viterbi_beats(beat_activation,
                                                           fps,
                                                           std::max(1.0f, min_bpm),
                                                           config.dbn_interval_tolerance,
                                                           activation_floor);
        if (!result.beat_frames.empty()) {
            result.downbeat_frames.push_back(result.beat_frames.front());
        }
        return result;
    }

    const std::size_t tempo_count =
        static_cast<std::size_t>(std::floor((max_bpm - min_bpm) / bpm_step)) + 1;
    const std::size_t phase_count = beats_per_bar;

    BEATIT_LOG_DEBUG("DBN config: all_candidates="
                     << (use_all_candidates ? "true" : "false")
                     << " used_candidates=" << candidate_count
                     << " floor=" << activation_floor
                     << " tol=" << tolerance
                     << " bpm=[" << min_bpm << "," << max_bpm << "]"
                     << " step=" << bpm_step
                     << " tempos=" << tempo_count
                     << " bpb=" << beats_per_bar
                     << " reference_bpm=" << reference_bpm);

    std::vector<double> beat_log(beat_activation.size(), 0.0);
    for (std::size_t i = 0; i < beat_activation.size(); ++i) {
        beat_log[i] = std::log(std::max<double>(beat_activation[i], floor_value));
    }

    std::vector<double> downbeat_log(downbeat_activation.size(), 0.0);
    for (std::size_t i = 0; i < downbeat_activation.size(); ++i) {
        downbeat_log[i] = std::log(std::max<double>(downbeat_activation[i], floor_value));
    }

    struct TempoParams {
        float bpm = 0.0f;
        std::size_t min_interval = 0;
        std::size_t max_interval = 0;
        double prior_penalty = 0.0;
    };

    std::vector<TempoParams> tempos;
    tempos.reserve(tempo_count);
    for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
        const float bpm = min_bpm + static_cast<float>(tempo_idx) * bpm_step;
        const double interval = (60.0 * fps) / bpm;
        const double min_interval = interval * (1.0 - tolerance);
        const double max_interval = interval * (1.0 + tolerance);
        double prior_penalty = 0.0;
        if (reference_bpm > 0.0f && tempo_prior_weight > 0.0f) {
            prior_penalty = tempo_prior_weight * std::abs(bpm - reference_bpm);
        }
        tempos.push_back({
            bpm,
            static_cast<std::size_t>(std::max(1.0, std::floor(min_interval))),
            static_cast<std::size_t>(std::max(1.0, std::ceil(max_interval))),
            prior_penalty
        });
    }

    struct Backref {
        int prev_idx = -1;
        int prev_tempo = -1;
    };

    const std::size_t state_count = tempo_count * phase_count;
    const std::size_t total_states = candidate_count * state_count;
    std::vector<double> scores(total_states, std::numeric_limits<double>::lowest());
    std::vector<Backref> backrefs(total_states);

    auto state_index = [&](std::size_t cand_idx, std::size_t tempo_idx, std::size_t phase_idx) {
        return (cand_idx * tempo_count + tempo_idx) * phase_count + phase_idx;
    };

    for (std::size_t ci = 0; ci < candidate_count; ++ci) {
        const std::size_t frame = use_all_candidates ? ci : candidates[ci];
        const double beat_obs = beat_log[frame];
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            const auto& tempo = tempos[tempo_idx];
            const std::size_t min_prev_frame =
                (frame > tempo.max_interval) ? frame - tempo.max_interval : 0;
            const std::size_t max_prev_frame =
                (frame > tempo.min_interval) ? frame - tempo.min_interval : 0;
            std::size_t start_idx = 0;
            std::size_t end_idx = 0;
            if (use_all_candidates) {
                start_idx = min_prev_frame;
                end_idx = std::min(max_prev_frame + 1, ci);
            } else {
                const auto start_it =
                    std::lower_bound(candidates.begin(),
                                     candidates.begin() + static_cast<std::ptrdiff_t>(ci),
                                     min_prev_frame);
                const auto end_it =
                    std::upper_bound(candidates.begin(),
                                     candidates.begin() + static_cast<std::ptrdiff_t>(ci),
                                     max_prev_frame);
                start_idx = static_cast<std::size_t>(std::distance(candidates.begin(), start_it));
                end_idx = static_cast<std::size_t>(std::distance(candidates.begin(), end_it));
            }

            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
                const bool is_downbeat = (phase_idx == 0);
                double obs = beat_obs - tempo.prior_penalty;
                if (config.dbn_use_downbeat && is_downbeat && frame < downbeat_log.size()) {
                    obs += downbeat_weight * downbeat_log[frame];
                }

                double best_score = obs;
                Backref best_backref;

                const std::size_t prev_phase =
                    (phase_idx + phase_count - 1) % phase_count;
                for (std::size_t cj = start_idx; cj < end_idx; ++cj) {
                    for (int tempo_delta = -1; tempo_delta <= 1; ++tempo_delta) {
                        const int prev_tempo = static_cast<int>(tempo_idx) + tempo_delta;
                        if (prev_tempo < 0 || prev_tempo >= static_cast<int>(tempo_count)) {
                            continue;
                        }
                        const std::size_t idx =
                            state_index(cj, static_cast<std::size_t>(prev_tempo), prev_phase);
                        const double prev_score = scores[idx];
                        if (prev_score == std::numeric_limits<double>::lowest()) {
                            continue;
                        }
                        const double tempo_penalty =
                            tempo_change_penalty * std::abs(tempo.bpm - tempos[prev_tempo].bpm);
                        const double candidate = prev_score + obs - tempo_penalty;
                        const double rewarded = candidate + transition_reward;
                        if (rewarded > best_score) {
                            best_score = rewarded;
                            best_backref.prev_idx = static_cast<int>(cj);
                            best_backref.prev_tempo = prev_tempo;
                        }
                    }
                }

                const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
                scores[idx] = best_score;
                backrefs[idx] = best_backref;
            }
        }
    }

    double best_score = std::numeric_limits<double>::lowest();
    std::size_t best_ci = 0;
    std::size_t best_tempo = 0;
    std::size_t best_phase = 0;
    for (std::size_t ci = 0; ci < candidate_count; ++ci) {
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
                const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
                const double score = scores[idx];
                if (score > best_score) {
                    best_score = score;
                    best_ci = ci;
                    best_tempo = tempo_idx;
                    best_phase = phase_idx;
                }
            }
        }
    }

    std::vector<std::size_t> beat_frames;
    std::vector<std::size_t> downbeat_frames;
    std::size_t ci = best_ci;
    std::size_t tempo_idx = best_tempo;
    std::size_t phase_idx = best_phase;

    while (true) {
        beat_frames.push_back(use_all_candidates ? ci : candidates[ci]);
        if (phase_idx == 0) {
            downbeat_frames.push_back(use_all_candidates ? ci : candidates[ci]);
        }
        const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
        const Backref ref = backrefs[idx];
        if (ref.prev_idx < 0) {
            break;
        }
        ci = static_cast<std::size_t>(ref.prev_idx);
        tempo_idx = static_cast<std::size_t>(ref.prev_tempo);
        phase_idx = (phase_idx + phase_count - 1) % phase_count;
    }

    std::reverse(beat_frames.begin(), beat_frames.end());
    std::reverse(downbeat_frames.begin(), downbeat_frames.end());

    result.beat_frames = std::move(beat_frames);
    result.downbeat_frames = std::move(downbeat_frames);

    if (result.downbeat_frames.empty() && !result.beat_frames.empty()) {
        result.downbeat_frames.push_back(result.beat_frames.front());
    }
    if (best_tempo < tempos.size()) {
        result.bpm = tempos[best_tempo].bpm;
    }

    return result;
}

}  // namespace beatit
