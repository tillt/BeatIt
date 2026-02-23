//
//  calmdad_sparse.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/dbn/calmdad.h"

#include <algorithm>
#include <cmath>

namespace beatit {

DBNPathResult CalmdadDecoder::decode_sparse(const CalmdadSparseDecodeRequest& request) const {
    const auto& candidate_frames = request.candidate_frames;
    const auto& beat_log = request.beat_log;
    const auto& downbeat_log = request.downbeat_log;
    const double fps = request.fps;
    const float min_bpm = request.min_bpm;
    const float max_bpm = request.max_bpm;
    const float bpm_step = request.bpm_step;
    const std::size_t beats_per_bar = request.beats_per_bar;
    const double tolerance = request.tolerance;
    const bool use_downbeat = request.use_downbeat;
    const double transition_reward = request.transition_reward;
    const double tempo_change_penalty = request.tempo_change_penalty;

    DBNPathResult result;
    if (candidate_frames.size() < 2 || beat_log.size() != candidate_frames.size() || fps <= 0.0) {
        return result;
    }

    const std::size_t candidate_count = candidate_frames.size();
    const std::size_t tempo_count =
        static_cast<std::size_t>(std::floor((max_bpm - min_bpm) / bpm_step)) + 1;
    const std::size_t phase_count = beats_per_bar;

    struct TempoParams {
        float bpm = 0.0f;
        std::size_t min_interval = 0;
        std::size_t max_interval = 0;
    };

    std::vector<TempoParams> tempos;
    tempos.reserve(tempo_count);
    for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
        const float bpm = min_bpm + static_cast<float>(tempo_idx) * bpm_step;
        const double interval = (60.0 * fps) / bpm;
        const double min_interval = interval * (1.0 - tolerance);
        const double max_interval = interval * (1.0 + tolerance);
        tempos.push_back({
            bpm,
            static_cast<std::size_t>(std::max(1.0, std::floor(min_interval))),
            static_cast<std::size_t>(std::max(1.0, std::ceil(max_interval))),
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
        const std::size_t frame = candidate_frames[ci];
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            const auto& tempo = tempos[tempo_idx];
            const std::size_t min_prev_frame =
                (frame > tempo.max_interval) ? frame - tempo.max_interval : 0;
            const std::size_t max_prev_frame =
                (frame > tempo.min_interval) ? frame - tempo.min_interval : 0;

            const auto start_it = std::lower_bound(candidate_frames.begin(),
                                                   candidate_frames.end(),
                                                   min_prev_frame);
            const auto end_it = std::upper_bound(candidate_frames.begin(),
                                                 candidate_frames.end(),
                                                 max_prev_frame);
            const std::size_t start_idx =
                static_cast<std::size_t>(std::distance(candidate_frames.begin(), start_it));
            const std::size_t end_idx =
                static_cast<std::size_t>(std::distance(candidate_frames.begin(), end_it));

            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
                const bool is_downbeat = (phase_idx == 0);
                double obs = beat_log[ci];
                if (use_downbeat && is_downbeat && ci < downbeat_log.size()) {
                    obs = downbeat_log[ci];
                }

                double best_score = obs;
                Backref best_backref;

                const std::size_t prev_phase =
                    (phase_idx + phase_count - 1) % phase_count;
                for (std::size_t cj = start_idx; cj < end_idx; ++cj) {
                    if (cj >= ci) {
                        break;
                    }
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
                        const double candidate = prev_score + obs - tempo_penalty + transition_reward;
                        if (candidate > best_score) {
                            best_score = candidate;
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
        beat_frames.push_back(candidate_frames[ci]);
        if (phase_idx == 0) {
            downbeat_frames.push_back(candidate_frames[ci]);
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

    result.decoded.beat_frames = std::move(beat_frames);
    result.decoded.downbeat_frames = std::move(downbeat_frames);
    if (best_tempo < tempos.size()) {
        result.decoded.bpm = tempos[best_tempo].bpm;
    }
    result.best_score = best_score;
    return result;
}

}  // namespace beatit
