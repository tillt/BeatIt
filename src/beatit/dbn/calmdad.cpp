//
//  calmdad.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//
// CALMDAD Analyzes Latent Meter Dynamics And Downbeats.
// A DBN-based beat and downbeat decoder.
//
#include "beatit/dbn/calmdad.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace beatit {

CalmdadDecoder::CalmdadDecoder(const BeatitConfig& config)
    : config_(config) {}

std::vector<std::size_t> CalmdadDecoder::viterbi_beats(const std::vector<float>& activation,
                                                       double fps,
                                                       double bpm,
                                                       double interval_tolerance,
                                                       float activation_floor) {
    std::vector<std::size_t> beats;
    if (activation.empty() || fps <= 0.0 || bpm <= 0.0) {
        return beats;
    }

    const double interval = (60.0 * fps) / bpm;
    const double tolerance = std::max(0.0, interval_tolerance);
    const std::size_t min_interval =
        static_cast<std::size_t>(std::max(1.0, std::floor(interval * (1.0 - tolerance))));
    const std::size_t max_interval =
        static_cast<std::size_t>(std::max<double>(min_interval,
                                                  std::ceil(interval * (1.0 + tolerance))));

    const double floor_value = std::max(1e-6f, activation_floor);
    const std::size_t frames = activation.size();

    std::vector<double> score(frames, std::numeric_limits<double>::lowest());
    std::vector<int> prev(frames, -1);

    for (std::size_t i = 0; i < frames; ++i) {
        const double obs = std::log(std::max<double>(activation[i], floor_value));
        score[i] = obs;
        prev[i] = -1;

        const std::size_t start = (i > max_interval) ? i - max_interval : 0;
        const std::size_t end = (i > min_interval) ? i - min_interval : 0;
        for (std::size_t j = start; j <= end; ++j) {
            if (score[j] == std::numeric_limits<double>::lowest()) {
                continue;
            }
            const double candidate = score[j] + obs;
            if (candidate > score[i]) {
                score[i] = candidate;
                prev[i] = static_cast<int>(j);
            }
        }
    }

    std::size_t best_idx = 0;
    double best_score = std::numeric_limits<double>::lowest();
    for (std::size_t i = 0; i < frames; ++i) {
        if (score[i] > best_score) {
            best_score = score[i];
            best_idx = i;
        }
    }

    int cursor = static_cast<int>(best_idx);
    while (cursor >= 0) {
        beats.push_back(static_cast<std::size_t>(cursor));
        cursor = prev[static_cast<std::size_t>(cursor)];
    }

    std::reverse(beats.begin(), beats.end());
    return beats;
}

namespace {

void log_frame_preview(const char* label,
                       const std::vector<std::size_t>& frames,
                       double fps,
                       std::size_t max_count) {
    auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
    debug_stream << label;
    const std::size_t count = std::min(max_count, frames.size());
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t frame = frames[i];
        debug_stream << " " << frame << "->" << (static_cast<double>(frame) / fps);
    }
    if (frames.empty()) {
        debug_stream << " none";
    }
}

struct ActivationLogData {
    std::vector<double> beat_log;
    std::vector<double> downbeat_log;
    double beat_min = std::numeric_limits<double>::infinity();
    double beat_max = 0.0;
    double downbeat_min = std::numeric_limits<double>::infinity();
    double downbeat_max = 0.0;
    double combined_max = 0.0;
    std::size_t beat_above_floor = 0;
    std::size_t downbeat_above_floor = 0;
};

ActivationLogData build_activation_logs(const std::vector<float>& beat_activation,
                                        const std::vector<float>& downbeat_activation,
                                        double epsilon,
                                        double floor_value) {
    ActivationLogData logs;
    logs.beat_log.assign(beat_activation.size(), 0.0);
    logs.downbeat_log.assign(downbeat_activation.size(), 0.0);
    for (std::size_t i = 0; i < beat_activation.size(); ++i) {
        const float raw_downbeat =
            (i < downbeat_activation.size()) ? downbeat_activation[i] : 0.0f;
        const double beat_value =
            static_cast<double>(beat_activation[i]) * (1.0 - epsilon) + floor_value;
        const double downbeat_value =
            static_cast<double>(raw_downbeat) * (1.0 - epsilon) + floor_value;
        const double combined_beat = std::max(floor_value, beat_value - downbeat_value);
        logs.beat_min = std::min<double>(logs.beat_min, beat_value);
        logs.beat_max = std::max<double>(logs.beat_max, beat_value);
        logs.downbeat_min = std::min<double>(logs.downbeat_min, downbeat_value);
        logs.downbeat_max = std::max<double>(logs.downbeat_max, downbeat_value);
        logs.combined_max = std::max<double>(logs.combined_max, combined_beat);
        if (combined_beat > floor_value) {
            ++logs.beat_above_floor;
        }
        logs.beat_log[i] = std::log(combined_beat);
        if (i < downbeat_activation.size()) {
            logs.downbeat_log[i] = std::log(downbeat_value);
            if (downbeat_value > floor_value) {
                ++logs.downbeat_above_floor;
            }
        }
    }
    return logs;
}

void log_activation_candidates(const ActivationLogData& logs, double floor_value, double fps) {
    constexpr std::size_t kDumpCount = 10;
    std::vector<std::size_t> beat_frames;
    beat_frames.reserve(kDumpCount);
    for (std::size_t i = 0; i < logs.beat_log.size() && beat_frames.size() < kDumpCount; ++i) {
        if (std::exp(logs.beat_log[i]) > floor_value) {
            beat_frames.push_back(i);
        }
    }
    log_frame_preview("DBN calmdad: first beat candidates (frame->s):",
                      beat_frames,
                      fps,
                      kDumpCount);

    std::vector<std::size_t> downbeat_frames;
    downbeat_frames.reserve(kDumpCount);
    for (std::size_t i = 0; i < logs.downbeat_log.size() && downbeat_frames.size() < kDumpCount; ++i) {
        if (std::exp(logs.downbeat_log[i]) > floor_value) {
            downbeat_frames.push_back(i);
        }
    }
    log_frame_preview("DBN calmdad: first downbeat candidates (frame->s):",
                      downbeat_frames,
                      fps,
                      kDumpCount);
}

void log_decoded_candidates(const DBNDecodeResult& decoded, double fps) {
    constexpr std::size_t kDumpCount = 10;
    log_frame_preview("DBN calmdad: first beats (frame->s):",
                      decoded.beat_frames,
                      fps,
                      kDumpCount);
    log_frame_preview("DBN calmdad: first downbeats (frame->s):",
                      decoded.downbeat_frames,
                      fps,
                      kDumpCount);
}

DBNPathResult decode_dbn_beats_candidate(const std::vector<double>& beat_log,
                                         const std::vector<double>& downbeat_log,
                                         double fps,
                                         float min_bpm,
                                         float max_bpm,
                                         float bpm_step,
                                         std::size_t beats_per_bar,
                                         double tolerance,
                                         bool use_downbeat,
                                         double transition_reward,
                                         double tempo_change_penalty,
                                         const BeatitConfig& config) {
    DBNPathResult result;
    if (beat_log.empty() || fps <= 0.0) {
        return result;
    }

    const std::size_t candidate_count = beat_log.size();
    if (candidate_count < 2) {
        result.decoded.beat_frames = CalmdadDecoder::viterbi_beats(
            config.dbn_use_all_candidates
                ? std::vector<float>(beat_log.begin(), beat_log.end())
                : std::vector<float>(),
            fps,
            std::max(1.0f, min_bpm),
            static_cast<float>(tolerance),
            0.01f);
        if (!result.decoded.beat_frames.empty()) {
            result.decoded.downbeat_frames.push_back(result.decoded.beat_frames.front());
            result.best_score = 0.0;
        }
        return result;
    }

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
        const std::size_t frame = ci;
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            const auto& tempo = tempos[tempo_idx];
            const std::size_t min_prev_frame =
                (frame > tempo.max_interval) ? frame - tempo.max_interval : 0;
            const std::size_t max_prev_frame =
                (frame > tempo.min_interval) ? frame - tempo.min_interval : 0;
            const std::size_t start_idx = min_prev_frame;
            const std::size_t end_idx = std::min(max_prev_frame + 1, ci);

            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
                const bool is_downbeat = (phase_idx == 0);
                double obs = beat_log[frame];
                if (use_downbeat && is_downbeat && frame < downbeat_log.size()) {
                    obs = downbeat_log[frame];
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
        beat_frames.push_back(ci);
        if (phase_idx == 0) {
            downbeat_frames.push_back(ci);
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
    if (result.decoded.downbeat_frames.empty() && !result.decoded.beat_frames.empty()) {
        result.decoded.downbeat_frames.push_back(result.decoded.beat_frames.front());
    }
    if (best_tempo < tempos.size()) {
        result.decoded.bpm = tempos[best_tempo].bpm;
    }
    result.best_score = best_score;
    return result;
}

}  // namespace

DBNDecodeResult CalmdadDecoder::decode(const CalmdadDecodeRequest& request) const {
    const auto& beat_activation = request.beat_activation;
    const auto& downbeat_activation = request.downbeat_activation;
    const double fps = request.fps;
    const float min_bpm = request.min_bpm;
    const float max_bpm = request.max_bpm;
    const float bpm_step = request.bpm_step;

    DBNDecodeResult best;
    if (beat_activation.empty() || fps <= 0.0) {
        return best;
    }

    const float local_step = std::max(0.1f, bpm_step);
    const float local_min = std::max(1.0f, min_bpm);
    const float local_max = std::max(local_min + local_step, max_bpm);
    const double tolerance = std::max(0.0, static_cast<double>(config_.dbn_interval_tolerance));
    const double epsilon = 1e-5;
    const double floor_value = epsilon / 2.0;
    const double transition_lambda = std::max(1e-6, static_cast<double>(config_.dbn_transition_lambda));
    const double transition_reward = std::log(transition_lambda);
    const double tempo_change_penalty = transition_reward;
    const bool use_downbeat = config_.dbn_use_downbeat;
    const ActivationLogData logs =
        build_activation_logs(beat_activation, downbeat_activation, epsilon, floor_value);

    std::vector<std::size_t> bpb_options = {3, 4};
    DBNPathResult best_path;
    std::size_t best_bpb = 0;
    BEATIT_LOG_DEBUG("DBN calmdad: frames=" << beat_activation.size()
                     << " floor=" << floor_value
                     << " epsilon=" << epsilon
                     << " tol=" << tolerance
                     << " bpm=[" << local_min << "," << local_max << "]"
                     << " step=" << local_step
                     << " lambda=" << transition_lambda
                     << " use_downbeat=" << (use_downbeat ? "true" : "false")
                     << " beat[min,max]=[" << logs.beat_min << "," << logs.beat_max << "]"
                     << " downbeat[min,max]=[" << logs.downbeat_min << "," << logs.downbeat_max
                     << "]"
                     << " combined_max=" << logs.combined_max
                     << " beat>floor=" << logs.beat_above_floor
                     << " downbeat>floor=" << logs.downbeat_above_floor);
    log_activation_candidates(logs, floor_value, fps);
    for (std::size_t bpb : bpb_options) {
        DBNPathResult path = decode_dbn_beats_candidate(
            logs.beat_log,
            logs.downbeat_log,
            fps,
            local_min,
            local_max,
            local_step,
            bpb,
            tolerance,
            use_downbeat,
            transition_reward,
            tempo_change_penalty,
            config_);
        if (path.best_score > best_path.best_score) {
            best_path = std::move(path);
            best_bpb = bpb;
        }
    }

    best = std::move(best_path.decoded);
    BEATIT_LOG_DEBUG("DBN calmdad: best_bpb=" << best_bpb
                     << " beats=" << best.beat_frames.size()
                     << " downbeats=" << best.downbeat_frames.size()
                     << " best_score=" << best_path.best_score);
    log_decoded_candidates(best, fps);
    return best;
}

}  // namespace beatit
