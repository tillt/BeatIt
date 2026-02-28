//
//  grid_phase_select.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "grid_projection.h"

#include "beatit/post/helpers.h"
#include "beatit/post/result_ops.h"
#include "beatit/post/window.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace beatit::detail {

namespace {

struct PhaseScore {
    double score = -std::numeric_limits<double>::infinity();
    std::size_t hits = 0;
    const char* source = "none";
};

std::size_t onset_from_peak(const std::vector<float>& activation,
                            std::size_t peak_frame,
                            float onset_ratio,
                            std::size_t onset_max_back) {
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
}

std::vector<std::size_t> build_onset_frames(const std::vector<std::size_t>& frames,
                                            const std::vector<float>& activation,
                                            float onset_ratio,
                                            std::size_t onset_max_back) {
    std::vector<std::size_t> out;
    out.reserve(frames.size());
    for (std::size_t frame : frames) {
        out.push_back(onset_from_peak(activation, frame, onset_ratio, onset_max_back));
    }
    detail::dedupe_frames(out);
    return out;
}

PhaseScore fallback_phase_score(const std::vector<std::size_t>& phase_frames,
                                const CoreMLResult& result,
                                bool allow_downbeat_phase,
                                float downbeat_peak_ratio,
                                float max_downbeat) {
    PhaseScore out;
    const std::size_t max_checks = std::min<std::size_t>(phase_frames.size(), 3);
    double sum = 0.0;

    if (allow_downbeat_phase) {
        const float threshold =
            max_downbeat > 0.0f ? static_cast<float>(max_downbeat * downbeat_peak_ratio) : 0.0f;
        for (std::size_t i = 0; i < max_checks; ++i) {
            const std::size_t frame = phase_frames[i];
            if (frame < result.downbeat_activation.size()) {
                const float value = result.downbeat_activation[frame];
                if (value >= threshold) {
                    sum += value;
                    out.hits += 1;
                }
            }
        }
        if (out.hits == 0) {
            for (std::size_t i = 0; i < max_checks; ++i) {
                const std::size_t frame = phase_frames[i];
                if (frame < result.downbeat_activation.size()) {
                    sum += result.downbeat_activation[frame];
                }
            }
            out.hits = max_checks;
        }
        out.source = "fallback_downbeat";
    } else {
        for (std::size_t i = 0; i < max_checks; ++i) {
            const std::size_t frame = phase_frames[i];
            if (frame < result.beat_activation.size()) {
                sum += result.beat_activation[frame];
            }
        }
        out.hits = max_checks;
        out.source = "fallback_beat";
    }

    if (out.hits > 0) {
        const double penalty = 0.01 * static_cast<double>(phase_frames.front());
        out.score = sum - penalty;
    }
    return out;
}

PhaseScore score_beat_phase_window(const std::vector<std::size_t>& phase_frames,
                                   const CoreMLResult& result,
                                   const std::vector<uint8_t>& phase_peak_mask,
                                   bool has_phase_peaks,
                                   std::size_t phase_window_frames,
                                   float max_beat,
                                   float peak_ratio) {
    PhaseScore out;
    double sum = 0.0;
    double weight = 0.0;

    if (has_phase_peaks) {
        for (std::size_t frame : phase_frames) {
            if (frame >= phase_window_frames) {
                break;
            }
            if (frame < phase_peak_mask.size() && phase_peak_mask[frame]) {
                const double value = (frame < result.beat_activation.size())
                    ? static_cast<double>(result.beat_activation[frame])
                    : 0.0;
                sum += value;
                weight += 1.0;
            }
        }
        if (weight > 0.0) {
            out.score = sum / weight;
            out.hits = static_cast<std::size_t>(weight);
            out.source = "beat_peak_mask";
        }
        return out;
    }

    const float threshold =
        max_beat > 0.0f ? static_cast<float>(max_beat * peak_ratio) : 0.0f;
    for (std::size_t frame : phase_frames) {
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
        out.score = sum / weight;
        out.hits = static_cast<std::size_t>(weight);
        out.source = "beat_threshold";
    }
    return out;
}

PhaseScore score_downbeat_phase_window(const std::vector<std::size_t>& phase_frames,
                                       const CoreMLResult& result,
                                       std::size_t phase_window_frames,
                                       std::size_t phase_window_end,
                                       float max_downbeat,
                                       float peak_ratio) {
    PhaseScore out;
    double sum = 0.0;
    double weight = 0.0;
    const float threshold =
        max_downbeat > 0.0f ? static_cast<float>(max_downbeat * peak_ratio) : 0.0f;
    const std::size_t end = std::min(phase_window_end, result.downbeat_activation.size());
    const double decay = std::max(1.0, phase_window_frames * 0.2);

    for (std::size_t frame : phase_frames) {
        if (frame >= end) {
            break;
        }
        const float value = result.downbeat_activation[frame];
        if (value >= threshold) {
            const double current_weight = std::exp(-static_cast<double>(frame) / decay);
            sum += static_cast<double>(value) * current_weight;
            weight += current_weight;
        }
    }
    if (weight > 0.0) {
        out.score = sum / weight;
        out.hits = static_cast<std::size_t>(weight + 0.5);
        out.source = "downbeat_window_decay";
    }
    return out;
}

} // namespace

void select_downbeat_phase(GridProjectionState& state,
                           DBNDecodeResult& decoded,
                           const CoreMLResult& result,
                           const BeatitConfig& config,
                           bool quality_low,
                           bool downbeat_override_ok,
                           bool use_window,
                           std::size_t window_start,
                           std::size_t used_frames,
                           double fps) {
    const double local_frame_rate =
        config.hop_size > 0
            ? static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size)
            : 0.0;
    const std::size_t phase_window_frames =
        (config.dbn_downbeat_phase_window_seconds > 0.0 && local_frame_rate > 0.0)
            ? static_cast<std::size_t>(
                  std::llround(config.dbn_downbeat_phase_window_seconds * local_frame_rate))
            : 0;
    const std::size_t phase_window_start = use_window ? window_start : 0;
    const std::size_t phase_window_end =
        (phase_window_frames > 0) ? std::min(used_frames, phase_window_start + phase_window_frames)
                                  : phase_window_start;
    const bool allow_downbeat_phase =
        !result.downbeat_activation.empty() && !quality_low && downbeat_override_ok;
    const std::size_t max_delay_frames =
        (config.dbn_downbeat_phase_max_delay_seconds > 0.0 && local_frame_rate > 0.0)
            ? static_cast<std::size_t>(
                  std::llround(config.dbn_downbeat_phase_max_delay_seconds * local_frame_rate))
            : 0;
    const float onset_ratio = 0.35f;
    const std::size_t onset_max_back = max_delay_frames > 0 ? max_delay_frames : 8;

    float max_beat = 0.0f;
    std::vector<uint8_t> phase_peak_mask;
    bool has_phase_peaks = false;
    if (phase_window_frames > 0) {
        for (std::size_t i = phase_window_start; i < phase_window_end; ++i) {
            if (allow_downbeat_phase) {
                state.max_downbeat = std::max(state.max_downbeat, result.downbeat_activation[i]);
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
                    if (curr >= beat_threshold && curr >= prev + peak_eps && curr >= next + peak_eps) {
                        const std::size_t onset_frame =
                            onset_from_peak(result.beat_activation, i, onset_ratio, onset_max_back);
                        if (onset_frame < phase_peak_mask.size()) {
                            phase_peak_mask[onset_frame] = 1;
                        }
                    has_phase_peaks = true;
                }
            }
        }
        if (config.dbn_trace && allow_downbeat_phase) {
            auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
            debug_stream << "DBN: phase selection window=["
                         << phase_window_start << "," << phase_window_end << ")"
                         << " downbeat_max=" << state.max_downbeat
                         << " beat_max=" << max_beat
                         << " activation_floor=" << state.activation_floor
                         << " has_phase_peaks=" << (has_phase_peaks ? 1 : 0)
                         << "\n";
        }
    }
    const float downbeat_threshold =
        state.max_downbeat > 0.0f
            ? static_cast<float>(state.max_downbeat * config.dbn_downbeat_phase_peak_ratio)
            : 0.0f;
    for (std::size_t candidate_phase = 0; candidate_phase < state.bpb; ++candidate_phase) {
        const auto projected =
            project_downbeats_from_beats(decoded.beat_frames, state.bpb, candidate_phase);
        if (projected.empty()) {
            continue;
        }
        const std::vector<float>& onset_activation =
            allow_downbeat_phase ? result.downbeat_activation : result.beat_activation;
        const auto projected_onsets =
            build_onset_frames(projected, onset_activation, onset_ratio, onset_max_back);
        const auto& phase_frames = projected_onsets.empty() ? projected : projected_onsets;
        PhaseScore candidate;

        if (phase_window_frames > 0 && !allow_downbeat_phase) {
            candidate = score_beat_phase_window(phase_frames,
                                                result,
                                                phase_peak_mask,
                                                has_phase_peaks,
                                                phase_window_frames,
                                                max_beat,
                                                config.dbn_downbeat_phase_peak_ratio);
        } else if (phase_window_frames > 0 && allow_downbeat_phase) {
            candidate = score_downbeat_phase_window(phase_frames,
                                                    result,
                                                    phase_window_frames,
                                                    phase_window_end,
                                                    state.max_downbeat,
                                                    config.dbn_downbeat_phase_peak_ratio);
        }
        if (!std::isfinite(candidate.score)) {
            candidate = fallback_phase_score(phase_frames,
                                             result,
                                             allow_downbeat_phase,
                                             config.dbn_downbeat_phase_peak_ratio,
                                             state.max_downbeat);
        }
        double delay_penalty = 0.0;
        if (max_delay_frames > 0 && phase_frames.front() > max_delay_frames &&
            std::strncmp(candidate.source, "fallback", 8) != 0 &&
            !(quality_low &&
              (std::strncmp(candidate.source, "beat_peak_mask", 14) == 0 ||
               std::strncmp(candidate.source, "beat_threshold", 14) == 0))) {
            const double delay = static_cast<double>(phase_frames.front() - max_delay_frames);
            delay_penalty = delay * 1000.0;
            candidate.score -= delay_penalty;
        }
        if (candidate.score > state.best_score) {
            state.best_score = candidate.score;
            state.best_phase = candidate_phase;
        }
    }
    BEATIT_LOG_DEBUG("DBN: phase_window_frames=" << phase_window_frames
                     << " max_downbeat=" << state.max_downbeat
                     << " threshold=" << downbeat_threshold
                     << " best_phase=" << state.best_phase
                     << " best_score=" << state.best_score);
    decoded.downbeat_frames =
        project_downbeats_from_beats(decoded.beat_frames, state.bpb, state.best_phase);
    if (config.dbn_trace) {
        const std::size_t preview = std::min<std::size_t>(6, decoded.downbeat_frames.size());
        auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
        debug_stream << "DBN: downbeat frames head:";
        for (std::size_t i = 0; i < preview; ++i) {
            const std::size_t frame = decoded.downbeat_frames[i];
            const double time_s = fps > 0.0 ? static_cast<double>(frame) / fps : 0.0;
            debug_stream << " " << frame << "(" << time_s << "s)";
        }
        debug_stream << "\n";
        if (!decoded.downbeat_frames.empty()) {
            const std::size_t first_frame = decoded.downbeat_frames.front();
            const double first_time = fps > 0.0 ? static_cast<double>(first_frame) / fps : 0.0;
            debug_stream << "DBN: downbeat selection start="
                         << first_frame << " (" << first_time << "s)"
                         << " bpb=" << state.bpb
                         << " phase=" << state.best_phase
                         << " score=" << state.best_score
                         << "\n";
        }
    }
}

} // namespace beatit::detail
