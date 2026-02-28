//
//  window.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/window.h"

#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace beatit::detail {

namespace {

std::size_t nearest_beat_frame(const std::vector<std::size_t>& beats, std::size_t frame) {
    if (beats.empty()) {
        return frame;
    }
    const auto it = std::lower_bound(beats.begin(), beats.end(), frame);
    if (it == beats.begin()) {
        return *it;
    }
    if (it == beats.end()) {
        return beats.back();
    }

    const std::size_t hi = *it;
    const std::size_t lo = *(it - 1);
    const std::size_t d_hi = hi - frame;
    const std::size_t d_lo = frame - lo;
    return (d_lo <= d_hi) ? lo : hi;
}

} // namespace

std::vector<std::size_t> deduplicate_peaks(const std::vector<std::size_t>& peaks, std::size_t width) {
    std::vector<std::size_t> result;
    if (peaks.empty()) {
        return result;
    }
    double p = static_cast<double>(peaks.front());
    std::size_t count = 1;
    for (std::size_t i = 1; i < peaks.size(); ++i) {
        const double next = static_cast<double>(peaks[i]);
        if (next - p <= static_cast<double>(width)) {
            ++count;
            p += (next - p) / static_cast<double>(count);
        } else {
            result.push_back(static_cast<std::size_t>(std::llround(p)));
            p = next;
            count = 1;
        }
    }
    result.push_back(static_cast<std::size_t>(std::llround(p)));
    return result;
}

std::vector<std::size_t> compute_minimal_peaks(const std::vector<float>& activation) {
    constexpr std::size_t window = 7;
    constexpr std::size_t half = window / 2;
    constexpr float threshold = 0.5f;
    std::vector<std::size_t> peaks;
    if (activation.empty()) {
        return peaks;
    }
    peaks.reserve(activation.size() / 10);
    for (std::size_t i = 0; i < activation.size(); ++i) {
        const float value = activation[i];
        if (value <= threshold) {
            continue;
        }
        const std::size_t start = (i > half) ? i - half : 0;
        const std::size_t end = std::min(activation.size() - 1, i + half);
        float local_max = value;
        for (std::size_t j = start; j <= end; ++j) {
            local_max = std::max(local_max, activation[j]);
        }
        if (value >= local_max) {
            peaks.push_back(i);
        }
    }
    return deduplicate_peaks(peaks, 1);
}

std::vector<std::size_t> align_downbeats_to_beats(const std::vector<std::size_t>& beats,
                                                  const std::vector<std::size_t>& downbeats) {
    if (beats.empty()) {
        return downbeats;
    }
    std::vector<std::size_t> aligned;
    aligned.reserve(downbeats.size());
    for (std::size_t db : downbeats) {
        aligned.push_back(nearest_beat_frame(beats, db));
    }
    std::sort(aligned.begin(), aligned.end());
    aligned.erase(std::unique(aligned.begin(), aligned.end()), aligned.end());
    return aligned;
}

std::pair<std::size_t, std::size_t> infer_bpb_phase(const std::vector<std::size_t>& beats,
                                                     const std::vector<std::size_t>& downbeats,
                                                     const std::vector<std::size_t>& candidates,
                                                     const BeatitConfig& config) {
    std::size_t best_bpb = candidates.empty() ? config.dbn_beats_per_bar : candidates.front();
    std::size_t best_phase = 0;
    std::size_t best_hits = 0;
    if (beats.empty() || downbeats.empty()) {
        return std::make_pair(best_bpb, best_phase);
    }
    const bool trace_enabled = config.dbn_trace && beatit_should_log("debug");
    if (trace_enabled) {
        auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
        debug_stream << "DBN bpb inference: beats=" << beats.size()
                     << " downbeats=" << downbeats.size() << " candidates=";
        for (std::size_t bpb : candidates) {
            debug_stream << " " << bpb;
        }
    }
    std::unordered_map<std::size_t, std::size_t> beat_index;
    beat_index.reserve(beats.size());
    for (std::size_t i = 0; i < beats.size(); ++i) {
        beat_index[beats[i]] = i;
    }
    std::unordered_map<std::size_t, std::size_t> hits_by_bpb;
    std::unordered_map<std::size_t, std::size_t> phase_by_bpb;
    hits_by_bpb.reserve(candidates.size());
    phase_by_bpb.reserve(candidates.size());
    for (std::size_t bpb : candidates) {
        if (bpb == 0) {
            continue;
        }
        std::size_t phase = 0;
        auto it = beat_index.find(downbeats.front());
        if (it != beat_index.end()) {
            phase = it->second % bpb;
        }
        std::size_t hits = 0;
        for (std::size_t db : downbeats) {
            auto idx = beat_index.find(db);
            if (idx != beat_index.end() && (idx->second % bpb) == phase) {
                ++hits;
            }
        }
        hits_by_bpb[bpb] = hits;
        phase_by_bpb[bpb] = phase;
        if (trace_enabled) {
            BEATIT_LOG_DEBUG("DBN bpb inference: bpb=" << bpb
                             << " phase=" << phase
                             << " hits=" << hits);
        }
        if (hits > best_hits) {
            best_hits = hits;
            best_bpb = bpb;
            best_phase = phase;
        }
    }
    if (candidates.size() >= 2 && best_bpb == 3 &&
        hits_by_bpb.count(3) && hits_by_bpb.count(4)) {
        const double hits3 = static_cast<double>(hits_by_bpb[3]);
        const double hits4 = static_cast<double>(hits_by_bpb[4]);
        if (hits4 > 0.0 && hits3 < (hits4 * 1.5)) {
            best_bpb = 4;
            best_phase = phase_by_bpb[4];
            best_hits = hits_by_bpb[4];
            if (trace_enabled) {
                BEATIT_LOG_DEBUG("DBN bpb inference: biasing to 4/4 (hits3="
                                 << hits3 << " hits4=" << hits4 << ")");
            }
        }
    }
    if (trace_enabled) {
        BEATIT_LOG_DEBUG("DBN bpb inference: best_bpb=" << best_bpb
                         << " best_phase=" << best_phase
                         << " best_hits=" << best_hits);
    }
    return std::make_pair(best_bpb, best_phase);
}

std::vector<std::size_t> project_downbeats_from_beats(const std::vector<std::size_t>& beats,
                                                      std::size_t bpb,
                                                      std::size_t phase) {
    std::vector<std::size_t> downbeats;
    if (beats.empty() || bpb == 0) {
        return downbeats;
    }
    downbeats.reserve((beats.size() / bpb) + 1);
    for (std::size_t i = 0; i < beats.size(); ++i) {
        if ((i % bpb) == phase) {
            downbeats.push_back(beats[i]);
        }
    }
    return downbeats;
}

std::size_t guard_projected_downbeat_phase(const std::vector<std::size_t>& projected_frames,
                                           const std::vector<float>& downbeat_activation,
                                           std::size_t projected_bpb,
                                           std::size_t inferred_phase) {
    if (projected_frames.empty() ||
        projected_bpb < 2 ||
        downbeat_activation.empty()) {
        return inferred_phase;
    }

    auto phase_score = [&](std::size_t phase) {
        double sum = 0.0;
        double weight = 0.0;
        std::size_t picks = 0;
        std::size_t ordinal = 0;
        for (std::size_t i = phase;
             i < projected_frames.size() && picks < 24;
             i += projected_bpb, ++ordinal) {
            const std::size_t frame = projected_frames[i];
            if (frame >= downbeat_activation.size()) {
                continue;
            }
            float value = downbeat_activation[frame];
            if (frame > 0) {
                value = std::max(value, downbeat_activation[frame - 1]);
            }
            if (frame + 1 < downbeat_activation.size()) {
                value = std::max(value, downbeat_activation[frame + 1]);
            }
            const double w = std::exp(-0.12 * static_cast<double>(ordinal));
            sum += static_cast<double>(value) * w;
            weight += w;
            picks += 1;
        }
        return weight > 0.0 ? (sum / weight) : 0.0;
    };

    const std::size_t normalized_inferred_phase = inferred_phase % projected_bpb;
    std::size_t best_phase = normalized_inferred_phase;
    double best_score = -1.0;
    std::vector<double> scores(projected_bpb, 0.0);
    for (std::size_t phase = 0; phase < projected_bpb; ++phase) {
        const double score = phase_score(phase);
        scores[phase] = score;
        if (score > best_score) {
            best_score = score;
            best_phase = phase;
        }
    }

    const double inferred_score = scores[normalized_inferred_phase];
    const double phase0_score = scores[0];
    const bool inferred_strong =
        inferred_score > (phase0_score + 0.06) &&
        inferred_score > (phase0_score * 1.15);
    if (best_phase == 0 && normalized_inferred_phase != 0 && !inferred_strong) {
        BEATIT_LOG_DEBUG("DBN projected phase guard: inferred_phase="
                         << normalized_inferred_phase
                         << " inferred_score=" << inferred_score
                         << " phase0_score=" << phase0_score
                         << " selected_phase=0");
        return 0;
    }

    return normalized_inferred_phase;
}

WindowSummary summarize_window(const std::vector<float>& activation,
                              std::size_t start,
                              std::size_t end,
                              float floor_value) {
    WindowSummary summary;
    if (start >= end || end > activation.size()) {
        return summary;
    }
    summary.frames = end - start;
    summary.min = std::numeric_limits<float>::infinity();
    summary.max = -std::numeric_limits<float>::infinity();
    double total = 0.0;
    for (std::size_t i = start; i < end; ++i) {
        const float value = activation[i];
        summary.min = std::min(summary.min, value);
        summary.max = std::max(summary.max, value);
        total += value;
        if (value >= floor_value) {
            ++summary.above;
        }
    }
    summary.mean = (summary.frames > 0) ? (total / static_cast<double>(summary.frames)) : 0.0;
    if (!std::isfinite(summary.min)) {
        summary.min = 0.0f;
    }
    if (!std::isfinite(summary.max)) {
        summary.max = 0.0f;
    }
    return summary;
}

double median_interval_bpm(const std::vector<std::size_t>& frames, double fps) {
    if (frames.size() < 2 || fps <= 0.0) {
        return 0.0;
    }
    std::vector<double> intervals;
    intervals.reserve(frames.size() - 1);
    for (std::size_t i = 1; i < frames.size(); ++i) {
        const std::size_t delta = frames[i] - frames[i - 1];
        if (delta > 0) {
            intervals.push_back(static_cast<double>(delta));
        }
    }
    if (intervals.empty()) {
        return 0.0;
    }
    std::nth_element(intervals.begin(),
                     intervals.begin() + intervals.size() / 2,
                     intervals.end());
    const double median = intervals[intervals.size() / 2];
    if (median <= 1.0) {
        return 0.0;
    }
    return (60.0 * fps) / median;
}

} // namespace beatit::detail
