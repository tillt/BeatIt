//
//  window.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/pp_window.h"

#include "beatit/pp_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>

namespace beatit::detail {

double window_tempo_score(const std::vector<float>& activation,
                          std::size_t start,
                          std::size_t end,
                          float min_bpm,
                          float max_bpm,
                          float peak_threshold,
                          double fps) {
    if (end <= start || activation.empty() || fps <= 0.0) {
        return 0.0;
    }
    const double min_interval_frames =
        std::max(1.0, (60.0 * fps) / std::max(1.0f, max_bpm));
    const double max_interval_frames =
        std::max(1.0, (60.0 * fps) / std::max(1.0f, min_bpm));
    const std::size_t peak_min_interval =
        static_cast<std::size_t>(std::max(1.0, std::floor(min_interval_frames)));
    const std::size_t peak_max_interval =
        static_cast<std::size_t>(std::max<double>(peak_min_interval,
                                                  std::ceil(max_interval_frames)));

    std::vector<float> slice;
    slice.reserve(end - start);
    for (std::size_t i = start; i < end; ++i) {
        slice.push_back(activation[i]);
    }

    std::vector<std::size_t> peaks =
        pick_peaks(slice, peak_threshold, peak_min_interval, peak_max_interval);
    if (peaks.size() < 4) {
        return 0.0;
    }
    std::vector<double> intervals;
    intervals.reserve(peaks.size() - 1);
    for (std::size_t i = 1; i < peaks.size(); ++i) {
        if (peaks[i] > peaks[i - 1]) {
            intervals.push_back(static_cast<double>(peaks[i] - peaks[i - 1]));
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
    std::vector<double> deviations;
    deviations.reserve(intervals.size());
    for (double v : intervals) {
        deviations.push_back(std::abs(v - median));
    }
    std::nth_element(deviations.begin(),
                     deviations.begin() + deviations.size() / 2,
                     deviations.end());
    const double mad = deviations[deviations.size() / 2];
    const double consistency = 1.0 / (1.0 + (mad / median));
    return static_cast<double>(peaks.size()) * consistency;
}

std::pair<std::size_t, std::size_t> select_dbn_window(const std::vector<float>& activation,
                                                       double window_seconds,
                                                       bool intro_mid_outro,
                                                       float min_bpm,
                                                       float max_bpm,
                                                       float peak_threshold,
                                                       double fps) {
    if (activation.empty() || window_seconds <= 0.0 || fps <= 0.0) {
        return {0, activation.size()};
    }
    const std::size_t total_frames = activation.size();
    std::size_t window_frames =
        static_cast<std::size_t>(std::max(1.0, std::round(window_seconds * fps)));
    if (window_frames >= total_frames) {
        return {0, total_frames};
    }

    auto clamp_window = [&](std::size_t start) {
        const std::size_t end = std::min(total_frames, start + window_frames);
        return std::make_pair(start, end);
    };

    if (intro_mid_outro && total_frames > window_frames) {
        const std::size_t intro_start = 0;
        const std::size_t mid_center = total_frames / 2;
        const std::size_t mid_start =
            (mid_center > (window_frames / 2)) ? (mid_center - (window_frames / 2)) : 0;
        const std::size_t outro_start = total_frames - window_frames;

        const auto intro = clamp_window(intro_start);
        const auto mid = clamp_window(mid_start);
        const auto outro = clamp_window(outro_start);

        std::array<std::pair<std::size_t, std::size_t>, 3> windows = {intro, mid, outro};
        double best_score = -1.0;
        std::pair<std::size_t, std::size_t> best = intro;
        for (const auto& w : windows) {
            const double score =
                window_tempo_score(activation,
                                   w.first,
                                   w.second,
                                   min_bpm,
                                   max_bpm,
                                   peak_threshold,
                                   fps);
            if (score > best_score) {
                best_score = score;
                best = w;
            }
        }
        return best;
    }

    const std::size_t step = std::max<std::size_t>(1, window_frames / 4);
    double best_score = -1.0;
    std::size_t best_start = 0;
    for (std::size_t start = 0; start + window_frames <= total_frames; start += step) {
        const double score =
            window_tempo_score(activation,
                               start,
                               start + window_frames,
                               min_bpm,
                               max_bpm,
                               peak_threshold,
                               fps);
        if (score > best_score) {
            best_score = score;
            best_start = start;
        }
    }
    if (best_score <= 1e-6) {
        return {0, total_frames};
    }
    return {best_start, best_start + window_frames};
}

std::pair<std::size_t, std::size_t> select_dbn_window_energy(const std::vector<float>& energy,
                                                              double window_seconds,
                                                              bool intro_mid_outro,
                                                              double fps) {
    if (energy.empty() || window_seconds <= 0.0 || fps <= 0.0) {
        return {0, energy.size()};
    }
    const std::size_t total_frames = energy.size();
    const std::size_t window_frames =
        static_cast<std::size_t>(std::max(1.0, std::round(window_seconds * fps)));
    if (window_frames >= total_frames) {
        return {0, total_frames};
    }

    auto clamp_window = [&](std::size_t start) {
        const std::size_t end = std::min(total_frames, start + window_frames);
        return std::make_pair(start, end);
    };

    auto mean_energy = [&](std::size_t start, std::size_t end) {
        double sum = 0.0;
        for (std::size_t i = start; i < end; ++i) {
            sum += static_cast<double>(energy[i]);
        }
        const double denom = std::max<std::size_t>(1, end - start);
        return sum / static_cast<double>(denom);
    };

    if (intro_mid_outro && total_frames > window_frames) {
        const std::size_t intro_start = 0;
        const std::size_t mid_center = total_frames / 2;
        const std::size_t mid_start =
            (mid_center > (window_frames / 2)) ? (mid_center - (window_frames / 2)) : 0;
        const std::size_t outro_start = total_frames - window_frames;

        const auto intro = clamp_window(intro_start);
        const auto mid = clamp_window(mid_start);
        const auto outro = clamp_window(outro_start);

        std::array<std::pair<std::size_t, std::size_t>, 3> windows = {intro, mid, outro};
        double best_score = -1.0;
        std::pair<std::size_t, std::size_t> best = intro;
        for (const auto& w : windows) {
            const double score = mean_energy(w.first, w.second);
            if (score > best_score) {
                best_score = score;
                best = w;
            }
        }
        return best;
    }

    const std::size_t step = std::max<std::size_t>(1, window_frames / 4);
    double best_score = -1.0;
    std::size_t best_start = 0;
    for (std::size_t start = 0; start + window_frames <= total_frames; start += step) {
        const double score = mean_energy(start, start + window_frames);
        if (score > best_score) {
            best_score = score;
            best_start = start;
        }
    }
    if (best_score <= 1e-9) {
        return {0, total_frames};
    }
    return {best_start, best_start + window_frames};
}

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
        std::size_t best = beats.front();
        std::size_t best_dist = (db > best) ? (db - best) : (best - db);
        for (std::size_t beat : beats) {
            const std::size_t dist = (db > beat) ? (db - beat) : (beat - db);
            if (dist < best_dist) {
                best = beat;
                best_dist = dist;
            }
        }
        aligned.push_back(best);
    }
    std::sort(aligned.begin(), aligned.end());
    aligned.erase(std::unique(aligned.begin(), aligned.end()), aligned.end());
    return aligned;
}

std::pair<std::size_t, std::size_t> infer_bpb_phase(const std::vector<std::size_t>& beats,
                                                     const std::vector<std::size_t>& downbeats,
                                                     const std::vector<std::size_t>& candidates,
                                                     const CoreMLConfig& config) {
    std::size_t best_bpb = candidates.empty() ? config.dbn_beats_per_bar : candidates.front();
    std::size_t best_phase = 0;
    std::size_t best_hits = 0;
    if (beats.empty() || downbeats.empty()) {
        return std::make_pair(best_bpb, best_phase);
    }
    if (config.dbn_trace) {
        std::cerr << "DBN bpb inference: beats=" << beats.size()
                  << " downbeats=" << downbeats.size() << " candidates=";
        for (std::size_t bpb : candidates) {
            std::cerr << " " << bpb;
        }
        std::cerr << "\n";
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
        if (config.dbn_trace) {
            std::cerr << "DBN bpb inference: bpb=" << bpb
                      << " phase=" << phase
                      << " hits=" << hits << "\n";
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
            if (config.dbn_trace) {
                std::cerr << "DBN bpb inference: biasing to 4/4 (hits3="
                          << hits3 << " hits4=" << hits4 << ")\n";
            }
        }
    }
    if (config.dbn_trace) {
        std::cerr << "DBN bpb inference: best_bpb=" << best_bpb
                  << " best_phase=" << best_phase
                  << " best_hits=" << best_hits << "\n";
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
                                           std::size_t inferred_phase,
                                           bool verbose) {
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
        if (verbose) {
            std::cerr << "DBN projected phase guard: inferred_phase="
                      << normalized_inferred_phase
                      << " inferred_score=" << inferred_score
                      << " phase0_score=" << phase0_score
                      << " selected_phase=0"
                      << "\n";
        }
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
