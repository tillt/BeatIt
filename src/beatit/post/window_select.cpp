//
//  window_select.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/window.h"

#include "beatit/post/helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace beatit::detail {

namespace {

std::size_t compute_window_frames(std::size_t total_frames, double window_seconds, double fps) {
    if (total_frames == 0 || window_seconds <= 0.0 || fps <= 0.0) {
        return 0;
    }
    return static_cast<std::size_t>(std::max(1.0, std::round(window_seconds * fps)));
}

std::pair<std::size_t, std::size_t> make_window_bounds(std::size_t total_frames,
                                                        std::size_t start,
                                                        std::size_t window_frames) {
    const std::size_t end = std::min(total_frames, start + window_frames);
    return std::make_pair(start, end);
}

std::array<std::pair<std::size_t, std::size_t>, 3> intro_mid_outro_windows(
    std::size_t total_frames, std::size_t window_frames) {
    const std::size_t intro_start = 0;
    const std::size_t mid_center = total_frames / 2;
    const std::size_t mid_start =
        (mid_center > (window_frames / 2)) ? (mid_center - (window_frames / 2)) : 0;
    const std::size_t outro_start = total_frames - window_frames;

    return {
        make_window_bounds(total_frames, intro_start, window_frames),
        make_window_bounds(total_frames, mid_start, window_frames),
        make_window_bounds(total_frames, outro_start, window_frames),
    };
}

template <typename ScoreFn>
std::pair<std::size_t, std::size_t> pick_best_intro_mid_outro(std::size_t total_frames,
                                                               std::size_t window_frames,
                                                               ScoreFn&& score_fn) {
    const auto windows = intro_mid_outro_windows(total_frames, window_frames);
    double best_score = -1.0;
    std::pair<std::size_t, std::size_t> best = windows.front();
    for (const auto& window : windows) {
        const double score = score_fn(window.first, window.second);
        if (score > best_score) {
            best_score = score;
            best = window;
        }
    }
    return best;
}

template <typename ScoreFn>
std::pair<std::size_t, std::size_t> pick_best_sliding_window(std::size_t total_frames,
                                                              std::size_t window_frames,
                                                              double min_accepted_score,
                                                              ScoreFn&& score_fn) {
    const std::size_t step = std::max<std::size_t>(1, window_frames / 4);
    double best_score = -1.0;
    std::size_t best_start = 0;
    for (std::size_t start = 0; start + window_frames <= total_frames; start += step) {
        const double score = score_fn(start, start + window_frames);
        if (score > best_score) {
            best_score = score;
            best_start = start;
        }
    }
    if (best_score <= min_accepted_score) {
        return {0, total_frames};
    }
    return {best_start, best_start + window_frames};
}

} // namespace

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
    if (activation.empty()) {
        return {0, activation.size()};
    }
    const std::size_t total_frames = activation.size();
    const std::size_t window_frames = compute_window_frames(total_frames, window_seconds, fps);
    if (window_frames == 0) {
        return {0, total_frames};
    }
    if (window_frames >= total_frames) {
        return {0, total_frames};
    }

    const auto score_window = [&](std::size_t start, std::size_t end) {
        return window_tempo_score(activation,
                                  start,
                                  end,
                                  min_bpm,
                                  max_bpm,
                                  peak_threshold,
                                  fps);
    };

    if (intro_mid_outro) {
        return pick_best_intro_mid_outro(total_frames, window_frames, score_window);
    }

    return pick_best_sliding_window(total_frames, window_frames, 1e-6, score_window);
}

std::pair<std::size_t, std::size_t> select_dbn_window_energy(const std::vector<float>& energy,
                                                              double window_seconds,
                                                              bool intro_mid_outro,
                                                              double fps) {
    if (energy.empty()) {
        return {0, energy.size()};
    }
    const std::size_t total_frames = energy.size();
    const std::size_t window_frames = compute_window_frames(total_frames, window_seconds, fps);
    if (window_frames == 0) {
        return {0, total_frames};
    }
    if (window_frames >= total_frames) {
        return {0, total_frames};
    }

    auto mean_energy = [&](std::size_t start, std::size_t end) {
        double sum = 0.0;
        for (std::size_t i = start; i < end; ++i) {
            sum += static_cast<double>(energy[i]);
        }
        const double denom = std::max<std::size_t>(1, end - start);
        return sum / static_cast<double>(denom);
    };

    if (intro_mid_outro) {
        return pick_best_intro_mid_outro(total_frames, window_frames, mean_energy);
    }

    return pick_best_sliding_window(total_frames, window_frames, 1e-9, mean_energy);
}

} // namespace beatit::detail
