//
//  grid_fill.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/helpers.h"

#include <algorithm>
#include <cmath>

namespace beatit::detail {

std::vector<std::size_t> fill_peaks_with_grid(const std::vector<float>& activation,
                                              std::size_t start_peak,
                                              std::size_t last_active_frame,
                                              double interval,
                                              float activation_floor) {
    std::vector<std::size_t> filled;
    if (activation.empty() || interval <= 1.0 || last_active_frame <= start_peak) {
        if (start_peak < activation.size()) {
            filled.push_back(start_peak);
        }
        return filled;
    }

    const std::size_t frames = activation.size();
    const std::size_t tail_end = std::min(last_active_frame, frames - 1);
    const std::size_t window = static_cast<std::size_t>(std::max(1.0, std::round(interval * 0.25)));
    const double min_spacing = interval * 0.5;

    filled.push_back(start_peak);
    double cursor = static_cast<double>(start_peak) + interval;
    while (cursor <= static_cast<double>(tail_end)) {
        const std::size_t center = static_cast<std::size_t>(std::llround(cursor));
        const std::size_t start = center > window ? center - window : 0;
        const std::size_t end = std::min(frames - 1, center + window);
        float best_value = -1.0f;
        std::size_t best_index = center;
        for (std::size_t k = start; k <= end; ++k) {
            const float value = activation[k];
            if (value > best_value) {
                best_value = value;
                best_index = k;
            }
        }
        std::size_t chosen = best_index;
        if (best_value < activation_floor ||
            static_cast<double>(best_index - filled.back()) < min_spacing) {
            chosen = center;
        }
        if (chosen > filled.back()) {
            filled.push_back(chosen);
        }
        cursor += interval;
    }

    return filled;
}

std::vector<std::size_t> fill_peaks_with_gaps(const std::vector<float>& activation,
                                              const std::vector<std::size_t>& peaks,
                                              double fps,
                                              float activation_floor,
                                              std::size_t last_active_frame,
                                              double base_interval_frames,
                                              float gap_tolerance,
                                              float offbeat_tolerance,
                                              std::size_t window_beats) {
    if (activation.empty() || peaks.size() < 2 || fps <= 0.0) {
        return peaks;
    }
    std::vector<std::size_t> filled;
    filled.reserve(peaks.size());
    const std::size_t frames = activation.size();

    const double gap_tolerance_ratio = 1.0 + static_cast<double>(gap_tolerance);
    const double offbeat_tolerance_ratio = 1.0 - static_cast<double>(offbeat_tolerance);
    const std::size_t window_beats_clamped = std::max<std::size_t>(1, window_beats);
    const double min_spacing =
        base_interval_frames > 1.0 ? base_interval_frames * 0.5 : 1.0;

    std::vector<std::size_t> intervals;
    intervals.reserve(peaks.size() - 1);
    for (std::size_t i = 1; i < peaks.size(); ++i) {
        if (peaks[i] > peaks[i - 1]) {
            intervals.push_back(peaks[i] - peaks[i - 1]);
        } else {
            intervals.push_back(0);
        }
    }

    for (std::size_t i = 0; i + 1 < peaks.size(); ++i) {
        const std::size_t current = peaks[i];
        const std::size_t next = peaks[i + 1];
        if (filled.empty() || current > filled.back()) {
            filled.push_back(current);
        }
        if (next <= current + 1) {
            continue;
        }

        double left = 0.0;
        double right = 0.0;
        std::size_t left_count = 0;
        std::size_t right_count = 0;
        for (std::size_t w = 0; w < window_beats_clamped && i >= 1 + w; ++w) {
            const std::size_t idx = i - 1 - w;
            if (idx < intervals.size() && intervals[idx] > 0) {
                left += static_cast<double>(intervals[idx]);
                ++left_count;
            }
        }
        for (std::size_t w = 0; w < window_beats_clamped && i + w < intervals.size(); ++w) {
            const std::size_t idx = i + w;
            if (idx < intervals.size() && intervals[idx] > 0) {
                right += static_cast<double>(intervals[idx]);
                ++right_count;
            }
        }
        double nominal_interval = 0.0;
        if (left_count > 0 && right_count > 0) {
            nominal_interval = 0.5 * (left / left_count + right / right_count);
        } else if (left_count > 0) {
            nominal_interval = left / left_count;
        } else if (right_count > 0) {
            nominal_interval = right / right_count;
        }
        if (base_interval_frames > 1.0 &&
            (nominal_interval <= 1.0 || nominal_interval > base_interval_frames * 1.5)) {
            nominal_interval = base_interval_frames;
        }
        if (nominal_interval <= 1.0) {
            continue;
        }

        const double gap = static_cast<double>(next - current);
        if (gap < nominal_interval * offbeat_tolerance_ratio) {
            continue;
        }
        if (gap <= nominal_interval * gap_tolerance_ratio) {
            continue;
        }

        const double interval = nominal_interval;
        const std::size_t window = static_cast<std::size_t>(std::max(1.0, std::round(interval * 0.25)));
        double cursor = static_cast<double>(current) + interval;
        while (cursor < static_cast<double>(next)) {
            const double remaining = static_cast<double>(next) - cursor;
            if (remaining < interval * offbeat_tolerance_ratio) {
                break;
            }
            if (remaining <= interval * gap_tolerance_ratio) {
                break;
            }
            const double target = cursor;
            const std::size_t center = static_cast<std::size_t>(std::llround(target));
            const std::size_t start = center > window ? center - window : 0;
            const std::size_t end = std::min(frames - 1, center + window);
            float best_value = -1.0f;
            std::size_t best_index = center;
            for (std::size_t k = start; k <= end; ++k) {
                const float value = activation[k];
                if (value > best_value) {
                    best_value = value;
                    best_index = k;
                }
            }
            std::size_t chosen = best_index;
            if (best_value < activation_floor ||
                (filled.size() > 0 &&
                 static_cast<double>(best_index - filled.back()) < min_spacing)) {
                chosen = center;
            }
            if (chosen > current && chosen < next) {
                if (filled.empty() || chosen > filled.back()) {
                    filled.push_back(chosen);
                }
            }
            cursor += interval;
        }
    }

    const std::size_t last = peaks.back();
    if (filled.empty() || last > filled.back()) {
        filled.push_back(last);
    }

    double tail_interval = base_interval_frames;
    if (tail_interval <= 1.0 && peaks.size() >= 2 && peaks.back() > peaks[peaks.size() - 2]) {
        tail_interval = static_cast<double>(peaks.back() - peaks[peaks.size() - 2]);
    }
    if (tail_interval > 1.0 && last_active_frame > last) {
        const std::size_t tail_end = std::min(last_active_frame, frames - 1);
        if (tail_end > last + static_cast<std::size_t>(tail_interval * 0.5)) {
            const std::size_t window =
                static_cast<std::size_t>(std::max(1.0, std::round(tail_interval * 0.25)));
            double cursor = static_cast<double>(last) + tail_interval;
            while (cursor <= static_cast<double>(tail_end)) {
                const std::size_t center = static_cast<std::size_t>(std::llround(cursor));
                const std::size_t start = center > window ? center - window : 0;
                const std::size_t end = std::min(frames - 1, center + window);
                float best_value = -1.0f;
                std::size_t best_index = center;
                for (std::size_t k = start; k <= end; ++k) {
                    const float value = activation[k];
                    if (value > best_value) {
                        best_value = value;
                        best_index = k;
                    }
                }
                std::size_t chosen = best_index;
                if (best_value < activation_floor ||
                    static_cast<double>(best_index - filled.back()) < min_spacing) {
                    chosen = center;
                }
                if (chosen > filled.back()) {
                    filled.push_back(chosen);
                } else {
                    const std::size_t fallback =
                        filled.back() + static_cast<std::size_t>(std::llround(min_spacing));
                    if (fallback <= tail_end) {
                        filled.push_back(fallback);
                    } else {
                        break;
                    }
                }
                cursor += tail_interval;
            }
        }
    }
    return filled;
}

} // namespace beatit::detail
