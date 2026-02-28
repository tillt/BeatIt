//
//  helpers.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/helpers.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace beatit::detail {

std::vector<double> interpolated_peak_positions(const std::vector<float>& activation,
                                                const std::vector<std::size_t>& peaks);

std::vector<double> positive_intervals(const std::vector<double>& positions);

namespace {

double median_value(std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
    return values[mid];
}

std::vector<double> positive_frame_intervals(const std::vector<std::size_t>& frames) {
    std::vector<double> intervals;
    intervals.reserve(frames.size() > 1 ? frames.size() - 1 : 0);
    for (std::size_t i = 1; i < frames.size(); ++i) {
        if (frames[i] > frames[i - 1]) {
            intervals.push_back(static_cast<double>(frames[i] - frames[i - 1]));
        }
    }
    return intervals;
}

std::vector<double> positive_interpolated_peak_intervals(const std::vector<float>& activation,
                                                         const std::vector<std::size_t>& peaks) {
    return positive_intervals(interpolated_peak_positions(activation, peaks));
}

} // namespace

double interpolate_peak_position(const std::vector<float>& activation, std::size_t frame) {
    double pos = static_cast<double>(frame);
    if (frame > 0 && frame + 1 < activation.size()) {
        const double prev = activation[frame - 1];
        const double curr = activation[frame];
        const double next = activation[frame + 1];
        const double denom = prev - 2.0 * curr + next;
        if (std::abs(denom) > 1e-9) {
            double offset = 0.5 * (prev - next) / denom;
            offset = std::max(-0.5, std::min(0.5, offset));
            pos += offset;
        }
    }
    return pos;
}

std::vector<double> interpolated_peak_positions(const std::vector<float>& activation,
                                                const std::vector<std::size_t>& peaks) {
    std::vector<double> positions;
    positions.reserve(peaks.size());
    for (std::size_t frame : peaks) {
        positions.push_back(interpolate_peak_position(activation, frame));
    }
    return positions;
}

std::vector<double> positive_intervals(const std::vector<double>& positions) {
    std::vector<double> intervals;
    intervals.reserve(positions.size() > 1 ? positions.size() - 1 : 0);
    for (std::size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] > positions[i - 1]) {
            intervals.push_back(positions[i] - positions[i - 1]);
        }
    }
    return intervals;
}

void fill_bpm_bins(IntervalStats& stats,
                   const std::vector<double>& intervals,
                   double fps,
                   double bpm_bin_width) {
    if (bpm_bin_width <= 0.0 || fps <= 0.0) {
        return;
    }
    std::unordered_map<int, int> bin_counts;
    for (double interval : intervals) {
        if (interval <= 0.0) {
            continue;
        }
        const double bpm = (60.0 * fps) / interval;
        const int bin = static_cast<int>(std::llround(bpm / bpm_bin_width));
        bin_counts[bin] += 1;
    }
    std::vector<std::pair<double, int>> bins;
    bins.reserve(bin_counts.size());
    for (const auto& entry : bin_counts) {
        bins.emplace_back(entry.first * bpm_bin_width, entry.second);
    }
    std::sort(bins.begin(), bins.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    const std::size_t keep = std::min<std::size_t>(5, bins.size());
    bins.resize(keep);
    stats.top_bpm_bins = std::move(bins);
}

void fill_interval_stats(IntervalStats& stats,
                         const std::vector<double>& intervals,
                         double fps,
                         double bpm_bin_width) {
    if (intervals.empty()) {
        return;
    }
    stats.count = intervals.size();
    stats.min_interval = *std::min_element(intervals.begin(), intervals.end());
    stats.max_interval = *std::max_element(intervals.begin(), intervals.end());
    stats.mean_interval = std::accumulate(intervals.begin(), intervals.end(), 0.0) /
                          static_cast<double>(intervals.size());
    std::vector<double> sorted = intervals;
    stats.median_interval = median_value(sorted);

    double variance = 0.0;
    for (double value : intervals) {
        const double diff = value - stats.mean_interval;
        variance += diff * diff;
    }
    stats.stdev_interval = std::sqrt(variance / static_cast<double>(intervals.size()));
    fill_bpm_bins(stats, intervals, fps, bpm_bin_width);
}

std::vector<std::size_t> pick_peaks(const std::vector<float>& activation,
                                    float threshold,
                                    std::size_t min_interval,
                                    std::size_t max_interval) {
    std::vector<std::size_t> peaks;
    if (activation.size() < 3) {
        return peaks;
    }

    std::size_t last_peak = 0;
    bool has_peak = false;
    for (std::size_t i = 1; i + 1 < activation.size(); ++i) {
        const float prev = activation[i - 1];
        const float curr = activation[i];
        const float next = activation[i + 1];
        if (curr >= threshold && curr >= prev && curr >= next) {
            if (!has_peak) {
                peaks.push_back(i);
                last_peak = i;
                has_peak = true;
                continue;
            }

            const std::size_t delta = i - last_peak;
            if (delta >= min_interval && delta <= max_interval) {
                peaks.push_back(i);
                last_peak = i;
                continue;
            }

            if (delta > max_interval) {
                // Allow a restart after long gaps instead of blocking forever.
                peaks.push_back(i);
                last_peak = i;
            }
        }
    }

    return peaks;
}

float score_peaks(const std::vector<float>& activation, const std::vector<std::size_t>& peaks) {
    float sum = 0.0f;
    for (std::size_t idx : peaks) {
        if (idx < activation.size()) {
            sum += activation[idx];
        }
    }
    return sum;
}

double median_interval_frames(const std::vector<std::size_t>& peaks) {
    if (peaks.size() < 2) {
        return 0.0;
    }
    std::vector<double> intervals = positive_frame_intervals(peaks);
    if (intervals.empty()) {
        return 0.0;
    }
    return median_value(intervals);
}

double median_interval_frames_interpolated(const std::vector<float>& activation,
                                           const std::vector<std::size_t>& peaks) {
    if (peaks.size() < 2 || activation.empty()) {
        return 0.0;
    }
    std::vector<double> intervals = positive_interpolated_peak_intervals(activation, peaks);
    if (intervals.empty()) {
        return 0.0;
    }
    return median_value(intervals);
}

double regression_interval_frames_interpolated(const std::vector<float>& activation,
                                               const std::vector<std::size_t>& peaks) {
    if (peaks.size() < 2 || activation.empty()) {
        return 0.0;
    }

    const std::vector<double> positions = interpolated_peak_positions(activation, peaks);

    const double n = static_cast<double>(positions.size());
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const double x = static_cast<double>(i);
        const double y = positions[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    const double denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-9) {
        return 0.0;
    }
    const double slope = (n * sum_xy - sum_x * sum_y) / denom;
    return slope > 0.0 ? slope : 0.0;
}

std::vector<std::size_t> filter_short_intervals(const std::vector<std::size_t>& frames,
                                                double min_interval_frames) {
    if (frames.size() < 2 || min_interval_frames <= 0.0) {
        return frames;
    }
    std::vector<std::size_t> filtered;
    filtered.reserve(frames.size());
    filtered.push_back(frames.front());
    for (std::size_t i = 1; i < frames.size(); ++i) {
        const std::size_t prev = filtered.back();
        const std::size_t curr = frames[i];
        if (curr > prev) {
            const double interval = static_cast<double>(curr - prev);
            if (interval >= min_interval_frames) {
                filtered.push_back(curr);
            }
        }
    }
    return filtered;
}

IntervalStats interval_stats_interpolated(const std::vector<float>& activation,
                                          const std::vector<std::size_t>& peaks,
                                          double fps,
                                          double bpm_bin_width) {
    IntervalStats stats;
    if (peaks.size() < 2 || activation.empty() || fps <= 0.0) {
        return stats;
    }

    std::vector<double> intervals = positive_interpolated_peak_intervals(activation, peaks);
    if (intervals.empty()) {
        return stats;
    }

    fill_interval_stats(stats, intervals, fps, bpm_bin_width);
    return stats;
}

IntervalStats interval_stats_frames(const std::vector<std::size_t>& frames,
                                    double fps,
                                    double bpm_bin_width) {
    IntervalStats stats;
    if (frames.size() < 2 || fps <= 0.0) {
        return stats;
    }

    std::vector<double> intervals = positive_frame_intervals(frames);
    if (intervals.empty()) {
        return stats;
    }

    fill_interval_stats(stats, intervals, fps, bpm_bin_width);
    return stats;
}

void trace_grid_peak_alignment(const std::vector<std::size_t>& beat_grid,
                               const std::vector<std::size_t>& downbeat_grid,
                               const std::vector<float>& beat_activation,
                               const std::vector<float>& downbeat_activation,
                               float activation_floor,
                               double fps) {
    if (fps <= 0.0) {
        return;
    }
    auto collect_peaks = [](const std::vector<float>& activation,
                            float floor,
                            std::vector<std::size_t>& peaks_out) {
        peaks_out.clear();
        if (activation.size() < 3) {
            return;
        }
        const std::size_t end = activation.size() - 1;
        for (std::size_t i = 1; i < end; ++i) {
            const float prev = activation[i - 1];
            const float curr = activation[i];
            const float next = activation[i + 1];
            if (curr >= floor && curr >= prev && curr >= next) {
                peaks_out.push_back(i);
            }
        }
    };

    auto compute_offsets = [&](const std::vector<std::size_t>& grid,
                               const std::vector<std::size_t>& peaks,
                               const char* label) {
        if (grid.empty() || peaks.empty()) {
            BEATIT_LOG_DEBUG("DBN align: " << label
                                           << "_peak_offset_s mean=nan std=nan count=0");
            return;
        }
        double sum = 0.0;
        double sum_sq = 0.0;
        std::size_t count = 0;
        for (const auto frame : grid) {
            auto it = std::lower_bound(peaks.begin(), peaks.end(), frame);
            std::size_t best = *peaks.begin();
            if (it == peaks.end()) {
                best = peaks.back();
            } else {
                best = *it;
                if (it != peaks.begin()) {
                    const std::size_t prev = *(it - 1);
                    if (frame - prev < best - frame) {
                        best = prev;
                    }
                }
            }
            const double delta = (static_cast<double>(best) -
                                  static_cast<double>(frame)) / fps;
            sum += delta;
            sum_sq += delta * delta;
            count += 1;
            if (count >= 64) {
                break;
            }
        }
        const double mean = sum / static_cast<double>(count);
        const double var = (sum_sq / static_cast<double>(count)) - mean * mean;
        const double stddev = var > 0.0 ? std::sqrt(var) : 0.0;
        BEATIT_LOG_DEBUG("DBN align: " << label
                                       << "_peak_offset_s mean=" << mean
                                       << " std=" << stddev
                                       << " count=" << count);
    };

    std::vector<std::size_t> beat_peaks;
    std::vector<std::size_t> downbeat_peaks;
    collect_peaks(beat_activation, activation_floor, beat_peaks);
    collect_peaks(downbeat_activation, activation_floor, downbeat_peaks);
    compute_offsets(beat_grid, beat_peaks, "beat");
    compute_offsets(downbeat_grid, downbeat_peaks, "downbeat");
}


} // namespace beatit::detail
