//
//  pp_helpers.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace beatit::detail {

struct IntervalStats {
    std::size_t count = 0;
    double min_interval = 0.0;
    double max_interval = 0.0;
    double mean_interval = 0.0;
    double median_interval = 0.0;
    double stdev_interval = 0.0;
    std::vector<std::pair<double, int>> top_bpm_bins;
};

std::vector<std::size_t> pick_peaks(const std::vector<float>& activation,
                                    float threshold,
                                    std::size_t min_interval,
                                    std::size_t max_interval);

float score_peaks(const std::vector<float>& activation,
                  const std::vector<std::size_t>& peaks);

double median_interval_frames(const std::vector<std::size_t>& peaks);

double median_interval_frames_interpolated(const std::vector<float>& activation,
                                           const std::vector<std::size_t>& peaks);

double regression_interval_frames_interpolated(const std::vector<float>& activation,
                                               const std::vector<std::size_t>& peaks);

std::vector<std::size_t> filter_short_intervals(const std::vector<std::size_t>& frames,
                                                double min_interval_frames);

IntervalStats interval_stats_interpolated(const std::vector<float>& activation,
                                          const std::vector<std::size_t>& peaks,
                                          double fps,
                                          double bpm_bin_width);

IntervalStats interval_stats_frames(const std::vector<std::size_t>& frames,
                                    double fps,
                                    double bpm_bin_width);

std::vector<std::size_t> fill_peaks_with_grid(const std::vector<float>& activation,
                                              std::size_t start_peak,
                                              std::size_t last_active_frame,
                                              double interval,
                                              float activation_floor);

std::vector<std::size_t> fill_peaks_with_gaps(const std::vector<float>& activation,
                                              const std::vector<std::size_t>& peaks,
                                              double fps,
                                              float activation_floor,
                                              std::size_t last_active_frame,
                                              double base_interval_frames,
                                              float gap_tolerance,
                                              float offbeat_tolerance,
                                              std::size_t window_beats);

void trace_grid_peak_alignment(const std::vector<std::size_t>& beat_grid,
                               const std::vector<std::size_t>& downbeat_grid,
                               const std::vector<float>& beat_activation,
                               const std::vector<float>& downbeat_activation,
                               float activation_floor,
                               double fps);

} // namespace beatit::detail
