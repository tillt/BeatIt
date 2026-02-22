//
//  refiner_constant_regions.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "refiner_constant_internal.h"

#include <algorithm>
#include <cmath>

namespace beatit {
namespace detail {

double round_bpm_within_range(double min_bpm, double center_bpm, double max_bpm) {
    double snap_bpm = std::round(center_bpm);
    if (snap_bpm > min_bpm && snap_bpm < max_bpm) {
        return snap_bpm;
    }

    const double round_bpm_width = max_bpm - min_bpm;
    if (round_bpm_width > 0.5) {
        if (center_bpm < 85.0) {
            return std::round(center_bpm * 2.0) / 2.0;
        }
        if (center_bpm > 127.0) {
            return std::round(center_bpm / 3.0 * 2.0) * 3.0 / 2.0;
        }
    }

    if (round_bpm_width > (1.0 / 12.0)) {
        return std::round(center_bpm * 12.0) / 12.0;
    }

    snap_bpm = std::round(center_bpm * 12.0) / 12.0;
    if (snap_bpm > min_bpm && snap_bpm < max_bpm) {
        return snap_bpm;
    }
    return center_bpm;
}

std::vector<BeatConstRegion> retrieve_constant_regions(const std::vector<unsigned long long>& coarse_beats,
                                                       double frame_rate,
                                                       const ConstantBeatRefinerConfig& config) {
    std::vector<BeatConstRegion> regions;
    if (coarse_beats.empty()) {
        return regions;
    }

    const double max_error = config.max_phase_error_seconds * frame_rate;
    const double max_error_sum = config.max_phase_error_sum_seconds * frame_rate;

    const auto border_error = [&](std::size_t window_start,
                                  std::size_t window_end,
                                  double mean_length) -> double {
        if (window_end <= window_start + 2) {
            return 0.0;
        }
        const double left_len =
            static_cast<double>(coarse_beats[window_start + 1] - coarse_beats[window_start]);
        const double right_len =
            static_cast<double>(coarse_beats[window_end] - coarse_beats[window_end - 1]);
        return std::fabs(left_len + right_len - (2.0 * mean_length));
    };

    std::size_t window_start = 0;
    std::size_t window_end = coarse_beats.size() - 1;

    while (window_start < coarse_beats.size() - 1) {
        const double mean_length =
            static_cast<double>(coarse_beats[window_end] - coarse_beats[window_start]) /
            static_cast<double>(window_end - window_start);

        std::size_t outliers = 0;
        double expected = static_cast<double>(coarse_beats[window_start]);
        double drift = 0.0;
        std::size_t probe = window_start + 1;

        for (; probe <= window_end; ++probe) {
            expected += mean_length;
            const double error = expected - static_cast<double>(coarse_beats[probe]);
            drift += error;
            if (std::fabs(drift) > max_error_sum) {
                break;
            }
            if (std::fabs(error) > max_error) {
                outliers++;
                if (outliers > config.max_outliers || probe == window_start + 1) {
                    break;
                }
            }
        }

        if (probe > window_end && border_error(window_start, window_end, mean_length) <= max_error / 2.0) {
            regions.push_back({static_cast<std::size_t>(coarse_beats[window_start]), mean_length});
            window_start = window_end;
            window_end = coarse_beats.size() - 1;
            continue;
        }

        if (window_end == 0) {
            break;
        }
        window_end--;
    }

    regions.push_back({static_cast<std::size_t>(coarse_beats.back()), 0.0});
    return regions;
}

double make_constant_bpm(const std::vector<BeatConstRegion>& regions,
                         double frame_rate,
                         const ConstantBeatRefinerConfig& config,
                         long long* out_first_beat,
                         std::size_t initial_silence_frames) {
    if (regions.size() < 2) {
        return 0.0;
    }

    int mid_region_index = 0;
    double longest_region_length = 0.0;
    double longest_region_beat_length = 0.0;
    int longest_region_number_of_beats = 0;

    for (std::size_t i = 0; i + 1 < regions.size(); ++i) {
        const double length =
            static_cast<double>(regions[i + 1].first_beat_frame - regions[i].first_beat_frame);
        if (regions[i].beat_length <= 0.0) {
            continue;
        }
        const int beat_count = static_cast<int>((length / regions[i].beat_length) + 0.5);
        if (beat_count > longest_region_number_of_beats) {
            longest_region_length = length;
            longest_region_beat_length = regions[i].beat_length;
            longest_region_number_of_beats = beat_count;
            mid_region_index = static_cast<int>(i);
        }
    }

    if (longest_region_length == 0.0) {
        return 0.0;
    }

    double longest_region_beat_length_min =
        longest_region_beat_length -
        ((config.max_phase_error_seconds * frame_rate) / longest_region_number_of_beats);
    double longest_region_beat_length_max =
        longest_region_beat_length +
        ((config.max_phase_error_seconds * frame_rate) / longest_region_number_of_beats);

    int start_region_index = mid_region_index;

    for (int i = 0; i < mid_region_index; ++i) {
        const double length =
            static_cast<double>(regions[i + 1].first_beat_frame - regions[i].first_beat_frame);
        const int number_of_beats = static_cast<int>((length / regions[i].beat_length) + 0.5);
        if (number_of_beats < static_cast<int>(config.min_region_beats)) {
            continue;
        }
        const double this_region_beat_length_min =
            regions[i].beat_length -
            ((config.max_phase_error_seconds * frame_rate) / number_of_beats);
        const double this_region_beat_length_max =
            regions[i].beat_length +
            ((config.max_phase_error_seconds * frame_rate) / number_of_beats);
        if (longest_region_beat_length > this_region_beat_length_min &&
            longest_region_beat_length < this_region_beat_length_max) {
            const double new_longest_region_length =
                static_cast<double>(regions[mid_region_index + 1].first_beat_frame -
                                    regions[i].first_beat_frame);

            const double beat_length_min =
                std::max(longest_region_beat_length_min, this_region_beat_length_min);
            const double beat_length_max =
                std::min(longest_region_beat_length_max, this_region_beat_length_max);

            const int max_number_of_beats =
                static_cast<int>(std::round(new_longest_region_length / beat_length_min));
            const int min_number_of_beats =
                static_cast<int>(std::round(new_longest_region_length / beat_length_max));
            if (min_number_of_beats != max_number_of_beats) {
                continue;
            }
            const int number_of_beats = min_number_of_beats;
            const double new_beat_length = new_longest_region_length / number_of_beats;
            if (new_beat_length > longest_region_beat_length_min &&
                new_beat_length < longest_region_beat_length_max) {
                longest_region_length = new_longest_region_length;
                longest_region_beat_length = new_beat_length;
                longest_region_number_of_beats = number_of_beats;
                longest_region_beat_length_min =
                    longest_region_beat_length -
                    ((config.max_phase_error_seconds * frame_rate) / longest_region_number_of_beats);
                longest_region_beat_length_max =
                    longest_region_beat_length +
                    ((config.max_phase_error_seconds * frame_rate) / longest_region_number_of_beats);
                start_region_index = i;
                break;
            }
        }
    }

    for (std::size_t i = regions.size() - 2; i > static_cast<std::size_t>(mid_region_index); --i) {
        const double length =
            static_cast<double>(regions[i + 1].first_beat_frame - regions[i].first_beat_frame);
        const int number_of_beats = static_cast<int>((length / regions[i].beat_length) + 0.5);
        if (number_of_beats < static_cast<int>(config.min_region_beats)) {
            continue;
        }
        const double this_region_beat_length_min =
            regions[i].beat_length -
            ((config.max_phase_error_seconds * frame_rate) / number_of_beats);
        const double this_region_beat_length_max =
            regions[i].beat_length +
            ((config.max_phase_error_seconds * frame_rate) / number_of_beats);
        if (longest_region_beat_length > this_region_beat_length_min &&
            longest_region_beat_length < this_region_beat_length_max) {
            const double new_longest_region_length =
                static_cast<double>(regions[i + 1].first_beat_frame -
                                    regions[start_region_index].first_beat_frame);
            const double beat_length_min =
                std::max(longest_region_beat_length_min, this_region_beat_length_min);
            const double beat_length_max =
                std::min(longest_region_beat_length_max, this_region_beat_length_max);
            const int max_number_of_beats =
                static_cast<int>(std::round(new_longest_region_length / beat_length_min));
            const int min_number_of_beats =
                static_cast<int>(std::round(new_longest_region_length / beat_length_max));
            if (min_number_of_beats != max_number_of_beats) {
                continue;
            }
            const int number_of_beats = min_number_of_beats;
            const double new_beat_length = new_longest_region_length / number_of_beats;
            if (new_beat_length > longest_region_beat_length_min &&
                new_beat_length < longest_region_beat_length_max) {
                longest_region_length = new_longest_region_length;
                longest_region_beat_length = new_beat_length;
                longest_region_number_of_beats = number_of_beats;
                break;
            }
        }
    }

    const double min_round_bpm = 60.0 * frame_rate / longest_region_beat_length_max;
    const double max_round_bpm = 60.0 * frame_rate / longest_region_beat_length_min;
    const double center_bpm = 60.0 * frame_rate / longest_region_beat_length;
    const double round_bpm = round_bpm_within_range(min_round_bpm, center_bpm, max_round_bpm);

    if (out_first_beat) {
        const double rounded_beat_length = 60.0 * frame_rate / round_bpm;
        const std::size_t first_measured_good_beat_frame =
            regions[static_cast<std::size_t>(start_region_index)].first_beat_frame;

        double possible_first_beat_offset =
            std::fmod(static_cast<double>(first_measured_good_beat_frame), rounded_beat_length);
        const double delta = rounded_beat_length - possible_first_beat_offset;
        const double error_threshold = rounded_beat_length / 4.0;
        if (delta < error_threshold) {
            possible_first_beat_offset -= rounded_beat_length;
        }

        const double skip_silence_frames =
            std::floor(static_cast<double>(initial_silence_frames) / rounded_beat_length) *
            rounded_beat_length;
        if (skip_silence_frames > 0.0) {
            possible_first_beat_offset += skip_silence_frames;
        }

        *out_first_beat = static_cast<long long>(std::llround(possible_first_beat_offset));
    }

    return round_bpm;
}

long long adjust_phase(long long first_beat,
                       double bpm,
                       const std::vector<unsigned long long>& coarse_beats,
                       double frame_rate,
                       const ConstantBeatRefinerConfig& config) {
    if (coarse_beats.empty() || bpm <= 0.0) {
        return first_beat;
    }

    const double beat_length = 60.0 * frame_rate / bpm;
    const double start_offset = std::fmod(static_cast<double>(first_beat), beat_length);
    double offset_adjust = 0.0;
    double offset_adjust_count = 0.0;

    for (unsigned long long frame : coarse_beats) {
        double offset = std::fmod(static_cast<double>(frame) - start_offset, beat_length);
        if (offset > beat_length / 2.0) {
            offset -= beat_length;
        }
        if (std::fabs(offset) < (config.max_phase_error_seconds * frame_rate)) {
            offset_adjust += offset;
            offset_adjust_count += 1.0;
        }
    }

    if (offset_adjust_count == 0.0) {
        return first_beat;
    }

    offset_adjust /= offset_adjust_count;
    return static_cast<long long>(std::llround(static_cast<double>(first_beat) + offset_adjust));
}

} // namespace detail
} // namespace beatit
