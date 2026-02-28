//
//  grid_anchor.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "grid_anchor.h"

#include "beatit/logging.hpp"
#include "beatit/post/helpers.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace beatit::detail {

std::size_t choose_seed_start_peak(const GridAnchorSeed& anchor,
                                   const DBNDecodeResult& decoded) {
    if (decoded.beat_frames.empty()) {
        return 0;
    }
    std::size_t start_peak = std::min(decoded.beat_frames.front(), anchor.earliest_peak);
    if (anchor.strongest_peak_value >= anchor.activation_floor) {
        start_peak = anchor.strongest_peak;
    }
    return start_peak;
}

double choose_grid_start_frame(const GridAnchorSeed& anchor,
                               const DBNDecodeResult& decoded,
                               bool reliable_downbeat_start,
                               double step_frames,
                               double fps) {
    if (decoded.beat_frames.empty()) {
        return 0.0;
    }

    const std::size_t start =
        reliable_downbeat_start && !decoded.downbeat_frames.empty()
        ? decoded.downbeat_frames.front()
        : std::min(decoded.beat_frames.front(),
                   std::min(anchor.earliest_peak, anchor.earliest_downbeat_peak));

    double grid_start = static_cast<double>(start);
    if (anchor.earliest_downbeat_peak > 0 && anchor.earliest_downbeat_peak < start) {
        grid_start = static_cast<double>(anchor.earliest_downbeat_peak);
    }

    if (!reliable_downbeat_start && anchor.strongest_peak_value >= anchor.activation_floor) {
        grid_start = static_cast<double>(anchor.strongest_peak);
    }

    if (!reliable_downbeat_start && anchor.earliest_downbeat_peak > 0 && step_frames > 1.0) {
        const double offset = static_cast<double>(anchor.earliest_downbeat_peak) - grid_start;
        const double half_step = step_frames * 0.5;
        if (std::abs(offset) <= half_step || anchor.earliest_downbeat_peak < grid_start) {
            grid_start = static_cast<double>(anchor.earliest_downbeat_peak);
        }
    }

    if (fps > 0.0) {
        constexpr double kGridStartAdvanceSeconds = 0.06;
        grid_start -= kGridStartAdvanceSeconds * fps;
    }

    return std::max(0.0, grid_start);
}

GridAnchorSeed seed_grid_anchor(DBNDecodeResult& decoded,
                                const CoreMLResult& result,
                                const BeatitConfig& config,
                                bool use_window,
                                std::size_t window_start,
                                std::size_t used_frames,
                                double base_interval) {
    GridAnchorSeed anchor;
    if (decoded.beat_frames.empty()) {
        return anchor;
    }
    anchor.earliest_peak = decoded.beat_frames.front();
    anchor.earliest_downbeat_peak = decoded.beat_frames.front();
    anchor.strongest_peak = decoded.beat_frames.front();
    anchor.activation_floor = std::max(0.01f, config.activation_threshold * 0.1f);
    float earliest_downbeat_value = 0.0f;
    if (base_interval <= 1.0 || decoded.beat_frames.empty()) {
        return anchor;
    }
    const std::size_t peak_search_start = use_window ? window_start : 0;
    const std::size_t peak_search_end = use_window
        ? std::min<std::size_t>(used_frames - 1,
                                window_start + static_cast<std::size_t>(std::llround(base_interval)))
        : std::min<std::size_t>(used_frames - 1,
                                static_cast<std::size_t>(std::llround(base_interval)));
    if (!result.beat_activation.empty()) {
        if (peak_search_start + 1 <= peak_search_end) {
            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                const float prev = result.beat_activation[i - 1];
                const float curr = result.beat_activation[i];
                const float next = result.beat_activation[i + 1];
                if (curr >= anchor.activation_floor && curr >= prev && curr >= next) {
                    anchor.earliest_peak = i;
                    break;
                }
            }
        }
        if (peak_search_start + 1 <= peak_search_end) {
            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                const float prev = result.beat_activation[i - 1];
                const float curr = result.beat_activation[i];
                const float next = result.beat_activation[i + 1];
                if (curr >= anchor.activation_floor && curr >= prev && curr >= next) {
                    if (curr > anchor.strongest_peak_value) {
                        anchor.strongest_peak_value = curr;
                        anchor.strongest_peak = i;
                    }
                }
            }
        }
        if (anchor.strongest_peak_value < 0.0f && anchor.earliest_peak < result.beat_activation.size()) {
            anchor.strongest_peak = anchor.earliest_peak;
            anchor.strongest_peak_value = result.beat_activation[anchor.earliest_peak];
        }
    }
    if (!result.downbeat_activation.empty()) {
        float max_downbeat = 0.0f;
        for (std::size_t i = peak_search_start; i <= peak_search_end; ++i) {
            max_downbeat = std::max(max_downbeat, result.downbeat_activation[i]);
        }
        const float onset_threshold =
            std::max(anchor.activation_floor, max_downbeat * config.dbn_downbeat_phase_peak_ratio);
        for (std::size_t i = peak_search_start; i <= peak_search_end; ++i) {
            const float curr = result.downbeat_activation[i];
            if (curr >= onset_threshold) {
                anchor.earliest_downbeat_peak = i;
                earliest_downbeat_value = curr;
                break;
            }
        }
    }
    auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
    debug_stream << "DBN grid: earliest_peak=" << anchor.earliest_peak
                 << " earliest_downbeat_peak=" << anchor.earliest_downbeat_peak
                 << " earliest_downbeat_value=" << earliest_downbeat_value
                 << " strongest_peak=" << anchor.strongest_peak
                 << " strongest_peak_value=" << anchor.strongest_peak_value
                 << " activation_floor=" << anchor.activation_floor;
    const std::size_t start_peak = choose_seed_start_peak(anchor, decoded);
    std::vector<std::size_t> forward = fill_peaks_with_grid(result.beat_activation,
                                                            start_peak,
                                                            used_frames - 1,
                                                            base_interval,
                                                            anchor.activation_floor);
    std::vector<std::size_t> backward;
    double cursor = static_cast<double>(start_peak) - base_interval;
    const std::size_t window =
        static_cast<std::size_t>(std::max(1.0, std::round(base_interval * 0.25)));
    while (cursor >= 0.0) {
        const std::size_t center = static_cast<std::size_t>(std::llround(cursor));
        const std::size_t start = center > window ? center - window : 0;
        const std::size_t end = std::min(used_frames - 1, center + window);
        float best_value = -1.0f;
        std::size_t best_index = center;
        for (std::size_t k = start; k <= end; ++k) {
            const float value = result.beat_activation[k];
            if (value > best_value) {
                best_value = value;
                best_index = k;
            }
        }
        std::size_t chosen = best_index;
        if (best_value < anchor.activation_floor) {
            chosen = center;
        }
        backward.push_back(chosen);
        if (cursor < base_interval) {
            break;
        }
        cursor -= base_interval;
    }
    std::sort(backward.begin(), backward.end());
    std::vector<std::size_t> combined;
    combined.reserve(backward.size() + forward.size());
    combined.insert(combined.end(), backward.begin(), backward.end());
    if (!forward.empty() && (combined.empty() || combined.back() != forward.front())) {
        combined.insert(combined.end(), forward.begin(), forward.end());
    } else if (forward.size() > 1) {
        combined.insert(combined.end(), forward.begin() + 1, forward.end());
    }
    decoded.beat_frames = std::move(combined);
    return anchor;
}

} // namespace beatit::detail
