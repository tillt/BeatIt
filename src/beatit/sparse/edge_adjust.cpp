//
//  edge_adjust.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "edge_adjust.h"

#include "beatit/sparse/waveform.h"
#include "refine_common.h"

#include <algorithm>
#include <cmath>

namespace beatit {
namespace detail {

double apply_sparse_edge_scale(std::vector<unsigned long long>& beats,
                               double ratio,
                               double min_ratio,
                               double max_ratio,
                               double min_delta) {
    if (beats.size() < 2) {
        return 1.0;
    }
    const double clamped_ratio = std::clamp(ratio, min_ratio, max_ratio);
    if (std::abs(clamped_ratio - 1.0) < min_delta) {
        return 1.0;
    }
    const long long anchor = static_cast<long long>(beats.front());
    for (std::size_t i = 0; i < beats.size(); ++i) {
        const long long current = static_cast<long long>(beats[i]);
        const double rel = static_cast<double>(current - anchor);
        const long long adjusted =
            anchor + static_cast<long long>(std::llround(rel * clamped_ratio));
        beats[i] = static_cast<unsigned long long>(std::max<long long>(0, adjusted));
    }
    return clamped_ratio;
}

double apply_sparse_edge_scale_from_back(std::vector<unsigned long long>& beats,
                                         double ratio,
                                         double min_ratio,
                                         double max_ratio,
                                         double min_delta) {
    if (beats.size() < 2) {
        return 1.0;
    }
    const double clamped_ratio = std::clamp(ratio, min_ratio, max_ratio);
    if (std::abs(clamped_ratio - 1.0) < min_delta) {
        return 1.0;
    }
    const long long anchor = static_cast<long long>(beats.back());
    for (std::size_t i = 0; i < beats.size(); ++i) {
        const long long current = static_cast<long long>(beats[i]);
        const double rel = static_cast<double>(current - anchor);
        const long long adjusted =
            anchor + static_cast<long long>(std::llround(rel * clamped_ratio));
        beats[i] = static_cast<unsigned long long>(std::max<long long>(0, adjusted));
    }
    return clamped_ratio;
}

double compute_sparse_edge_ratio(const std::vector<unsigned long long>& beats,
                                 const EdgeOffsetMetrics& first,
                                 const EdgeOffsetMetrics& last,
                                 double sample_rate) {
    const double base_step = sparse_median_frame_diff(beats);
    const double beats_span = static_cast<double>(beats.size() - 1);
    if (!(base_step > 0.0) || !(beats_span > 0.0)) {
        return 1.0;
    }
    const double err_delta_frames = ((last.median_ms - first.median_ms) * sample_rate) / 1000.0;
    const double per_beat_adjust = err_delta_frames / beats_span;
    return 1.0 + (per_beat_adjust / base_step);
}

bool apply_sparse_uniform_shift(std::vector<unsigned long long>& beats, long long shift_frames) {
    if (beats.empty() || shift_frames == 0) {
        return false;
    }
    for (std::size_t i = 0; i < beats.size(); ++i) {
        const long long shifted = static_cast<long long>(beats[i]) + shift_frames;
        beats[i] = static_cast<unsigned long long>(std::max<long long>(0, shifted));
    }
    return true;
}

bool should_accept_sparse_opening_anchor_guard(const SparseOpeningAnchorGuardInput& input) {
    const bool base_valid = std::isfinite(input.base_intro_abs_ms) &&
                            std::isfinite(input.base_outro_abs_ms);
    const bool candidate_valid = std::isfinite(input.candidate_intro_abs_ms) &&
                                 std::isfinite(input.candidate_outro_abs_ms);
    if (!base_valid || !candidate_valid) {
        return false;
    }

    const bool opening_is_problem = input.base_intro_abs_ms >= 45.0;
    const bool late_is_stable = input.base_outro_abs_ms <= 40.0;
    if (!opening_is_problem || !late_is_stable) {
        return false;
    }

    const double intro_improvement = input.base_intro_abs_ms - input.candidate_intro_abs_ms;
    const double outro_worsening = input.candidate_outro_abs_ms - input.base_outro_abs_ms;

    return intro_improvement >= 15.0 &&
           input.candidate_intro_abs_ms <= (input.base_intro_abs_ms - 10.0) &&
           outro_worsening <= 12.0 &&
           input.candidate_outro_abs_ms <= 50.0;
}

} // namespace detail
} // namespace beatit
