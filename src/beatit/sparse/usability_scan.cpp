//
//  usability_scan.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "usability_scan.h"

#include <algorithm>
#include <cmath>

namespace beatit {
namespace detail {
namespace {

double clamp_unit(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double window_distance_penalty(const SparseUsabilityWindow& window,
                               const SparseUsabilityPickRequest& request) {
    return request.distance_weight * std::abs(window.start_seconds - request.target_seconds);
}

} // namespace

double score_sparse_usability_window(const SparseUsabilityFeatures& features) {
    const double silence_ratio = clamp_unit(features.silence_ratio);
    const double onset_density = clamp_unit(features.onset_density);
    const double periodicity = clamp_unit(features.periodicity);
    const double transient_strength = clamp_unit(features.transient_strength);
    const double instability = clamp_unit(features.instability);

    const double positive =
        (0.30 * onset_density) +
        (0.40 * periodicity) +
        (0.30 * transient_strength);
    const double penalty =
        (0.45 * silence_ratio) +
        (0.35 * instability);

    return positive - penalty;
}

bool sparse_window_is_usable(const SparseUsabilityFeatures& features) {
    return clamp_unit(features.silence_ratio) <= 0.45 &&
           clamp_unit(features.periodicity) >= 0.35 &&
           clamp_unit(features.transient_strength) >= 0.25 &&
           clamp_unit(features.instability) <= 0.55;
}

SparseUsabilityWindow build_sparse_usability_window(double start_seconds,
                                                    double duration_seconds,
                                                    const SparseUsabilityFeatures& features) {
    SparseUsabilityWindow window;
    window.start_seconds = start_seconds;
    window.duration_seconds = duration_seconds;
    window.features = features;
    window.score = score_sparse_usability_window(features);
    window.usable = sparse_window_is_usable(features);
    return window;
}

std::size_t pick_sparse_usability_window(const std::vector<SparseUsabilityWindow>& windows,
                                         const SparseUsabilityPickRequest& request) {
    std::size_t best_index = windows.size();
    double best_adjusted_score = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < windows.size(); ++i) {
        const auto& window = windows[i];
        if (!window.usable || window.score < request.min_score) {
            continue;
        }

        const double distance = std::abs(window.start_seconds - request.target_seconds);
        if (distance > request.max_snap_seconds) {
            continue;
        }

        const double adjusted_score = window.score - window_distance_penalty(window, request);
        if (adjusted_score > best_adjusted_score) {
            best_adjusted_score = adjusted_score;
            best_index = i;
        }
    }

    return best_index;
}

} // namespace detail
} // namespace beatit
