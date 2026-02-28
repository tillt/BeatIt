//
//  interior_select.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "interior_select.h"

#include <algorithm>
#include <cmath>

namespace beatit {
namespace detail {

double score_sparse_interior_candidate(const SparseWindowPhaseMetrics& metrics) {
    if (metrics.count < 8 ||
        !std::isfinite(metrics.median_abs_ms) ||
        !std::isfinite(metrics.abs_limit_exceed_ratio) ||
        !std::isfinite(metrics.signed_limit_exceed_ratio) ||
        !std::isfinite(metrics.odd_even_gap_ms)) {
        return std::numeric_limits<double>::infinity();
    }

    return metrics.median_abs_ms +
           (60.0 * metrics.abs_limit_exceed_ratio) +
           (40.0 * metrics.signed_limit_exceed_ratio) +
           (0.25 * metrics.odd_even_gap_ms);
}

SparseInteriorPickResult pick_best_sparse_interior_candidate(
    const std::vector<SparseInteriorCandidate>& candidates) {
    SparseInteriorPickResult result;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const double score = score_sparse_interior_candidate(candidates[i].metrics);
        if (!(score < result.score)) {
            continue;
        }
        result.index = i;
        result.score = score;
    }
    return result;
}

} // namespace detail
} // namespace beatit
