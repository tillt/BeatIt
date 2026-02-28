//
//  sparse_interior_select_test.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "sparse/interior_select.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {

beatit::detail::SparseWindowPhaseMetrics make_metrics(double abs_ms,
                                                      double abs_ratio,
                                                      double signed_ratio,
                                                      double odd_even_ms,
                                                      std::size_t count = 16) {
    beatit::detail::SparseWindowPhaseMetrics metrics;
    metrics.median_abs_ms = abs_ms;
    metrics.abs_limit_exceed_ratio = abs_ratio;
    metrics.signed_limit_exceed_ratio = signed_ratio;
    metrics.odd_even_gap_ms = odd_even_ms;
    metrics.count = count;
    return metrics;
}

bool test_scores_lower_for_cleaner_interior() {
    const auto clean = make_metrics(18.0, 0.05, 0.10, 4.0);
    const auto messy = make_metrics(55.0, 0.40, 0.50, 18.0);
    if (!(beatit::detail::score_sparse_interior_candidate(clean) <
          beatit::detail::score_sparse_interior_candidate(messy))) {
        std::cerr << "Sparse interior select test failed: cleaner window should score lower.\n";
        return false;
    }
    return true;
}

bool test_rejects_invalid_candidate() {
    beatit::detail::SparseWindowPhaseMetrics invalid;
    invalid.count = 4;
    const double score = beatit::detail::score_sparse_interior_candidate(invalid);
    if (std::isfinite(score)) {
        std::cerr << "Sparse interior select test failed: invalid candidate should score infinite.\n";
        return false;
    }
    return true;
}

bool test_rejects_wrapped_sign_candidate() {
    const auto wrapped = make_metrics(18.0, 0.10, 0.90, 5.0);
    const double score = beatit::detail::score_sparse_interior_candidate(wrapped);
    if (std::isfinite(score)) {
        std::cerr << "Sparse interior select test failed: wrapped-sign candidate should score infinite.\n";
        return false;
    }
    return true;
}

bool test_picks_best_candidate() {
    std::vector<beatit::detail::SparseInteriorCandidate> candidates{
        {30.0, make_metrics(42.0, 0.35, 0.45, 12.0)},
        {50.0, make_metrics(14.0, 0.00, 0.06, 5.0)},
        {70.0, make_metrics(20.0, 0.15, 0.20, 8.0)},
    };
    const auto picked = beatit::detail::pick_best_sparse_interior_candidate(candidates);
    if (picked.index != 1) {
        std::cerr << "Sparse interior select test failed: expected candidate index 1, got "
                  << picked.index << ".\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!test_scores_lower_for_cleaner_interior()) {
        return 1;
    }
    if (!test_rejects_invalid_candidate()) {
        return 1;
    }
    if (!test_rejects_wrapped_sign_candidate()) {
        return 1;
    }
    if (!test_picks_best_candidate()) {
        return 1;
    }

    std::cout << "Sparse interior select test passed.\n";
    return 0;
}
