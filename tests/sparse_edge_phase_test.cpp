//
//  sparse_edge_phase_test.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "sparse/edge_phase.h"

#include <iostream>

namespace {

bool test_phase_try_rejects_clean_to_bad_interior_swap() {
    beatit::detail::SparseEdgePhaseTryResult result;
    result.base_score = 598.058;
    result.minus_score = 570.809;
    result.plus_score = 674.245;
    result.base_between_abs_ms = 15.6236;
    result.base_middle_abs_ms = 206.939;
    result.minus_between_abs_ms = 130.703;
    result.minus_middle_abs_ms = 81.7687;
    result.plus_between_abs_ms = 150.136;
    result.plus_middle_abs_ms = 84.6712;

    const int selected = beatit::detail::select_sparse_edge_phase_candidate(result);
    if (selected != 0) {
        std::cerr << "Sparse edge phase test failed: expected base candidate, got "
                  << selected << ".\n";
        return false;
    }
    return true;
}

bool test_phase_try_accepts_clear_minus_improvement() {
    beatit::detail::SparseEdgePhaseTryResult result;
    result.base_score = 420.0;
    result.minus_score = 390.0;
    result.plus_score = 430.0;
    result.base_between_abs_ms = 120.0;
    result.base_middle_abs_ms = 140.0;
    result.minus_between_abs_ms = 60.0;
    result.minus_middle_abs_ms = 55.0;
    result.plus_between_abs_ms = 130.0;
    result.plus_middle_abs_ms = 135.0;

    const int selected = beatit::detail::select_sparse_edge_phase_candidate(result);
    if (selected != -1) {
        std::cerr << "Sparse edge phase test failed: expected minus candidate, got "
                  << selected << ".\n";
        return false;
    }
    return true;
}

bool test_phase_try_rejects_large_interior_tradeoff() {
    beatit::detail::SparseEdgePhaseTryResult result;
    result.base_score = 561.738;
    result.minus_score = 525.461;
    result.plus_score = 526.155;
    result.base_global_delta_ms = 139.796;
    result.minus_global_delta_ms = 113.016;
    result.plus_global_delta_ms = 167.551;
    result.base_between_abs_ms = 223.469;
    result.base_middle_abs_ms = 55.6916;
    result.minus_between_abs_ms = 50.7937;
    result.minus_middle_abs_ms = 219.773;
    result.plus_between_abs_ms = 126.213;
    result.plus_middle_abs_ms = 150.499;

    const int selected = beatit::detail::select_sparse_edge_phase_candidate(result);
    if (selected != 0) {
        std::cerr << "Sparse edge phase test failed: expected base candidate for large interior tradeoff, got "
                  << selected << ".\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!test_phase_try_rejects_clean_to_bad_interior_swap()) {
        return 1;
    }
    if (!test_phase_try_accepts_clear_minus_improvement()) {
        return 1;
    }
    if (!test_phase_try_rejects_large_interior_tradeoff()) {
        return 1;
    }

    std::cout << "Sparse edge phase test passed.\n";
    return 0;
}
