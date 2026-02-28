//
//  sparse_edge_adjust_test.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "sparse/edge_adjust.h"

#include <iostream>
#include <vector>

namespace {

bool test_apply_sparse_edge_scale_from_back_preserves_last_beat() {
    std::vector<unsigned long long> beats{100, 200, 300, 400};
    const double applied =
        beatit::detail::apply_sparse_edge_scale_from_back(beats, 0.5, 0.1, 2.0, 1e-6);
    if (applied != 0.5) {
        std::cerr << "Sparse edge adjust test failed: expected applied ratio 0.5.\n";
        return false;
    }
    if (beats.back() != 400ULL) {
        std::cerr << "Sparse edge adjust test failed: last beat was not preserved.\n";
        return false;
    }
    if (beats.front() != 250ULL || beats[1] != 300ULL || beats[2] != 350ULL) {
        std::cerr << "Sparse edge adjust test failed: unexpected back-anchored scaling result.\n";
        return false;
    }
    return true;
}

bool test_opening_anchor_guard_accepts_clear_opening_improvement() {
    const beatit::detail::SparseOpeningAnchorGuardInput input{
        73.0,
        30.0,
        48.0,
        34.0
    };
    if (!beatit::detail::should_accept_sparse_opening_anchor_guard(input)) {
        std::cerr << "Sparse edge adjust test failed: expected opening guard acceptance.\n";
        return false;
    }
    return true;
}

bool test_opening_anchor_guard_rejects_late_regression() {
    const beatit::detail::SparseOpeningAnchorGuardInput input{
        73.0,
        30.0,
        45.0,
        58.0
    };
    if (beatit::detail::should_accept_sparse_opening_anchor_guard(input)) {
        std::cerr << "Sparse edge adjust test failed: expected opening guard rejection.\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!test_apply_sparse_edge_scale_from_back_preserves_last_beat()) {
        return 1;
    }
    if (!test_opening_anchor_guard_accepts_clear_opening_improvement()) {
        return 1;
    }
    if (!test_opening_anchor_guard_rejects_late_regression()) {
        return 1;
    }

    std::cout << "Sparse edge adjust test passed.\n";
    return 0;
}
