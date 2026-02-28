//
//  interior_select.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/sparse/phase_metrics.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace beatit {
namespace detail {

struct SparseInteriorCandidate {
    double start_seconds = 0.0;
    SparseWindowPhaseMetrics metrics;
};

struct SparseInteriorPickResult {
    std::size_t index = std::numeric_limits<std::size_t>::max();
    double score = std::numeric_limits<double>::infinity();
};

double score_sparse_interior_candidate(const SparseWindowPhaseMetrics& metrics);

SparseInteriorPickResult pick_best_sparse_interior_candidate(
    const std::vector<SparseInteriorCandidate>& candidates);

} // namespace detail
} // namespace beatit
