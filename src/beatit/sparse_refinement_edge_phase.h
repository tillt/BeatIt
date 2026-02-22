//
//  sparse_refinement_edge_phase.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/sparse_probe.h"

#include <limits>
#include <vector>

namespace beatit {
namespace detail {

struct SparseEdgePhaseTryResult {
    double base_score = std::numeric_limits<double>::infinity();
    double minus_score = std::numeric_limits<double>::infinity();
    double plus_score = std::numeric_limits<double>::infinity();
    int selected = 0;
    bool applied = false;
};

SparseEdgePhaseTryResult apply_sparse_edge_phase_try(
    std::vector<unsigned long long>* projected,
    double bpm_hint,
    const SparseSampleProvider& provider,
    double sample_rate,
    double probe_duration,
    double between_probe_start,
    double middle_probe_start,
    double first_window_start,
    double last_window_start);

} // namespace detail
} // namespace beatit
