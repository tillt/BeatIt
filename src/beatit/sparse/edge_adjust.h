//
//  edge_adjust.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "edge_metrics.h"

#include <vector>

namespace beatit {
namespace detail {

struct SparseOpeningAnchorGuardInput {
    double base_intro_abs_ms = 0.0;
    double base_outro_abs_ms = 0.0;
    double candidate_intro_abs_ms = 0.0;
    double candidate_outro_abs_ms = 0.0;
};

double apply_sparse_edge_scale(std::vector<unsigned long long>& beats,
                               double ratio,
                               double min_ratio,
                               double max_ratio,
                               double min_delta);

double apply_sparse_edge_scale_from_back(std::vector<unsigned long long>& beats,
                                         double ratio,
                                         double min_ratio,
                                         double max_ratio,
                                         double min_delta);

double compute_sparse_edge_ratio(const std::vector<unsigned long long>& beats,
                                 const EdgeOffsetMetrics& first,
                                 const EdgeOffsetMetrics& last,
                                 double sample_rate);

bool apply_sparse_uniform_shift(std::vector<unsigned long long>& beats, long long shift_frames);

bool should_accept_sparse_opening_anchor_guard(const SparseOpeningAnchorGuardInput& input);

} // namespace detail
} // namespace beatit
