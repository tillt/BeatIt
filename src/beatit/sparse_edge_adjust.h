//
//  sparse_edge_adjust.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "sparse_edge_metrics.h"

#include <vector>

namespace beatit {
namespace detail {

double apply_sparse_edge_scale(std::vector<unsigned long long>* beats,
                               double ratio,
                               double min_ratio,
                               double max_ratio,
                               double min_delta);

double compute_sparse_edge_ratio(const std::vector<unsigned long long>& beats,
                                 const EdgeOffsetMetrics& first,
                                 const EdgeOffsetMetrics& last,
                                 double sample_rate);

} // namespace detail
} // namespace beatit
