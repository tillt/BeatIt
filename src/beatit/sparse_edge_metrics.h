//
//  sparse_edge_metrics.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/sparse_probe.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace beatit {
namespace detail {

struct EdgeOffsetMetrics {
    double median_ms = std::numeric_limits<double>::infinity();
    double mad_ms = std::numeric_limits<double>::infinity();
    std::size_t count = 0;
};

EdgeOffsetMetrics measure_edge_offsets(const std::vector<unsigned long long>& beats,
                                       double bpm_hint,
                                       bool from_end,
                                       double sample_rate,
                                       const SparseSampleProvider& provider);

} // namespace detail
} // namespace beatit
