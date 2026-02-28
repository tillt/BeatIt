//
//  phase_metrics.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/analysis.h"
#include "beatit/sparse/probe.h"

#include <cstddef>
#include <limits>

namespace beatit {
namespace detail {

struct SparseWindowPhaseMetrics {
    double median_ms = std::numeric_limits<double>::infinity();
    double median_abs_ms = std::numeric_limits<double>::infinity();
    double odd_even_gap_ms = std::numeric_limits<double>::infinity();
    double abs_limit_exceed_ratio = std::numeric_limits<double>::infinity();
    double signed_limit_exceed_ratio = std::numeric_limits<double>::infinity();
    std::size_t count = 0;
};

double sparse_signed_phase_limit_ms(double bpm_hint);

double sparse_abs_phase_limit_ms(double bpm_hint);

SparseWindowPhaseMetrics measure_sparse_window_phase(const AnalysisResult& result,
                                                     double bpm_hint,
                                                     double window_start_s,
                                                     double probe_duration,
                                                     double sample_rate,
                                                     const SparseSampleProvider& provider);

} // namespace detail
} // namespace beatit
