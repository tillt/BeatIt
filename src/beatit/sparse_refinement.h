//
//  sparse_refinement.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#pragma once

#include "beatit/analysis.h"
#include "beatit/coreml.h"

#include <vector>

namespace beatit {
namespace detail {

struct SparseProbeObservation {
    double start = 0.0;
    AnalysisResult analysis;
    double bpm = 0.0;
    double conf = 0.0;
    double phase_abs_ms = 0.0;
};

void apply_sparse_bounded_grid_refit(AnalysisResult* result, double sample_rate);

void apply_sparse_anchor_state_refit(AnalysisResult* result,
                                     double sample_rate,
                                     double probe_duration,
                                     const std::vector<SparseProbeObservation>& probes,
                                     bool verbose);

} // namespace detail
} // namespace beatit
