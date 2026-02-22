//
//  sparse_refinement.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#pragma once

#include "beatit/analysis.h"
#include "beatit/coreml.h"
#include "beatit/sparse_probe.h"

#include <vector>

namespace beatit {
namespace detail {

void apply_sparse_bounded_grid_refit(AnalysisResult* result, double sample_rate);

void apply_sparse_anchor_state_refit(AnalysisResult* result,
                                     double sample_rate,
                                     double probe_duration,
                                     const std::vector<SparseProbeObservation>& probes,
                                     bool verbose);

struct SparseWaveformRefitParams {
    const CoreMLConfig* config = nullptr;
    const SparseSampleProvider* provider = nullptr;
    const SparseEstimateBpm* estimate_bpm_from_beats = nullptr;
    const std::vector<SparseProbeObservation>* probes = nullptr;
    double sample_rate = 0.0;
    double probe_duration = 0.0;
    double between_probe_start = 0.0;
    double middle_probe_start = 0.0;
};

void apply_sparse_waveform_edge_refit(AnalysisResult* result,
                                      const SparseWaveformRefitParams& params);

} // namespace detail
} // namespace beatit
