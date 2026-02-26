//
//  probe.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/analysis.h"
#include "beatit/coreml.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace beatit {
namespace detail {

using SparseSampleProvider =
    std::function<std::size_t(double start_seconds,
                              double duration_seconds,
                              std::vector<float>* out_samples)>;
using SparseRunProbe =
    std::function<AnalysisResult(double probe_start,
                                 double probe_duration,
                                 double forced_reference_bpm)>;
using SparseEstimateBpm =
    std::function<float(const std::vector<unsigned long long>& beat_samples,
                        double sample_rate)>;
using SparseNormalizeBpm =
    std::function<float(float bpm, float min_bpm, float max_bpm)>;

struct SparseProbeObservation {
    double start = 0.0;
    AnalysisResult analysis;
    double bpm = 0.0;
    double conf = 0.0;
    double phase_abs_ms = 0.0;
};

AnalysisResult analyze_sparse_probe_window(const BeatitConfig& original_config,
                                           double sample_rate,
                                           double total_duration_seconds,
                                           const SparseSampleProvider& provider,
                                           const SparseRunProbe& run_probe_fn,
                                           const SparseEstimateBpm& estimate_bpm_from_beats_fn,
                                           const SparseNormalizeBpm& normalize_bpm_fn);

} // namespace detail
} // namespace beatit
