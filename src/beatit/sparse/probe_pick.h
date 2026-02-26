//
//  probe_pick.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/sparse/probe.h"

#include <limits>
#include <vector>

namespace beatit {
namespace detail {

struct SparseProbeSelectionParams {
    const BeatitConfig* config = nullptr;
    double sample_rate = 0.0;
    double total_duration_seconds = 0.0;
    const SparseSampleProvider* provider = nullptr;
    const SparseRunProbe* run_probe = nullptr;
};

struct SparseProbeSelectionResult {
    AnalysisResult result;
    std::vector<SparseProbeObservation> probes;
    double probe_duration = 0.0;
    double between_probe_start = 0.0;
    double middle_probe_start = 0.0;
    bool low_confidence = true;
    double selected_intro_median_abs_ms = std::numeric_limits<double>::infinity();
    bool have_consensus = false;
    double consensus_bpm = 0.0;
};

SparseProbeSelectionResult select_sparse_probe_result(const SparseProbeSelectionParams& params);

} // namespace detail
} // namespace beatit
