//
//  sparse_probe_score.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/sparse_phase_metrics.h"
#include "beatit/sparse_probe.h"

#include <limits>
#include <vector>

namespace beatit {
namespace detail {

struct IntroPhaseMetrics {
    double median_abs_ms = std::numeric_limits<double>::infinity();
    double odd_even_gap_ms = std::numeric_limits<double>::infinity();
    std::size_t count = 0;
};

struct WindowPhaseGate {
    bool has_data = false;
    double beat_ms = 0.0;
    double signed_limit_ms = 0.0;
    double abs_limit_ms = 0.0;
    bool signed_exceeds = false;
    bool abs_exceeds = false;
    bool unstable_and = false;
    bool unstable_or = false;
};

struct DecisionOutcome {
    std::size_t selected_index = 0;
    double selected_score = std::numeric_limits<double>::infinity();
    double score_margin = 0.0;
    bool low_confidence = true;
    const char* mode = "unified";
};

double sparse_estimate_probe_confidence(const AnalysisResult& result, double sample_rate);

double sparse_estimate_intro_phase_abs_ms(const AnalysisResult& result,
                                          double bpm_hint,
                                          double sample_rate,
                                          const SparseSampleProvider& provider);

double sparse_probe_quality_score(const SparseProbeObservation& probe);

bool sparse_probe_is_usable(const SparseProbeObservation& probe);

double sparse_consensus_from_probes(const std::vector<SparseProbeObservation>& probes,
                                    double min_bpm,
                                    double max_bpm);

std::vector<double> sparse_probe_mode_errors(const std::vector<SparseProbeObservation>& probes,
                                             double consensus_bpm,
                                             double min_bpm,
                                             double max_bpm);

IntroPhaseMetrics sparse_measure_intro_phase(const AnalysisResult& result,
                                             double bpm_hint,
                                             double sample_rate,
                                             const SparseSampleProvider& provider);

DecisionOutcome sparse_decide_unified(const std::vector<SparseProbeObservation>& probes,
                                      const std::vector<double>& probe_mode_errors,
                                      const std::vector<IntroPhaseMetrics>& probe_intro_metrics);

WindowPhaseGate sparse_evaluate_window_phase_gate(const SparseWindowPhaseMetrics& metrics,
                                                  double bpm_hint);

} // namespace detail
} // namespace beatit
