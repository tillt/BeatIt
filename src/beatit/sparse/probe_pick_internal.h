//
//  probe_pick_internal.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/sparse/probe_pick.h"
#include "probe_score.h"

#include <utility>
#include <vector>

namespace beatit {
namespace detail {

using ProbeResult = SparseProbeObservation;

struct ProbeWindowStarts {
    double left = 0.0;
    double right = 0.0;
    double middle = 0.0;
    double between = 0.0;
};

struct ProbeMetricsSnapshot {
    double consensus_bpm = 0.0;
    std::vector<double> mode_errors;
    std::vector<IntroPhaseMetrics> intro_metrics;
    std::vector<SparseWindowPhaseMetrics> middle_metrics;
    ProbeWindowStarts starts;
};

struct SelectedProbeDiagnostics {
    SparseWindowPhaseMetrics middle;
    SparseWindowPhaseMetrics between;
    SparseWindowPhaseMetrics left;
    SparseWindowPhaseMetrics right;
    WindowPhaseGate middle_gate;
    WindowPhaseGate between_gate;
    WindowPhaseGate left_gate;
    WindowPhaseGate right_gate;
    bool middle_gate_triggered = false;
    bool consistency_gate_triggered = false;
    bool consistency_edges_low_mismatch = false;
    bool consistency_between_high_mismatch = false;
    bool consistency_middle_high_mismatch = false;
};

struct ProbeBuildContext {
    const BeatitConfig& config;
    const SparseSampleProvider& provider;
    const SparseRunProbe& run_probe;
    double sample_rate = 0.0;
    double probe_duration = 0.0;
    double min_allowed_start = 0.0;
    double max_allowed_start = 0.0;
    double quality_shift_step = 0.0;
    std::size_t max_quality_shifts = 0;
};

double clamp_probe_start(const ProbeBuildContext& context, double start_s);

ProbeResult run_probe_observation(const ProbeBuildContext& context, double start_s);

ProbeResult seek_quality_probe(const ProbeBuildContext& context,
                               double seed_start,
                               bool shift_right);

void push_unique_probe(std::vector<ProbeResult>& probes, ProbeResult&& probe);

DecisionOutcome make_selection_decision(const std::vector<ProbeResult>& probes,
                                        const ProbeMetricsSnapshot& metrics);

bool should_add_disagreement_probe(const std::vector<ProbeResult>& probes,
                                   const ProbeMetricsSnapshot& metrics);

double select_repair_start(const std::vector<ProbeResult>& probes,
                           const ProbeMetricsSnapshot& metrics,
                           double fallback_start);

std::pair<double, double> probe_start_extents(const std::vector<ProbeResult>& probes,
                                              double fallback_min,
                                              double fallback_max);

ProbeWindowStarts compute_probe_window_starts(const std::vector<ProbeResult>& probes,
                                              double min_allowed_start,
                                              double max_allowed_start);

ProbeMetricsSnapshot recompute_probe_metrics(const std::vector<ProbeResult>& probes,
                                             const BeatitConfig& config,
                                             double sample_rate,
                                             double probe_duration,
                                             double min_allowed_start,
                                             double max_allowed_start,
                                             const SparseSampleProvider& provider);

bool window_has_high_mismatch(const SparseWindowPhaseMetrics& metrics, const WindowPhaseGate& gate);

bool window_has_low_mismatch(const SparseWindowPhaseMetrics& metrics, const WindowPhaseGate& gate);

SelectedProbeDiagnostics evaluate_selected_probe_diagnostics(const ProbeResult& selected_probe,
                                                             const SparseWindowPhaseMetrics& middle_metrics,
                                                             double bpm_hint,
                                                             const ProbeWindowStarts& starts,
                                                             double probe_duration,
                                                             double sample_rate,
                                                             const SparseSampleProvider& provider);

bool read_seed_right_first_override();

} // namespace detail
} // namespace beatit
