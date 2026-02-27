//
//  probe_pick_helpers.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "probe_pick_internal.h"
#include "beatit/sparse/waveform.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace beatit {
namespace detail {

double clamp_probe_start(const ProbeBuildContext& context, double start_s) {
    return std::clamp(start_s, 0.0, context.max_allowed_start);
}

ProbeResult run_probe_observation(const ProbeBuildContext& context, double start_s) {
    ProbeResult probe;
    probe.start = clamp_probe_start(context, start_s);
    probe.analysis = context.run_probe(probe.start, context.probe_duration, 0.0);
    probe.bpm = probe.analysis.estimated_bpm;
    probe.conf = sparse_estimate_probe_confidence(probe.analysis, context.sample_rate);
    probe.phase_abs_ms = sparse_estimate_intro_phase_abs_ms(probe.analysis,
                                                            probe.bpm,
                                                            context.sample_rate,
                                                            context.provider);
    return probe;
}

ProbeResult seek_quality_probe(const ProbeBuildContext& context,
                               double seed_start,
                               bool shift_right) {
    double start_s = std::clamp(seed_start, context.min_allowed_start, context.max_allowed_start);
    ProbeResult best = run_probe_observation(context, start_s);
    double best_score = sparse_probe_quality_score(best);
    if (sparse_probe_is_usable(best)) {
        return best;
    }

    for (std::size_t round = 0; round < context.max_quality_shifts; ++round) {
        double next_start = shift_right
            ? (start_s + context.quality_shift_step)
            : (start_s - context.quality_shift_step);
        next_start = std::clamp(next_start, context.min_allowed_start, context.max_allowed_start);
        if (std::abs(next_start - start_s) < 0.5) {
            break;
        }
        start_s = next_start;
        ProbeResult candidate = run_probe_observation(context, start_s);
        const double score = sparse_probe_quality_score(candidate);
        if (score > best_score) {
            best = candidate;
            best_score = score;
        }
        if (sparse_probe_is_usable(candidate)) {
            return candidate;
        }
    }
    return best;
}

void push_unique_probe(std::vector<ProbeResult>& probes, ProbeResult&& probe) {
    const double incoming_score = sparse_probe_quality_score(probe);
    for (auto& existing : probes) {
        if (std::abs(existing.start - probe.start) < 1.0) {
            if (incoming_score > sparse_probe_quality_score(existing)) {
                existing = std::move(probe);
            }
            return;
        }
    }
    probes.push_back(std::move(probe));
}

SelectionDecisionSnapshot make_selection_decision(const std::vector<ProbeResult>& probes,
                                                  const ProbeMetricsSnapshot& metrics) {
    SelectionDecisionSnapshot snapshot;
    snapshot.decision = sparse_decide_unified(probes, metrics.mode_errors, metrics.intro_metrics);
    snapshot.selected_index = snapshot.decision.selected_index;
    snapshot.selected_score = snapshot.decision.selected_score;
    snapshot.score_margin = snapshot.decision.score_margin;
    snapshot.low_confidence = snapshot.decision.low_confidence;
    snapshot.selected_intro_metrics = metrics.intro_metrics[snapshot.selected_index];
    return snapshot;
}

bool should_add_disagreement_probe(const std::vector<ProbeResult>& probes,
                                   const ProbeMetricsSnapshot& metrics) {
    if (probes.size() != 2 || metrics.consensus_bpm <= 0.0 || metrics.mode_errors.empty()) {
        return false;
    }
    const double max_err = *std::max_element(metrics.mode_errors.begin(), metrics.mode_errors.end());
    return max_err > 0.025;
}

double select_repair_start(const std::vector<ProbeResult>& probes,
                           const ProbeMetricsSnapshot& metrics,
                           double fallback_start) {
    if (!metrics.have_consensus || probes.empty()) {
        return fallback_start;
    }

    std::size_t best_mode_index = 0;
    double best_mode_error = metrics.mode_errors[0];
    for (std::size_t i = 1; i < probes.size(); ++i) {
        if (metrics.mode_errors[i] < best_mode_error) {
            best_mode_error = metrics.mode_errors[i];
            best_mode_index = i;
        }
    }
    return probes[best_mode_index].start;
}

std::pair<double, double> probe_start_extents(const std::vector<ProbeResult>& probes,
                                              double fallback_min,
                                              double fallback_max) {
    if (probes.empty()) {
        return {fallback_min, fallback_max};
    }
    double min_start = probes.front().start;
    double max_start = probes.front().start;
    for (const auto& probe : probes) {
        min_start = std::min(min_start, probe.start);
        max_start = std::max(max_start, probe.start);
    }
    return {min_start, max_start};
}

ProbeWindowStarts compute_probe_window_starts(const std::vector<ProbeResult>& probes,
                                              double min_allowed_start,
                                              double max_allowed_start) {
    const auto clamp_start = [min_allowed_start, max_allowed_start](double start_s) {
        return std::clamp(start_s, min_allowed_start, max_allowed_start);
    };

    const auto extents = probe_start_extents(probes, min_allowed_start, max_allowed_start);
    ProbeWindowStarts starts;
    starts.left = extents.first;
    starts.right = extents.second;
    starts.middle = clamp_start(0.5 * (starts.left + starts.right));
    starts.between = clamp_start(0.5 * (starts.left + starts.middle));
    return starts;
}

ProbeMetricsSnapshot recompute_probe_metrics(const std::vector<ProbeResult>& probes,
                                             const BeatitConfig& config,
                                             double sample_rate,
                                             double probe_duration,
                                             double min_allowed_start,
                                             double max_allowed_start,
                                             const SparseSampleProvider& provider) {
    ProbeMetricsSnapshot snapshot;
    snapshot.consensus_bpm = sparse_consensus_from_probes(probes, config.min_bpm, config.max_bpm);
    snapshot.have_consensus = snapshot.consensus_bpm > 0.0;
    snapshot.mode_errors =
        sparse_probe_mode_errors(probes, snapshot.consensus_bpm, config.min_bpm, config.max_bpm);
    snapshot.starts = compute_probe_window_starts(probes, min_allowed_start, max_allowed_start);

    snapshot.intro_metrics.assign(probes.size(), IntroPhaseMetrics{});
    snapshot.middle_metrics.assign(probes.size(), SparseWindowPhaseMetrics{});
    for (std::size_t i = 0; i < probes.size(); ++i) {
        const double bpm_hint = snapshot.have_consensus ? snapshot.consensus_bpm : probes[i].bpm;
        snapshot.intro_metrics[i] =
            sparse_measure_intro_phase(probes[i].analysis, bpm_hint, sample_rate, provider);
        snapshot.middle_metrics[i] = measure_sparse_window_phase(probes[i].analysis,
                                                                 bpm_hint,
                                                                 snapshot.starts.middle,
                                                                 probe_duration,
                                                                 sample_rate,
                                                                 provider);
    }

    return snapshot;
}

bool window_has_high_mismatch(const SparseWindowPhaseMetrics& metrics, const WindowPhaseGate& gate) {
    constexpr double kHighMismatchAbsRatio = 0.75;
    constexpr double kHighMismatchSignedRatio = 0.85;
    return gate.has_data && std::isfinite(metrics.abs_limit_exceed_ratio) &&
           std::isfinite(metrics.signed_limit_exceed_ratio) &&
           metrics.abs_limit_exceed_ratio >= kHighMismatchAbsRatio &&
           metrics.signed_limit_exceed_ratio >= kHighMismatchSignedRatio;
}

bool window_has_low_mismatch(const SparseWindowPhaseMetrics& metrics, const WindowPhaseGate& gate) {
    constexpr double kLowMismatchAbsRatio = 0.20;
    constexpr double kLowMismatchSignedRatio = 0.25;
    return gate.has_data && std::isfinite(metrics.abs_limit_exceed_ratio) &&
           std::isfinite(metrics.signed_limit_exceed_ratio) &&
           metrics.abs_limit_exceed_ratio <= kLowMismatchAbsRatio &&
           metrics.signed_limit_exceed_ratio <= kLowMismatchSignedRatio;
}

SelectedProbeDiagnostics evaluate_selected_probe_diagnostics(const ProbeResult& selected_probe,
                                                             const SparseWindowPhaseMetrics& middle_metrics,
                                                             double bpm_hint,
                                                             const ProbeWindowStarts& starts,
                                                             double probe_duration,
                                                             double sample_rate,
                                                             const SparseSampleProvider& provider) {
    SelectedProbeDiagnostics diagnostics;
    diagnostics.middle = middle_metrics;
    diagnostics.middle_gate = sparse_evaluate_window_phase_gate(diagnostics.middle, bpm_hint);
    diagnostics.middle_gate_triggered = diagnostics.middle_gate.unstable_or;

    diagnostics.between = measure_sparse_window_phase(selected_probe.analysis,
                                                      bpm_hint,
                                                      starts.between,
                                                      probe_duration,
                                                      sample_rate,
                                                      provider);
    diagnostics.left = measure_sparse_window_phase(selected_probe.analysis,
                                                   bpm_hint,
                                                   starts.left,
                                                   probe_duration,
                                                   sample_rate,
                                                   provider);
    diagnostics.right = measure_sparse_window_phase(selected_probe.analysis,
                                                    bpm_hint,
                                                    starts.right,
                                                    probe_duration,
                                                    sample_rate,
                                                    provider);

    diagnostics.between_gate = sparse_evaluate_window_phase_gate(diagnostics.between, bpm_hint);
    diagnostics.left_gate = sparse_evaluate_window_phase_gate(diagnostics.left, bpm_hint);
    diagnostics.right_gate = sparse_evaluate_window_phase_gate(diagnostics.right, bpm_hint);

    diagnostics.consistency_between_high_mismatch =
        window_has_high_mismatch(diagnostics.between, diagnostics.between_gate);
    diagnostics.consistency_middle_high_mismatch =
        window_has_high_mismatch(diagnostics.middle, diagnostics.middle_gate);
    diagnostics.consistency_edges_low_mismatch =
        window_has_low_mismatch(diagnostics.left, diagnostics.left_gate) &&
        window_has_low_mismatch(diagnostics.right, diagnostics.right_gate);
    diagnostics.consistency_gate_triggered =
        diagnostics.consistency_edges_low_mismatch &&
        (diagnostics.consistency_between_high_mismatch ||
         diagnostics.consistency_middle_high_mismatch);

    return diagnostics;
}

bool read_seed_right_first_override() {
    const char* value = std::getenv("BEATIT_SPARSE_SEED_ORDER");
    if (!value || value[0] == '\0') {
        return false;
    }
    return value[0] == 'r' || value[0] == 'R';
}

} // namespace detail
} // namespace beatit
