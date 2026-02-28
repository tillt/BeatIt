//
//  probe_pick_helpers.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "probe_pick_internal.h"
#include "beatit/sparse/usability_scan.h"
#include "beatit/sparse/waveform.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace beatit {
namespace detail {
namespace {

bool probe_intro_metrics_match_bpm_hint(const ProbeResult& probe, double bpm_hint) {
    if (!(probe.intro_phase_bpm > 0.0) || !(bpm_hint > 0.0)) {
        return false;
    }
    return std::abs(probe.intro_phase_bpm - bpm_hint) <= 0.01;
}

IntroPhaseMetrics cached_intro_phase_metrics(const ProbeResult& probe) {
    IntroPhaseMetrics metrics;
    metrics.median_abs_ms = probe.phase_abs_ms;
    metrics.odd_even_gap_ms = probe.intro_odd_even_gap_ms;
    return metrics;
}

ProbeWindowStarts resolve_diagnostic_window_starts(const ProbeWindowStarts& starts,
                                                   double probe_duration,
                                                   double sample_rate,
                                                   double bpm_hint,
                                                   const SparseSampleProvider& provider) {
    ProbeWindowStarts resolved = starts;
    if (!(sample_rate > 0.0) || !(bpm_hint > 0.0) || !provider) {
        return resolved;
    }

    const double scan_start = std::max(0.0, starts.left);
    const double scan_total_seconds = std::max(0.0, (starts.right - starts.left) + probe_duration);
    if (!(scan_total_seconds > 0.0)) {
        return resolved;
    }

    const double scan_window_seconds = std::clamp(probe_duration * 0.5, 20.0, 30.0);
    const double scan_hop_seconds = std::max(10.0, 0.5 * scan_window_seconds);
    const SparseSampleProvider scan_provider =
        [&provider, scan_start](double start_seconds,
                                double duration_seconds,
                                std::vector<float>* out_samples) -> std::size_t {
            return provider(scan_start + start_seconds, duration_seconds, out_samples);
        };

    auto windows = scan_sparse_usability_windows(SparseUsabilityScanRequest{
        scan_total_seconds,
        scan_window_seconds,
        scan_hop_seconds,
        sample_rate,
        std::max(40.0, bpm_hint * 0.75),
        std::min(240.0, bpm_hint * 1.25),
        &scan_provider
    });
    if (windows.empty()) {
        return resolved;
    }
    for (auto& window : windows) {
        window.start_seconds += scan_start;
    }

    const SparseInteriorWindowTargets interior_targets =
        resolve_sparse_interior_targets(windows, starts.middle, starts.between, 0.0);
    resolved.middle = interior_targets.middle_start_seconds;
    resolved.between = interior_targets.between_start_seconds;
    return resolved;
}

} // namespace

double clamp_probe_start(const ProbeBuildContext& context, double start_s) {
    return std::clamp(start_s, 0.0, context.max_allowed_start);
}

ProbeResult run_probe_observation(const ProbeBuildContext& context, double start_s) {
    ProbeResult probe;
    probe.start = clamp_probe_start(context, start_s);
    probe.analysis = context.run_probe(probe.start, context.probe_duration, 0.0);
    probe.bpm = probe.analysis.estimated_bpm;
    probe.conf = sparse_estimate_probe_confidence(probe.analysis, context.sample_rate);
    const IntroPhaseMetrics intro_metrics =
        sparse_measure_intro_phase(probe.analysis,
                                   probe.bpm,
                                   context.sample_rate,
                                   context.provider);
    probe.phase_abs_ms = intro_metrics.median_abs_ms;
    probe.intro_odd_even_gap_ms = intro_metrics.odd_even_gap_ms;
    probe.intro_phase_bpm = probe.bpm;
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

double push_quality_probe(std::vector<ProbeResult>& probes,
                          const ProbeBuildContext& context,
                          double seed_start,
                          bool shift_right) {
    ProbeResult probe = seek_quality_probe(context, seed_start, shift_right);
    const double start = probe.start;
    push_unique_probe(probes, std::move(probe));
    return start;
}

double centered_probe_start(double total_duration_seconds,
                            double probe_duration,
                            double max_allowed_start) {
    return std::clamp(0.5 * (total_duration_seconds - probe_duration), 0.0, max_allowed_start);
}

void push_observed_probe(std::vector<ProbeResult>& probes,
                         const ProbeBuildContext& context,
                         double start_s) {
    push_unique_probe(probes, run_probe_observation(context, start_s));
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
    if (!(metrics.consensus_bpm > 0.0) || probes.empty()) {
        return fallback_start;
    }

    const auto best_mode_it =
        std::min_element(metrics.mode_errors.begin(), metrics.mode_errors.end());
    const std::size_t best_mode_index =
        static_cast<std::size_t>(std::distance(metrics.mode_errors.begin(), best_mode_it));
    return probes[best_mode_index].start;
}

ProbeWindowStarts compute_probe_window_starts(const std::vector<ProbeResult>& probes,
                                              double min_allowed_start,
                                              double max_allowed_start) {
    const auto clamp_start = [min_allowed_start, max_allowed_start](double start_s) {
        return std::clamp(start_s, min_allowed_start, max_allowed_start);
    };

    ProbeWindowStarts starts;
    starts.left = min_allowed_start;
    starts.right = max_allowed_start;
    if (!probes.empty()) {
        starts.left = probes.front().start;
        starts.right = probes.front().start;
        for (const auto& probe : probes) {
            starts.left = std::min(starts.left, probe.start);
            starts.right = std::max(starts.right, probe.start);
        }
    }
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
    snapshot.mode_errors =
        sparse_probe_mode_errors(probes, snapshot.consensus_bpm, config.min_bpm, config.max_bpm);
    snapshot.starts = compute_probe_window_starts(probes, min_allowed_start, max_allowed_start);

    snapshot.intro_metrics.assign(probes.size(), IntroPhaseMetrics{});
    snapshot.middle_metrics.assign(probes.size(), SparseWindowPhaseMetrics{});
    for (std::size_t i = 0; i < probes.size(); ++i) {
        const double bpm_hint = (snapshot.consensus_bpm > 0.0) ? snapshot.consensus_bpm : probes[i].bpm;
        snapshot.intro_metrics[i] = probe_intro_metrics_match_bpm_hint(probes[i], bpm_hint)
            ? cached_intro_phase_metrics(probes[i])
            : sparse_measure_intro_phase(probes[i].analysis, bpm_hint, sample_rate, provider);
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

bool selected_middle_gate_triggered(const SelectedProbeDiagnostics& diagnostics) {
    return diagnostics.middle_gate.unstable_or;
}

bool selected_consistency_edges_low_mismatch(const SelectedProbeDiagnostics& diagnostics) {
    return window_has_low_mismatch(diagnostics.left, diagnostics.left_gate) &&
           window_has_low_mismatch(diagnostics.right, diagnostics.right_gate);
}

bool selected_consistency_between_high_mismatch(const SelectedProbeDiagnostics& diagnostics) {
    return window_has_high_mismatch(diagnostics.between, diagnostics.between_gate);
}

bool selected_consistency_middle_high_mismatch(const SelectedProbeDiagnostics& diagnostics) {
    return window_has_high_mismatch(diagnostics.middle, diagnostics.middle_gate);
}

bool selected_consistency_gate_triggered(const SelectedProbeDiagnostics& diagnostics) {
    return selected_consistency_edges_low_mismatch(diagnostics) &&
           (selected_consistency_between_high_mismatch(diagnostics) ||
            selected_consistency_middle_high_mismatch(diagnostics));
}

SelectedProbeDiagnostics evaluate_selected_probe_diagnostics(const ProbeResult& selected_probe,
                                                             const SparseWindowPhaseMetrics& middle_metrics,
                                                             double bpm_hint,
                                                             const ProbeWindowStarts& starts,
                                                             double probe_duration,
                                                             double sample_rate,
                                                             const SparseSampleProvider& provider) {
    SelectedProbeDiagnostics diagnostics;
    const ProbeWindowStarts diagnostic_starts =
        resolve_diagnostic_window_starts(starts,
                                         probe_duration,
                                         sample_rate,
                                         bpm_hint,
                                         provider);
    diagnostics.middle_start = diagnostic_starts.middle;
    diagnostics.between_start = diagnostic_starts.between;

    diagnostics.middle = (diagnostic_starts.middle == starts.middle)
        ? middle_metrics
        : measure_sparse_window_phase(selected_probe.analysis,
                                      bpm_hint,
                                      diagnostic_starts.middle,
                                      probe_duration,
                                      sample_rate,
                                      provider);
    diagnostics.middle_gate = sparse_evaluate_window_phase_gate(diagnostics.middle, bpm_hint);

    diagnostics.between = measure_sparse_window_phase(selected_probe.analysis,
                                                      bpm_hint,
                                                      diagnostic_starts.between,
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
