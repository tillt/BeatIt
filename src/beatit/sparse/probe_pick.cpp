//
//  probe_pick.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/sparse/probe_pick.h"
#include "beatit/logging.hpp"
#include "beatit/sparse/phase_metrics.h"
#include "beatit/sparse/waveform.h"
#include "probe_score.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

namespace beatit {
namespace detail {

namespace {

using ProbeResult = SparseProbeObservation;

struct ProbeWindowStarts {
    double left = 0.0;
    double right = 0.0;
    double middle = 0.0;
    double between = 0.0;
};

struct ProbeMetricsSnapshot {
    bool have_consensus = false;
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

template <typename TLogStream>
void append_probe_debug_line(TLogStream& debug_stream,
                             const std::vector<ProbeResult>& probes,
                             const ProbeMetricsSnapshot& snapshot,
                             const DecisionOutcome& decision,
                             std::size_t selected_index,
                             double selected_score,
                             double score_margin,
                             double selected_mode_error,
                             const IntroPhaseMetrics& selected_intro_metrics,
                             const SelectedProbeDiagnostics& diagnostics,
                             double anchor_start,
                             bool interior_probe_added,
                             bool low_confidence) {
    debug_stream << "Sparse probes:";
    for (std::size_t i = 0; i < probes.size(); ++i) {
        debug_stream << " start=" << probes[i].start
                     << " bpm=" << probes[i].bpm
                     << " conf=" << probes[i].conf
                     << " mode_err=" << snapshot.mode_errors[i];
    }

    debug_stream << " consensus=" << snapshot.consensus_bpm
                 << " anchor_start=" << anchor_start
                 << " left_probe_start_s=" << snapshot.starts.left
                 << " right_probe_start_s=" << snapshot.starts.right
                 << " decision=" << decision.mode
                 << " selected_start=" << probes[selected_index].start
                 << " selected_score=" << selected_score
                 << " score_margin=" << score_margin
                 << " selected_mode_err=" << selected_mode_error
                 << " selected_conf=" << probes[selected_index].conf
                 << " selected_intro_abs_ms=" << selected_intro_metrics.median_abs_ms
                 << " selected_middle_abs_ms=" << diagnostics.middle.median_abs_ms
                 << " selected_middle_abs_exceed_ratio="
                 << diagnostics.middle.abs_limit_exceed_ratio
                 << " selected_middle_signed_exceed_ratio="
                 << diagnostics.middle.signed_limit_exceed_ratio
                 << " middle_gate_triggered=" << (diagnostics.middle_gate_triggered ? 1 : 0)
                 << " consistency_gate_triggered="
                 << (diagnostics.consistency_gate_triggered ? 1 : 0)
                 << " consistency_edges_low_mismatch="
                 << (diagnostics.consistency_edges_low_mismatch ? 1 : 0)
                 << " consistency_between_high_mismatch="
                 << (diagnostics.consistency_between_high_mismatch ? 1 : 0)
                 << " consistency_middle_high_mismatch="
                 << (diagnostics.consistency_middle_high_mismatch ? 1 : 0)
                 << " selected_between_abs_ms=" << diagnostics.between.median_abs_ms
                 << " selected_between_abs_exceed_ratio="
                 << diagnostics.between.abs_limit_exceed_ratio
                 << " selected_between_signed_exceed_ratio="
                 << diagnostics.between.signed_limit_exceed_ratio
                 << " selected_left_abs_ms=" << diagnostics.left.median_abs_ms
                 << " selected_left_abs_exceed_ratio="
                 << diagnostics.left.abs_limit_exceed_ratio
                 << " selected_left_signed_exceed_ratio="
                 << diagnostics.left.signed_limit_exceed_ratio
                 << " selected_right_abs_ms=" << diagnostics.right.median_abs_ms
                 << " selected_right_abs_exceed_ratio="
                 << diagnostics.right.abs_limit_exceed_ratio
                 << " selected_right_signed_exceed_ratio="
                 << diagnostics.right.signed_limit_exceed_ratio
                 << " middle_probe_start_s=" << snapshot.starts.middle
                 << " between_probe_start_s=" << snapshot.starts.between
                 << " interior_probe_added=" << (interior_probe_added ? 1 : 0)
                 << " repair=" << (low_confidence ? 1 : 0);
}

} // namespace

SparseProbeSelectionResult select_sparse_probe_result(const SparseProbeSelectionParams& params) {
    SparseProbeSelectionResult out;
    if (!params.config || !params.provider || !params.run_probe) {
        return out;
    }
    const BeatitConfig& original_config = *params.config;
    const SparseSampleProvider& provider = *params.provider;
    const SparseRunProbe& run_probe_fn = *params.run_probe;
    const double sample_rate_ = params.sample_rate;
    const double total_duration_seconds = params.total_duration_seconds;
    if (!(sample_rate_ > 0.0) || total_duration_seconds <= 0.0) {
        return out;
    }

    auto run_probe = [&](double probe_start,
                         double probe_duration,
                         double forced_reference_bpm = 0.0) -> AnalysisResult {
        return run_probe_fn(probe_start, probe_duration, forced_reference_bpm);
    };

    const double probe_duration = std::max(20.0, original_config.dbn_window_seconds);
    const double total = std::max(0.0, total_duration_seconds);
    const double max_start = std::max(0.0, total - probe_duration);
    const auto clamp_start = [&](double s) {
        return std::min(std::max(0.0, s), max_start);
    };
    constexpr double kSparseEdgeExclusionSeconds = 10.0;
    const double min_allowed_start = clamp_start(kSparseEdgeExclusionSeconds);
    const double max_allowed_start = clamp_start(
        std::max(0.0, total - kSparseEdgeExclusionSeconds - probe_duration));
    const double quality_shift_step = std::clamp(probe_duration * 0.25, 5.0, 20.0);
    const std::size_t max_quality_shifts = 8;
    double left_anchor_start = min_allowed_start;

    auto run_probe_result = [&](double start_s) {
        ProbeResult p;
        p.start = clamp_start(start_s);
        p.analysis = run_probe(p.start, probe_duration);
        p.bpm = p.analysis.estimated_bpm;
        p.conf = sparse_estimate_probe_confidence(p.analysis, sample_rate_);
        p.phase_abs_ms = sparse_estimate_intro_phase_abs_ms(p.analysis,
                                                            p.bpm,
                                                            sample_rate_,
                                                            provider);
        return p;
    };
    auto seek_quality_probe = [&](double seed_start, bool shift_right) {
        double start_s = std::clamp(seed_start, min_allowed_start, max_allowed_start);
        ProbeResult best = run_probe_result(start_s);
        double best_score = sparse_probe_quality_score(best);
        if (sparse_probe_is_usable(best)) {
            return best;
        }
        for (std::size_t round = 0; round < max_quality_shifts; ++round) {
            double next_start = shift_right
                ? (start_s + quality_shift_step)
                : (start_s - quality_shift_step);
            next_start = std::clamp(next_start, min_allowed_start, max_allowed_start);
            if (std::abs(next_start - start_s) < 0.5) {
                break;
            }
            start_s = next_start;
            ProbeResult candidate = run_probe_result(start_s);
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
    };

    std::vector<ProbeResult> probes;
    probes.reserve(3);
    const bool seed_right_first = read_seed_right_first_override();
    auto push_unique_probe = [&](ProbeResult&& probe) {
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
    };

    const bool has_distinct_right = max_allowed_start > min_allowed_start + 0.5;
    if (seed_right_first && has_distinct_right) {
        ProbeResult right_probe = seek_quality_probe(max_allowed_start, false);
        push_unique_probe(std::move(right_probe));
    }

    ProbeResult left_probe = seek_quality_probe(min_allowed_start, true);
    left_anchor_start = left_probe.start;
    push_unique_probe(std::move(left_probe));

    if (!seed_right_first && has_distinct_right) {
        ProbeResult right_probe = seek_quality_probe(max_allowed_start, false);
        push_unique_probe(std::move(right_probe));
    }

    if (probes.size() < 2 && (max_allowed_start - min_allowed_start) > 1.0) {
        push_unique_probe(run_probe_result(clamp_start(total * 0.5 - probe_duration * 0.5)));
    }
    ProbeMetricsSnapshot metrics = recompute_probe_metrics(probes,
                                                           original_config,
                                                           sample_rate_,
                                                           probe_duration,
                                                           min_allowed_start,
                                                           max_allowed_start,
                                                           provider);
    if (probes.size() >= 2 && metrics.consensus_bpm > 0.0) {
        const double max_err = metrics.mode_errors.empty()
            ? 0.0
            : *std::max_element(metrics.mode_errors.begin(), metrics.mode_errors.end());
        if (max_err > 0.025 && probes.size() == 2) {
            push_unique_probe(run_probe_result(clamp_start(total * 0.5 - probe_duration * 0.5)));
            metrics = recompute_probe_metrics(probes,
                                              original_config,
                                              sample_rate_,
                                              probe_duration,
                                              min_allowed_start,
                                              max_allowed_start,
                                              provider);
        }
    }

    const double anchor_start = left_anchor_start;
    DecisionOutcome decision =
        sparse_decide_unified(probes, metrics.mode_errors, metrics.intro_metrics);
    std::size_t selected_index = decision.selected_index;
    double selected_score = decision.selected_score;
    double score_margin = decision.score_margin;
    bool low_confidence = decision.low_confidence;
    IntroPhaseMetrics selected_intro_metrics = metrics.intro_metrics[selected_index];
    SelectedProbeDiagnostics diagnostics;
    bool interior_probe_added = false;

    const auto refresh_selected = [&]() {
        const double bpm_hint =
            metrics.have_consensus ? metrics.consensus_bpm : probes[selected_index].bpm;
        diagnostics = evaluate_selected_probe_diagnostics(probes[selected_index],
                                                          metrics.middle_metrics[selected_index],
                                                          bpm_hint,
                                                          metrics.starts,
                                                          probe_duration,
                                                          sample_rate_,
                                                          provider);
    };
    refresh_selected();

    if (probes.size() == 2) {
        if (diagnostics.middle_gate_triggered || diagnostics.consistency_gate_triggered) {
            push_unique_probe(run_probe_result(metrics.starts.between));
            metrics = recompute_probe_metrics(probes,
                                              original_config,
                                              sample_rate_,
                                              probe_duration,
                                              min_allowed_start,
                                              max_allowed_start,
                                              provider);
            decision = sparse_decide_unified(probes, metrics.mode_errors, metrics.intro_metrics);
            selected_index = decision.selected_index;
            selected_score = decision.selected_score;
            score_margin = decision.score_margin;
            low_confidence = decision.low_confidence;
            selected_intro_metrics = metrics.intro_metrics[selected_index];
            refresh_selected();
            interior_probe_added = true;
        }
    }
    refresh_selected();

    AnalysisResult result = probes[selected_index].analysis;
    const double selected_mode_error = metrics.mode_errors[selected_index];
    if (low_confidence) {
        const double repair_bpm = metrics.have_consensus
            ? metrics.consensus_bpm
            : (probes[selected_index].bpm > 0.0 ? probes[selected_index].bpm : 0.0);
        double repair_start = anchor_start;
        if (metrics.have_consensus && !probes.empty()) {
            std::size_t best_mode_index = 0;
            double best_mode_error = metrics.mode_errors[0];
            for (std::size_t i = 1; i < probes.size(); ++i) {
                if (metrics.mode_errors[i] < best_mode_error) {
                    best_mode_error = metrics.mode_errors[i];
                    best_mode_index = i;
                }
            }
            repair_start = probes[best_mode_index].start;
        }
        result = run_probe(repair_start, probe_duration, repair_bpm);
    }

    auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
    append_probe_debug_line(debug_stream,
                            probes,
                            metrics,
                            decision,
                            selected_index,
                            selected_score,
                            score_margin,
                            selected_mode_error,
                            selected_intro_metrics,
                            diagnostics,
                            anchor_start,
                            interior_probe_added,
                            low_confidence);

    out.result = std::move(result);
    out.probes = std::move(probes);
    out.probe_duration = probe_duration;
    out.between_probe_start = metrics.starts.between;
    out.middle_probe_start = metrics.starts.middle;
    out.low_confidence = low_confidence;
    out.selected_intro_median_abs_ms = selected_intro_metrics.median_abs_ms;
    out.have_consensus = metrics.have_consensus;
    out.consensus_bpm = metrics.consensus_bpm;
    return out;
}

} // namespace detail
} // namespace beatit
