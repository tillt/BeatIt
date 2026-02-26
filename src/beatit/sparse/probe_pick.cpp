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
#include <string>
#include <vector>

namespace beatit {
namespace detail {

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

    using ProbeResult = SparseProbeObservation;
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
    const bool seed_right_first = []() {
        const char* value = std::getenv("BEATIT_SPARSE_SEED_ORDER");
        if (!value || value[0] == '\0') {
            return false;
        }
        return value[0] == 'r' || value[0] == 'R';
    }();
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
    auto probe_start_extents = [&]() -> std::pair<double, double> {
        if (probes.empty()) {
            return {min_allowed_start, max_allowed_start};
        }
        double min_start = probes.front().start;
        double max_start = probes.front().start;
        for (const auto& p : probes) {
            min_start = std::min(min_start, p.start);
            max_start = std::max(max_start, p.start);
        }
        return {min_start, max_start};
    };

    double consensus_bpm = sparse_consensus_from_probes(probes,
                                                        original_config.min_bpm,
                                                        original_config.max_bpm);
    if (probes.size() >= 2 && consensus_bpm > 0.0) {
        const std::vector<double> mode_errors =
            sparse_probe_mode_errors(probes,
                                     consensus_bpm,
                                     original_config.min_bpm,
                                     original_config.max_bpm);
        const double max_err = mode_errors.empty()
            ? 0.0
            : *std::max_element(mode_errors.begin(), mode_errors.end());
        if (max_err > 0.025 && probes.size() == 2) {
            push_unique_probe(run_probe_result(clamp_start(total * 0.5 - probe_duration * 0.5)));
        }
    }

    bool have_consensus = false;
    std::vector<double> probe_mode_errors;
    std::vector<IntroPhaseMetrics> probe_intro_metrics;
    std::vector<SparseWindowPhaseMetrics> probe_middle_metrics;
    double left_probe_start = min_allowed_start;
    double right_probe_start = max_allowed_start;
    double middle_probe_start = clamp_start(total * 0.5 - probe_duration * 0.5);
    double between_probe_start = clamp_start(0.5 * (min_allowed_start + middle_probe_start));

    auto recompute_probe_scores = [&]() {
        consensus_bpm = sparse_consensus_from_probes(probes,
                                                     original_config.min_bpm,
                                                     original_config.max_bpm);
        have_consensus = consensus_bpm > 0.0;

        probe_mode_errors = sparse_probe_mode_errors(probes,
                                                     consensus_bpm,
                                                     original_config.min_bpm,
                                                     original_config.max_bpm);

        const auto probe_extents = probe_start_extents();
        left_probe_start = probe_extents.first;
        right_probe_start = probe_extents.second;
        middle_probe_start = clamp_start(0.5 * (probe_extents.first + probe_extents.second));
        between_probe_start = clamp_start(0.5 * (probe_extents.first + middle_probe_start));

        probe_intro_metrics.assign(probes.size(), IntroPhaseMetrics{});
        probe_middle_metrics.assign(probes.size(), SparseWindowPhaseMetrics{});
        for (std::size_t i = 0; i < probes.size(); ++i) {
            const double bpm_hint = have_consensus ? consensus_bpm : probes[i].bpm;
            probe_intro_metrics[i] = sparse_measure_intro_phase(probes[i].analysis,
                                                                bpm_hint,
                                                                sample_rate_,
                                                                provider);
            probe_middle_metrics[i] = measure_sparse_window_phase(probes[i].analysis,
                                                                  bpm_hint,
                                                                  middle_probe_start,
                                                                  probe_duration,
                                                                  sample_rate_,
                                                                  provider);
        }
    };

    recompute_probe_scores();

    const double anchor_start = left_anchor_start;
    DecisionOutcome decision = sparse_decide_unified(probes,
                                                     probe_mode_errors,
                                                     probe_intro_metrics);
    std::size_t selected_index = decision.selected_index;
    double selected_score = decision.selected_score;
    double score_margin = decision.score_margin;
    bool low_confidence = decision.low_confidence;
    IntroPhaseMetrics selected_intro_metrics = probe_intro_metrics[selected_index];
    SparseWindowPhaseMetrics selected_middle_metrics = probe_middle_metrics[selected_index];
    SparseWindowPhaseMetrics selected_between_metrics;
    SparseWindowPhaseMetrics selected_left_window_metrics;
    SparseWindowPhaseMetrics selected_right_window_metrics;
    WindowPhaseGate selected_middle_gate;
    WindowPhaseGate selected_between_gate;
    WindowPhaseGate selected_left_gate;
    WindowPhaseGate selected_right_gate;
    bool middle_gate_triggered = false;
    bool consistency_gate_triggered = false;
    bool consistency_edges_low_mismatch = false;
    bool consistency_between_high_mismatch = false;
    bool consistency_middle_high_mismatch = false;
    bool interior_probe_added = false;

    auto refresh_selected_window_diagnostics = [&]() {
        const double selected_bpm_hint = have_consensus ? consensus_bpm : probes[selected_index].bpm;
        selected_middle_metrics = probe_middle_metrics[selected_index];
        selected_middle_gate =
            sparse_evaluate_window_phase_gate(selected_middle_metrics, selected_bpm_hint);
        selected_between_metrics = measure_sparse_window_phase(probes[selected_index].analysis,
                                                               selected_bpm_hint,
                                                               between_probe_start,
                                                               probe_duration,
                                                               sample_rate_,
                                                               provider);
        selected_between_gate =
            sparse_evaluate_window_phase_gate(selected_between_metrics, selected_bpm_hint);
        selected_left_window_metrics = measure_sparse_window_phase(probes[selected_index].analysis,
                                                                   selected_bpm_hint,
                                                                   left_probe_start,
                                                                   probe_duration,
                                                                   sample_rate_,
                                                                   provider);
        selected_right_window_metrics = measure_sparse_window_phase(probes[selected_index].analysis,
                                                                    selected_bpm_hint,
                                                                    right_probe_start,
                                                                    probe_duration,
                                                                    sample_rate_,
                                                                    provider);
        selected_left_gate = sparse_evaluate_window_phase_gate(
            selected_left_window_metrics, selected_bpm_hint);
        selected_right_gate = sparse_evaluate_window_phase_gate(
            selected_right_window_metrics, selected_bpm_hint);
    };

    auto evaluate_consistency_gate = [&]() {
        constexpr double kHotAbsRatio = 0.75;
        constexpr double kHotSignedRatio = 0.85;
        constexpr double kCalmAbsRatio = 0.20;
        constexpr double kCalmSignedRatio = 0.25;

        const auto window_high_mismatch = [&](const SparseWindowPhaseMetrics& metrics,
                                              const WindowPhaseGate& gate) {
            return gate.has_data &&
                   std::isfinite(metrics.abs_limit_exceed_ratio) &&
                   std::isfinite(metrics.signed_limit_exceed_ratio) &&
                   metrics.abs_limit_exceed_ratio >= kHotAbsRatio &&
                   metrics.signed_limit_exceed_ratio >= kHotSignedRatio;
        };
        const auto window_low_mismatch = [&](const SparseWindowPhaseMetrics& metrics,
                                             const WindowPhaseGate& gate) {
            return gate.has_data &&
                   std::isfinite(metrics.abs_limit_exceed_ratio) &&
                   std::isfinite(metrics.signed_limit_exceed_ratio) &&
                   metrics.abs_limit_exceed_ratio <= kCalmAbsRatio &&
                   metrics.signed_limit_exceed_ratio <= kCalmSignedRatio;
        };

        consistency_between_high_mismatch =
            window_high_mismatch(selected_between_metrics, selected_between_gate);
        consistency_middle_high_mismatch =
            window_high_mismatch(selected_middle_metrics, selected_middle_gate);
        consistency_edges_low_mismatch =
            window_low_mismatch(selected_left_window_metrics, selected_left_gate) &&
            window_low_mismatch(selected_right_window_metrics, selected_right_gate);
        return consistency_edges_low_mismatch &&
               (consistency_between_high_mismatch || consistency_middle_high_mismatch);
    };

    if (probes.size() == 2) {
        refresh_selected_window_diagnostics();
        middle_gate_triggered = selected_middle_gate.unstable_or;
        consistency_gate_triggered = evaluate_consistency_gate();
        if (middle_gate_triggered || consistency_gate_triggered) {
            push_unique_probe(run_probe_result(between_probe_start));
            recompute_probe_scores();
            decision = sparse_decide_unified(probes, probe_mode_errors, probe_intro_metrics);
            selected_index = decision.selected_index;
            selected_score = decision.selected_score;
            score_margin = decision.score_margin;
            low_confidence = decision.low_confidence;
            selected_intro_metrics = probe_intro_metrics[selected_index];
            refresh_selected_window_diagnostics();
            middle_gate_triggered = selected_middle_gate.unstable_or;
            consistency_gate_triggered = evaluate_consistency_gate();
            interior_probe_added = true;
        }
    }
    refresh_selected_window_diagnostics();
    middle_gate_triggered = selected_middle_gate.unstable_or;
    consistency_gate_triggered = evaluate_consistency_gate();

    AnalysisResult result = probes[selected_index].analysis;
    const double selected_mode_error = probe_mode_errors[selected_index];
    if (low_confidence) {
        const double repair_bpm = have_consensus
            ? consensus_bpm
            : (probes[selected_index].bpm > 0.0 ? probes[selected_index].bpm : 0.0);
        double repair_start = anchor_start;
        if (have_consensus && !probes.empty()) {
            std::size_t best_mode_index = 0;
            double best_mode_error = probe_mode_errors[0];
            for (std::size_t i = 1; i < probes.size(); ++i) {
                if (probe_mode_errors[i] < best_mode_error) {
                    best_mode_error = probe_mode_errors[i];
                    best_mode_index = i;
                }
            }
            repair_start = probes[best_mode_index].start;
        }
        result = run_probe(repair_start, probe_duration, repair_bpm);
    }

    auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
    debug_stream << "Sparse probes:";
    for (std::size_t i = 0; i < probes.size(); ++i) {
        const auto& p = probes[i];
        debug_stream << " start=" << p.start
                     << " bpm=" << p.bpm
                     << " conf=" << p.conf
                     << " mode_err=" << probe_mode_errors[i];
    }
    debug_stream << " consensus=" << consensus_bpm
                 << " anchor_start=" << anchor_start
                 << " left_probe_start_s=" << left_probe_start
                 << " right_probe_start_s=" << right_probe_start
                 << " decision=" << decision.mode
                 << " selected_start=" << probes[selected_index].start
                 << " selected_score=" << selected_score
                 << " score_margin=" << score_margin
                 << " selected_mode_err=" << selected_mode_error
                 << " selected_conf=" << probes[selected_index].conf
                 << " selected_intro_abs_ms=" << selected_intro_metrics.median_abs_ms
                 << " selected_odd_even_gap_ms=" << selected_intro_metrics.odd_even_gap_ms
                 << " selected_middle_ms=" << selected_middle_metrics.median_ms
                 << " selected_middle_abs_ms=" << selected_middle_metrics.median_abs_ms
                 << " selected_middle_odd_even_gap_ms=" << selected_middle_metrics.odd_even_gap_ms
                 << " selected_middle_abs_p90_ms=" << selected_middle_metrics.abs_p90_ms
                 << " selected_middle_abs_p95_ms=" << selected_middle_metrics.abs_p95_ms
                 << " selected_middle_abs_exceed_ratio=" << selected_middle_metrics.abs_limit_exceed_ratio
                 << " selected_middle_signed_exceed_ratio="
                 << selected_middle_metrics.signed_limit_exceed_ratio
                 << " middle_gate_triggered=" << (middle_gate_triggered ? 1 : 0)
                 << " consistency_gate_triggered=" << (consistency_gate_triggered ? 1 : 0)
                 << " consistency_edges_low_mismatch="
                 << (consistency_edges_low_mismatch ? 1 : 0)
                 << " consistency_between_high_mismatch="
                 << (consistency_between_high_mismatch ? 1 : 0)
                 << " consistency_middle_high_mismatch="
                 << (consistency_middle_high_mismatch ? 1 : 0)
                 << " middle_gate_has_data=" << (selected_middle_gate.has_data ? 1 : 0)
                 << " middle_gate_signed_limit_ms=" << selected_middle_gate.signed_limit_ms
                 << " middle_gate_abs_limit_ms=" << selected_middle_gate.abs_limit_ms
                 << " middle_gate_signed_exceeds=" << (selected_middle_gate.signed_exceeds ? 1 : 0)
                 << " middle_gate_abs_exceeds=" << (selected_middle_gate.abs_exceeds ? 1 : 0)
                 << " middle_gate_and=" << (selected_middle_gate.unstable_and ? 1 : 0)
                 << " middle_gate_or=" << (selected_middle_gate.unstable_or ? 1 : 0)
                 << " selected_between_ms=" << selected_between_metrics.median_ms
                 << " selected_between_abs_ms=" << selected_between_metrics.median_abs_ms
                 << " selected_between_odd_even_gap_ms=" << selected_between_metrics.odd_even_gap_ms
                 << " selected_between_abs_p90_ms=" << selected_between_metrics.abs_p90_ms
                 << " selected_between_abs_p95_ms=" << selected_between_metrics.abs_p95_ms
                 << " selected_between_abs_exceed_ratio=" << selected_between_metrics.abs_limit_exceed_ratio
                 << " selected_between_signed_exceed_ratio="
                 << selected_between_metrics.signed_limit_exceed_ratio
                 << " between_gate_has_data=" << (selected_between_gate.has_data ? 1 : 0)
                 << " between_gate_signed_exceeds=" << (selected_between_gate.signed_exceeds ? 1 : 0)
                 << " between_gate_abs_exceeds=" << (selected_between_gate.abs_exceeds ? 1 : 0)
                 << " between_gate_and=" << (selected_between_gate.unstable_and ? 1 : 0)
                 << " between_gate_or=" << (selected_between_gate.unstable_or ? 1 : 0)
                 << " selected_left_ms=" << selected_left_window_metrics.median_ms
                 << " selected_left_abs_ms=" << selected_left_window_metrics.median_abs_ms
                 << " selected_left_odd_even_gap_ms=" << selected_left_window_metrics.odd_even_gap_ms
                 << " selected_left_abs_p90_ms=" << selected_left_window_metrics.abs_p90_ms
                 << " selected_left_abs_p95_ms=" << selected_left_window_metrics.abs_p95_ms
                 << " selected_left_abs_exceed_ratio=" << selected_left_window_metrics.abs_limit_exceed_ratio
                 << " selected_left_signed_exceed_ratio="
                 << selected_left_window_metrics.signed_limit_exceed_ratio
                 << " left_gate_has_data=" << (selected_left_gate.has_data ? 1 : 0)
                 << " left_gate_signed_exceeds=" << (selected_left_gate.signed_exceeds ? 1 : 0)
                 << " left_gate_abs_exceeds=" << (selected_left_gate.abs_exceeds ? 1 : 0)
                 << " left_gate_and=" << (selected_left_gate.unstable_and ? 1 : 0)
                 << " left_gate_or=" << (selected_left_gate.unstable_or ? 1 : 0)
                 << " selected_right_ms=" << selected_right_window_metrics.median_ms
                 << " selected_right_abs_ms=" << selected_right_window_metrics.median_abs_ms
                 << " selected_right_odd_even_gap_ms=" << selected_right_window_metrics.odd_even_gap_ms
                 << " selected_right_abs_p90_ms=" << selected_right_window_metrics.abs_p90_ms
                 << " selected_right_abs_p95_ms=" << selected_right_window_metrics.abs_p95_ms
                 << " selected_right_abs_exceed_ratio=" << selected_right_window_metrics.abs_limit_exceed_ratio
                 << " selected_right_signed_exceed_ratio="
                 << selected_right_window_metrics.signed_limit_exceed_ratio
                 << " right_gate_has_data=" << (selected_right_gate.has_data ? 1 : 0)
                 << " right_gate_signed_exceeds=" << (selected_right_gate.signed_exceeds ? 1 : 0)
                 << " right_gate_abs_exceeds=" << (selected_right_gate.abs_exceeds ? 1 : 0)
                 << " right_gate_and=" << (selected_right_gate.unstable_and ? 1 : 0)
                 << " right_gate_or=" << (selected_right_gate.unstable_or ? 1 : 0)
                 << " middle_probe_start_s=" << middle_probe_start
                 << " between_probe_start_s=" << between_probe_start
                 << " interior_probe_added=" << (interior_probe_added ? 1 : 0)
                 << " repair=" << (low_confidence ? 1 : 0);

    out.result = std::move(result);
    out.probes = std::move(probes);
    out.probe_duration = probe_duration;
    out.between_probe_start = between_probe_start;
    out.middle_probe_start = middle_probe_start;
    out.low_confidence = low_confidence;
    out.selected_intro_median_abs_ms = selected_intro_metrics.median_abs_ms;
    out.have_consensus = have_consensus;
    out.consensus_bpm = consensus_bpm;
    return out;
}

} // namespace detail
} // namespace beatit
