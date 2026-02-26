//
//  probe_pick.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/logging.hpp"
#include "probe_pick_internal.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace beatit {
namespace detail {

namespace {

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

    const double probe_duration = std::max(20.0, original_config.dbn_window_seconds);
    const double total = std::max(0.0, total_duration_seconds);
    const double max_start = std::max(0.0, total - probe_duration);
    const auto clamp_start = [max_start](double s) { return std::clamp(s, 0.0, max_start); };
    constexpr double kSparseEdgeExclusionSeconds = 10.0;
    const double min_allowed_start = clamp_start(kSparseEdgeExclusionSeconds);
    const double max_allowed_start = clamp_start(
        std::max(0.0, total - kSparseEdgeExclusionSeconds - probe_duration));
    const double quality_shift_step = std::clamp(probe_duration * 0.25, 5.0, 20.0);
    const std::size_t max_quality_shifts = 8;
    ProbeBuildContext context{
        original_config,
        provider,
        run_probe_fn,
        sample_rate_,
        probe_duration,
        min_allowed_start,
        max_allowed_start,
        quality_shift_step,
        max_quality_shifts
    };
    double left_anchor_start = min_allowed_start;

    std::vector<ProbeResult> probes;
    probes.reserve(3);
    const bool seed_right_first = read_seed_right_first_override();

    const bool has_distinct_right = max_allowed_start > min_allowed_start + 0.5;
    if (seed_right_first && has_distinct_right) {
        ProbeResult right_probe = seek_quality_probe(context, max_allowed_start, false);
        push_unique_probe(probes, std::move(right_probe));
    }

    ProbeResult left_probe = seek_quality_probe(context, min_allowed_start, true);
    left_anchor_start = left_probe.start;
    push_unique_probe(probes, std::move(left_probe));

    if (!seed_right_first && has_distinct_right) {
        ProbeResult right_probe = seek_quality_probe(context, max_allowed_start, false);
        push_unique_probe(probes, std::move(right_probe));
    }

    if (probes.size() < 2 && (max_allowed_start - min_allowed_start) > 1.0) {
        push_unique_probe(probes,
                          run_probe_observation(context,
                                                clamp_start(total * 0.5 - probe_duration * 0.5)));
    }

    ProbeMetricsSnapshot metrics = recompute_probe_metrics(probes,
                                                           original_config,
                                                           sample_rate_,
                                                           probe_duration,
                                                           min_allowed_start,
                                                           max_allowed_start,
                                                           provider);
    if (should_add_disagreement_probe(probes, metrics)) {
        push_unique_probe(probes,
                          run_probe_observation(context,
                                                clamp_start(total * 0.5 - probe_duration * 0.5)));
        metrics = recompute_probe_metrics(probes,
                                          original_config,
                                          sample_rate_,
                                          probe_duration,
                                          min_allowed_start,
                                          max_allowed_start,
                                          provider);
    }

    const double anchor_start = left_anchor_start;
    SelectionDecisionSnapshot selection = make_selection_decision(probes, metrics);
    SelectedProbeDiagnostics diagnostics;
    bool interior_probe_added = false;

    const auto refresh_selected = [&]() {
        const double bpm_hint =
            metrics.have_consensus ? metrics.consensus_bpm : probes[selection.selected_index].bpm;
        diagnostics = evaluate_selected_probe_diagnostics(probes[selection.selected_index],
                                                          metrics.middle_metrics[selection.selected_index],
                                                          bpm_hint,
                                                          metrics.starts,
                                                          probe_duration,
                                                          sample_rate_,
                                                          provider);
    };
    refresh_selected();

    if (probes.size() == 2) {
        if (diagnostics.middle_gate_triggered || diagnostics.consistency_gate_triggered) {
            push_unique_probe(probes, run_probe_observation(context, metrics.starts.between));
            metrics = recompute_probe_metrics(probes,
                                              original_config,
                                              sample_rate_,
                                              probe_duration,
                                              min_allowed_start,
                                              max_allowed_start,
                                              provider);
            selection = make_selection_decision(probes, metrics);
            refresh_selected();
            interior_probe_added = true;
        }
    }
    refresh_selected();

    AnalysisResult result = probes[selection.selected_index].analysis;
    const double selected_mode_error = metrics.mode_errors[selection.selected_index];
    if (selection.low_confidence) {
        const double repair_bpm = metrics.have_consensus
            ? metrics.consensus_bpm
            : (probes[selection.selected_index].bpm > 0.0
                   ? probes[selection.selected_index].bpm
                   : 0.0);
        const double repair_start = select_repair_start(probes, metrics, anchor_start);
        result = run_probe_fn(repair_start, probe_duration, repair_bpm);
    }

    auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
    append_probe_debug_line(debug_stream,
                            probes,
                            metrics,
                            selection.decision,
                            selection.selected_index,
                            selection.selected_score,
                            selection.score_margin,
                            selected_mode_error,
                            selection.selected_intro_metrics,
                            diagnostics,
                            anchor_start,
                            interior_probe_added,
                            selection.low_confidence);

    out.result = std::move(result);
    out.probes = std::move(probes);
    out.probe_duration = probe_duration;
    out.between_probe_start = metrics.starts.between;
    out.middle_probe_start = metrics.starts.middle;
    out.low_confidence = selection.low_confidence;
    out.selected_intro_median_abs_ms = selection.selected_intro_metrics.median_abs_ms;
    out.have_consensus = metrics.have_consensus;
    out.consensus_bpm = metrics.consensus_bpm;
    return out;
}

} // namespace detail
} // namespace beatit
