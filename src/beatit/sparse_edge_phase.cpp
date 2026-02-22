//
//  sparse_edge_phase.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "sparse_edge_phase.h"

#include "beatit/sparse_phase_metrics.h"
#include "sparse_edge_metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace beatit {
namespace detail {
namespace {

struct PhaseScoreSummary {
    bool valid = false;
    double score = std::numeric_limits<double>::infinity();
    double global_delta_ms = std::numeric_limits<double>::infinity();
    double intro_abs_ms = std::numeric_limits<double>::infinity();
    double outro_abs_ms = std::numeric_limits<double>::infinity();
    double between_abs_ms = std::numeric_limits<double>::infinity();
    double middle_abs_ms = std::numeric_limits<double>::infinity();
    double phase_consensus_penalty_ms = std::numeric_limits<double>::infinity();
    double periodicity_penalty_ms = std::numeric_limits<double>::infinity();
};

std::pair<SparseWindowPhaseMetrics, SparseWindowPhaseMetrics> measure_middle_windows(
    const std::vector<unsigned long long>& beats,
    double bpm_hint,
    double between_probe_start,
    double middle_probe_start,
    double probe_duration,
    double sample_rate,
    const SparseSampleProvider& provider) {
    SparseWindowPhaseMetrics between_metrics;
    SparseWindowPhaseMetrics middle_metrics;
    if (beats.size() < 32) {
        return {between_metrics, middle_metrics};
    }
    AnalysisResult tmp;
    tmp.coreml_beat_projected_sample_frames = beats;
    between_metrics = measure_sparse_window_phase(tmp,
                                                  bpm_hint,
                                                  between_probe_start,
                                                  probe_duration,
                                                  sample_rate,
                                                  provider);
    middle_metrics = measure_sparse_window_phase(tmp,
                                                 bpm_hint,
                                                 middle_probe_start,
                                                 probe_duration,
                                                 sample_rate,
                                                 provider);
    return {between_metrics, middle_metrics};
}

std::pair<SparseWindowPhaseMetrics, SparseWindowPhaseMetrics> measure_window_phase_pair(
    const std::vector<unsigned long long>& beats,
    double bpm_hint,
    double first_window_start,
    double last_window_start,
    double probe_duration,
    double sample_rate,
    const SparseSampleProvider& provider) {
    AnalysisResult tmp;
    tmp.coreml_beat_projected_sample_frames = beats;
    return std::pair<SparseWindowPhaseMetrics, SparseWindowPhaseMetrics>{
        measure_sparse_window_phase(tmp,
                                    bpm_hint,
                                    first_window_start,
                                    probe_duration,
                                    sample_rate,
                                    provider),
        measure_sparse_window_phase(tmp,
                                    bpm_hint,
                                    last_window_start,
                                    probe_duration,
                                    sample_rate,
                                    provider)};
}

std::vector<unsigned long long> apply_ratio_candidate(const std::vector<unsigned long long>& beats,
                                                      double ratio_value) {
    std::vector<unsigned long long> candidate = beats;
    if (candidate.size() < 2 || !(ratio_value > 0.0)) {
        return candidate;
    }
    const long long anchor = static_cast<long long>(candidate.front());
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        const long long current = static_cast<long long>(candidate[i]);
        const double rel = static_cast<double>(current - anchor);
        const long long adjusted =
            anchor + static_cast<long long>(std::llround(rel * ratio_value));
        candidate[i] =
            static_cast<unsigned long long>(std::max<long long>(0, adjusted));
    }
    return candidate;
}

PhaseScoreSummary score_phase_candidate(const std::vector<unsigned long long>& beats,
                                        double bpm_hint,
                                        const SparseSampleProvider& provider,
                                        double sample_rate,
                                        double probe_duration,
                                        double between_probe_start,
                                        double middle_probe_start,
                                        double first_window_start,
                                        double last_window_start) {
    PhaseScoreSummary out;
    if (beats.size() < 64) {
        return out;
    }
    const EdgeOffsetMetrics intro_m = measure_edge_offsets(beats, bpm_hint, false, sample_rate, provider);
    const EdgeOffsetMetrics outro_m = measure_edge_offsets(beats, bpm_hint, true, sample_rate, provider);
    const auto middle_pair = measure_middle_windows(beats,
                                                    bpm_hint,
                                                    between_probe_start,
                                                    middle_probe_start,
                                                    probe_duration,
                                                    sample_rate,
                                                    provider);
    const auto edge_phase_pair = measure_window_phase_pair(beats,
                                                           bpm_hint,
                                                           first_window_start,
                                                           last_window_start,
                                                           probe_duration,
                                                           sample_rate,
                                                           provider);
    if (intro_m.count < 8 || outro_m.count < 8 ||
        middle_pair.first.count < 8 || middle_pair.second.count < 8 ||
        edge_phase_pair.first.count < 8 || edge_phase_pair.second.count < 8) {
        return out;
    }
    if (!std::isfinite(intro_m.median_ms) || !std::isfinite(outro_m.median_ms) ||
        !std::isfinite(middle_pair.first.median_abs_ms) ||
        !std::isfinite(middle_pair.second.median_abs_ms) ||
        !std::isfinite(middle_pair.first.median_ms) ||
        !std::isfinite(middle_pair.second.median_ms) ||
        !std::isfinite(edge_phase_pair.first.median_ms) ||
        !std::isfinite(edge_phase_pair.second.median_ms)) {
        return out;
    }

    out.valid = true;
    out.intro_abs_ms = std::abs(intro_m.median_ms);
    out.outro_abs_ms = std::abs(outro_m.median_ms);
    out.global_delta_ms = std::abs(outro_m.median_ms - intro_m.median_ms);
    out.between_abs_ms = middle_pair.first.median_abs_ms;
    out.middle_abs_ms = middle_pair.second.median_abs_ms;

    const double edge_consensus_ms =
        0.5 * (edge_phase_pair.first.median_ms + edge_phase_pair.second.median_ms);
    out.phase_consensus_penalty_ms =
        std::abs(middle_pair.first.median_ms - edge_consensus_ms) +
        std::abs(middle_pair.second.median_ms - edge_consensus_ms);

    const auto mismatch_excess = [](double interior, double edge, double slack) {
        if (!std::isfinite(interior) || !std::isfinite(edge)) {
            return 0.0;
        }
        return std::max(0.0, interior - edge - slack);
    };
    const double edge_abs_ratio =
        0.5 * (edge_phase_pair.first.abs_limit_exceed_ratio +
               edge_phase_pair.second.abs_limit_exceed_ratio);
    const double edge_signed_ratio =
        0.5 * (edge_phase_pair.first.signed_limit_exceed_ratio +
               edge_phase_pair.second.signed_limit_exceed_ratio);
    const double interior_abs_ratio = std::max(middle_pair.first.abs_limit_exceed_ratio,
                                               middle_pair.second.abs_limit_exceed_ratio);
    const double interior_signed_ratio = std::max(
        middle_pair.first.signed_limit_exceed_ratio,
        middle_pair.second.signed_limit_exceed_ratio);
    const double edge_odd_even_ms =
        0.5 * (edge_phase_pair.first.odd_even_gap_ms +
               edge_phase_pair.second.odd_even_gap_ms);
    const double interior_odd_even_ms =
        std::max(middle_pair.first.odd_even_gap_ms, middle_pair.second.odd_even_gap_ms);
    const double ratio_penalty =
        (220.0 * mismatch_excess(interior_abs_ratio, edge_abs_ratio, 0.10)) +
        (180.0 * mismatch_excess(interior_signed_ratio, edge_signed_ratio, 0.10));
    const double odd_even_penalty =
        0.75 * mismatch_excess(interior_odd_even_ms, edge_odd_even_ms, 15.0);
    out.periodicity_penalty_ms = ratio_penalty + odd_even_penalty;

    out.score = (0.60 * out.global_delta_ms) +
                (0.35 * (out.intro_abs_ms + out.outro_abs_ms)) +
                out.between_abs_ms +
                out.middle_abs_ms +
                (0.55 * out.phase_consensus_penalty_ms) +
                out.periodicity_penalty_ms;
    return out;
}

} // namespace

SparseEdgePhaseTryResult apply_sparse_edge_phase_try(
    std::vector<unsigned long long>* projected,
    double bpm_hint,
    const SparseSampleProvider& provider,
    double sample_rate,
    double probe_duration,
    double between_probe_start,
    double middle_probe_start,
    double first_window_start,
    double last_window_start) {
    SparseEdgePhaseTryResult out;
    if (!projected || projected->size() < 128) {
        return out;
    }

    const auto base_score = score_phase_candidate(*projected,
                                                  bpm_hint,
                                                  provider,
                                                  sample_rate,
                                                  probe_duration,
                                                  between_probe_start,
                                                  middle_probe_start,
                                                  first_window_start,
                                                  last_window_start);
    out.base_score = base_score.score;
    if (!base_score.valid) {
        return out;
    }

    const double intervals = static_cast<double>(projected->size() - 1);
    const double minus_intervals = intervals - 1.0;
    const double plus_intervals = intervals + 1.0;
    const double minus_ratio = (minus_intervals > 0.0) ? (intervals / minus_intervals) : 1.0;
    const double plus_ratio = (plus_intervals > 0.0) ? (intervals / plus_intervals) : 1.0;

    const std::vector<unsigned long long> minus_candidate =
        apply_ratio_candidate(*projected, minus_ratio);
    const std::vector<unsigned long long> plus_candidate =
        apply_ratio_candidate(*projected, plus_ratio);
    const auto minus_score = score_phase_candidate(minus_candidate,
                                                   bpm_hint,
                                                   provider,
                                                   sample_rate,
                                                   probe_duration,
                                                   between_probe_start,
                                                   middle_probe_start,
                                                   first_window_start,
                                                   last_window_start);
    const auto plus_score = score_phase_candidate(plus_candidate,
                                                  bpm_hint,
                                                  provider,
                                                  sample_rate,
                                                  probe_duration,
                                                  between_probe_start,
                                                  middle_probe_start,
                                                  first_window_start,
                                                  last_window_start);
    out.minus_score = minus_score.score;
    out.plus_score = plus_score.score;

    double best_score = base_score.score;
    std::vector<unsigned long long> best_beats = *projected;
    int best_choice = 0;
    if (minus_score.valid && minus_score.score < (best_score - 2.0)) {
        best_score = minus_score.score;
        best_beats = minus_candidate;
        best_choice = -1;
    }
    if (plus_score.valid && plus_score.score < (best_score - 2.0)) {
        best_score = plus_score.score;
        best_beats = plus_candidate;
        best_choice = 1;
    }
    if (best_choice != 0) {
        *projected = std::move(best_beats);
        out.selected = best_choice;
        out.applied = true;
    }

    return out;
}

} // namespace detail
} // namespace beatit
