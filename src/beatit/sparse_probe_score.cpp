//
//  sparse_probe_score.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "sparse_probe_score.h"

#include "beatit/sparse_waveform.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace beatit {
namespace detail {
namespace {

struct ProbeModeCandidate {
    double bpm = 0.0;
    double conf = 0.0;
    double mode = 1.0;
    double penalty = 1.0;
};

double mode_penalty(double mode) {
    const double kTol = 1e-6;
    if (std::fabs(mode - 1.0) < kTol) {
        return 1.0;
    }
    if (std::fabs(mode - 0.5) < kTol || std::fabs(mode - 2.0) < kTol) {
        return 1.15;
    }
    if (std::fabs(mode - 1.5) < kTol || std::fabs(mode - (2.0 / 3.0)) < kTol) {
        return 1.9;
    }
    if (std::fabs(mode - 3.0) < kTol) {
        return 2.4;
    }
    return 2.0;
}

std::vector<ProbeModeCandidate> expand_modes(double bpm,
                                             double conf,
                                             double min_bpm,
                                             double max_bpm) {
    std::vector<ProbeModeCandidate> modes;
    if (!(bpm > 0.0) || !(conf > 0.0)) {
        return modes;
    }
    static const double kModes[] = {0.5, 1.0, 2.0};
    for (double m : kModes) {
        const double cand = bpm * m;
        if (cand >= min_bpm && cand <= max_bpm) {
            modes.push_back({cand, conf, m, mode_penalty(m)});
        }
    }
    return modes;
}

double relative_diff(double a, double b) {
    const double mean = 0.5 * (a + b);
    return mean > 0.0 ? (std::abs(a - b) / mean) : 1.0;
}

std::size_t probe_beat_count(const AnalysisResult& r) {
    return sparse_select_beats(r).size();
}

} // namespace

double sparse_estimate_probe_confidence(const AnalysisResult& result, double sample_rate) {
    const auto& beats = !result.coreml_beat_projected_sample_frames.empty()
        ? result.coreml_beat_projected_sample_frames
        : result.coreml_beat_sample_frames;
    if (beats.size() < 8 || sample_rate <= 0.0) {
        return 0.0;
    }
    std::vector<double> intervals;
    intervals.reserve(beats.size() - 1);
    for (std::size_t i = 1; i < beats.size(); ++i) {
        if (beats[i] > beats[i - 1]) {
            intervals.push_back(static_cast<double>(beats[i] - beats[i - 1]) / sample_rate);
        }
    }
    if (intervals.size() < 4) {
        return 0.0;
    }
    double sum = 0.0;
    double sum_sq = 0.0;
    for (double v : intervals) {
        sum += v;
        sum_sq += v * v;
    }
    const double n = static_cast<double>(intervals.size());
    const double mean = sum / n;
    if (!(mean > 0.0)) {
        return 0.0;
    }
    const double var = std::max(0.0, (sum_sq / n) - (mean * mean));
    const double cv = std::sqrt(var) / mean;
    return (1.0 / (1.0 + cv)) * std::min(1.0, n / 32.0);
}

double sparse_estimate_intro_phase_abs_ms(const AnalysisResult& result,
                                          double bpm_hint,
                                          double sample_rate,
                                          const SparseSampleProvider& provider) {
    if (sample_rate <= 0.0 || bpm_hint <= 0.0 || !provider) {
        return std::numeric_limits<double>::infinity();
    }
    const auto& beats = sparse_select_beats(result);
    if (beats.size() < 12) {
        return std::numeric_limits<double>::infinity();
    }

    const std::size_t probe_beats = std::min<std::size_t>(24, beats.size());
    const double beat_period_s = 60.0 / bpm_hint;
    const double intro_s = std::max(20.0, beat_period_s * static_cast<double>(probe_beats + 4));

    std::vector<float> intro_samples;
    if (!sparse_load_samples(provider, 0.0, intro_s, &intro_samples)) {
        return std::numeric_limits<double>::infinity();
    }
    const std::size_t radius = sparse_waveform_radius(sample_rate, bpm_hint);
    if (radius == 0) {
        return std::numeric_limits<double>::infinity();
    }

    std::vector<double> signed_offsets_ms;
    std::vector<double> abs_offsets_ms;
    abs_offsets_ms.reserve(probe_beats);
    sparse_collect_offsets(beats,
                           0,
                           probe_beats,
                           0,
                           intro_samples,
                           radius,
                           SparsePeakMode::AbsoluteMax,
                           sample_rate,
                           &signed_offsets_ms,
                           &abs_offsets_ms);
    if (abs_offsets_ms.size() < 8) {
        return std::numeric_limits<double>::infinity();
    }
    return sparse_median_inplace(&abs_offsets_ms);
}

double sparse_probe_quality_score(const SparseProbeObservation& probe) {
    const std::size_t beat_count = probe_beat_count(probe.analysis);
    if (!(probe.bpm > 0.0) || beat_count < 4) {
        return 0.0;
    }
    const double beat_factor =
        std::min(1.0, static_cast<double>(beat_count) / 24.0);
    const double phase_factor = std::isfinite(probe.phase_abs_ms)
        ? (1.0 / (1.0 + (probe.phase_abs_ms / 120.0)))
        : 0.15;
    return probe.conf * beat_factor * phase_factor;
}

bool sparse_probe_is_usable(const SparseProbeObservation& probe) {
    const std::size_t beat_count = probe_beat_count(probe.analysis);
    return (probe.bpm > 0.0) &&
           (beat_count >= 16) &&
           (probe.conf >= 0.55) &&
           (!std::isfinite(probe.phase_abs_ms) || probe.phase_abs_ms <= 120.0);
}

double sparse_consensus_from_probes(const std::vector<SparseProbeObservation>& probes,
                                    double min_bpm,
                                    double max_bpm) {
    std::vector<ProbeModeCandidate> all_modes;
    for (const auto& p : probes) {
        const auto modes = expand_modes(p.bpm, p.conf, min_bpm, max_bpm);
        for (const auto& m : modes) {
            all_modes.push_back(m);
        }
    }
    if (all_modes.empty()) {
        return 0.0;
    }
    double best_bpm = 0.0;
    double best_score = std::numeric_limits<double>::infinity();
    for (const auto& cand : all_modes) {
        double score = 0.0;
        double support = 0.0;
        for (const auto& p : probes) {
            const auto modes = expand_modes(p.bpm, p.conf, min_bpm, max_bpm);
            double best_local = std::numeric_limits<double>::infinity();
            for (const auto& m : modes) {
                best_local = std::min(best_local, relative_diff(cand.bpm, m.bpm) * m.penalty);
            }
            score += best_local / std::max(1e-6, p.conf);
            if (best_local <= 0.02) {
                support += p.conf;
            }
        }
        score *= cand.penalty;
        score /= (1.0 + (0.8 * support));
        if (score < best_score) {
            best_score = score;
            best_bpm = cand.bpm;
        }
    }
    return best_bpm;
}

std::vector<double> sparse_probe_mode_errors(const std::vector<SparseProbeObservation>& probes,
                                             double consensus_bpm,
                                             double min_bpm,
                                             double max_bpm) {
    std::vector<double> errors(probes.size(), 1.0);
    const bool have_consensus = consensus_bpm > 0.0;
    for (std::size_t i = 0; i < probes.size(); ++i) {
        if (!have_consensus) {
            errors[i] = 0.0;
            continue;
        }
        const auto modes = expand_modes(probes[i].bpm, probes[i].conf, min_bpm, max_bpm);
        double mode_error = 1.0;
        for (const auto& m : modes) {
            mode_error = std::min(mode_error, relative_diff(consensus_bpm, m.bpm) * m.penalty);
        }
        errors[i] = mode_error;
    }
    return errors;
}

IntroPhaseMetrics sparse_measure_intro_phase(const AnalysisResult& result,
                                             double bpm_hint,
                                             double sample_rate,
                                             const SparseSampleProvider& provider) {
    IntroPhaseMetrics metrics;
    if (sample_rate <= 0.0 || bpm_hint <= 0.0 || !provider) {
        return metrics;
    }
    const auto& beats = sparse_select_beats(result);
    if (beats.size() < 12) {
        return metrics;
    }

    const std::size_t probe_beats = std::min<std::size_t>(24, beats.size());
    const double beat_period_s = 60.0 / bpm_hint;
    const double intro_s = std::max(20.0, beat_period_s * static_cast<double>(probe_beats + 4));

    std::vector<float> intro_samples;
    if (!sparse_load_samples(provider, 0.0, intro_s, &intro_samples)) {
        return metrics;
    }
    const std::size_t radius = sparse_waveform_radius(sample_rate, bpm_hint);
    if (radius == 0) {
        return metrics;
    }

    std::vector<double> signed_offsets_ms;
    std::vector<double> abs_offsets_ms;
    signed_offsets_ms.reserve(probe_beats);
    abs_offsets_ms.reserve(probe_beats);
    sparse_collect_offsets(beats,
                           0,
                           probe_beats,
                           0,
                           intro_samples,
                           radius,
                           SparsePeakMode::ThresholdedLocalMax,
                           sample_rate,
                           &signed_offsets_ms,
                           &abs_offsets_ms);

    if (abs_offsets_ms.size() < 8) {
        return metrics;
    }
    metrics.count = abs_offsets_ms.size();
    metrics.median_abs_ms = sparse_median_inplace(&abs_offsets_ms);

    std::vector<double> odd;
    std::vector<double> even;
    odd.reserve(signed_offsets_ms.size() / 2);
    even.reserve((signed_offsets_ms.size() + 1) / 2);
    for (std::size_t i = 0; i < signed_offsets_ms.size(); ++i) {
        if ((i % 2) == 0) {
            even.push_back(signed_offsets_ms[i]);
        } else {
            odd.push_back(signed_offsets_ms[i]);
        }
    }
    if (!odd.empty() && !even.empty()) {
        metrics.odd_even_gap_ms = std::fabs(
            sparse_median_inplace(&even) - sparse_median_inplace(&odd));
    } else {
        metrics.odd_even_gap_ms = 0.0;
    }
    return metrics;
}

DecisionOutcome sparse_decide_unified(const std::vector<SparseProbeObservation>& probes,
                                      const std::vector<double>& probe_mode_errors,
                                      const std::vector<IntroPhaseMetrics>& probe_intro_metrics) {
    DecisionOutcome decision;
    double second_best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < probes.size(); ++i) {
        const auto& intro = probe_intro_metrics[i];
        const double tempo_term = probe_mode_errors[i];
        const double phase_term =
            std::isfinite(intro.median_abs_ms) ? (intro.median_abs_ms / 120.0) : 2.0;
        const double stability_term =
            std::isfinite(intro.odd_even_gap_ms) ? (intro.odd_even_gap_ms / 220.0) : 1.0;
        const double confidence_term = 1.0 - std::min(1.0, std::max(0.0, probes[i].conf));
        const double score =
            (2.4 * tempo_term) +
            (1.2 * phase_term) +
            (0.9 * stability_term) +
            (0.5 * confidence_term);
        if (score < decision.selected_score) {
            second_best = decision.selected_score;
            decision.selected_score = score;
            decision.selected_index = i;
        } else if (score < second_best) {
            second_best = score;
        }
    }
    decision.score_margin = std::isfinite(second_best)
        ? std::max(0.0, second_best - decision.selected_score)
        : 0.0;
    const auto& intro = probe_intro_metrics[decision.selected_index];
    const bool severe_intro_miss =
        std::isfinite(intro.median_abs_ms) && intro.median_abs_ms > 220.0;
    const bool intro_miss_with_weak_conf =
        std::isfinite(intro.median_abs_ms) &&
        intro.median_abs_ms > 170.0 &&
        probes[decision.selected_index].conf < 0.80;
    decision.low_confidence =
        (probes[decision.selected_index].conf < 0.55) ||
        (probe_mode_errors[decision.selected_index] > 0.03) ||
        severe_intro_miss ||
        intro_miss_with_weak_conf ||
        (std::isfinite(intro.odd_even_gap_ms) && intro.odd_even_gap_ms > 220.0);
    return decision;
}

WindowPhaseGate sparse_evaluate_window_phase_gate(const SparseWindowPhaseMetrics& metrics,
                                                  double bpm_hint) {
    WindowPhaseGate gate;
    gate.has_data = metrics.count >= 10 &&
                    std::isfinite(metrics.median_ms) &&
                    std::isfinite(metrics.median_abs_ms);
    if (!gate.has_data) {
        return gate;
    }
    gate.beat_ms = bpm_hint > 0.0 ? (60000.0 / bpm_hint) : 500.0;
    gate.signed_limit_ms = sparse_signed_phase_limit_ms(bpm_hint);
    gate.abs_limit_ms = sparse_abs_phase_limit_ms(bpm_hint);
    gate.signed_exceeds = std::fabs(metrics.median_ms) > gate.signed_limit_ms;
    gate.abs_exceeds = metrics.median_abs_ms > gate.abs_limit_ms;
    gate.unstable_and = gate.signed_exceeds && gate.abs_exceeds;
    gate.unstable_or = gate.signed_exceeds || gate.abs_exceeds;
    return gate;
}

} // namespace detail
} // namespace beatit
