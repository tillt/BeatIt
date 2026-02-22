//
//  sparse_probe.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#include "beatit/sparse_probe.h"
#include "beatit/sparse_refinement.h"
#include "beatit/sparse_waveform.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace beatit {
namespace detail {

AnalysisResult analyze_sparse_probe_window(const CoreMLConfig& original_config,
                                           double sample_rate,
                                           double total_duration_seconds,
                                           const SparseSampleProvider& provider,
                                           const SparseRunProbe& run_probe_fn,
                                           const SparseEstimateBpm& estimate_bpm_from_beats_fn,
                                           const SparseNormalizeBpm& normalize_bpm_fn) {
    AnalysisResult result;

    const double sample_rate_ = sample_rate;
    auto run_probe = [&](double probe_start,
                         double probe_duration,
                         double forced_reference_bpm = 0.0) -> AnalysisResult {
        return run_probe_fn(probe_start, probe_duration, forced_reference_bpm);
    };
    auto estimate_bpm_from_beats_local = [&](const std::vector<unsigned long long>& beat_samples,
                                             double sample_rate_for_beats) -> float {
        return estimate_bpm_from_beats_fn(beat_samples, sample_rate_for_beats);
    };
    auto normalize_bpm_to_range_local = [&](float bpm,
                                            float min_bpm,
                                            float max_bpm) -> float {
        return normalize_bpm_fn(bpm, min_bpm, max_bpm);
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
    auto estimate_probe_confidence = [&](const AnalysisResult& r) -> double {
        const auto& beats = !r.coreml_beat_projected_sample_frames.empty()
            ? r.coreml_beat_projected_sample_frames
            : r.coreml_beat_sample_frames;
        if (beats.size() < 8 || sample_rate_ <= 0.0) {
            return 0.0;
        }
        std::vector<double> intervals;
        intervals.reserve(beats.size() - 1);
        for (std::size_t i = 1; i < beats.size(); ++i) {
            if (beats[i] > beats[i - 1]) {
                intervals.push_back(static_cast<double>(beats[i] - beats[i - 1]) / sample_rate_);
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
    };
    auto mode_penalty = [](double mode) {
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
    };
    auto expand_modes = [&](double bpm, double conf) {
        struct ModeCand {
            double bpm = 0.0;
            double conf = 0.0;
            double mode = 1.0;
            double penalty = 1.0;
        };
        std::vector<ModeCand> out;
        if (!(bpm > 0.0) || !(conf > 0.0)) {
            return out;
        }
        static const double kModes[] = {0.5, 1.0, 2.0};
        for (double m : kModes) {
            const double cand = bpm * m;
            if (cand >= original_config.min_bpm && cand <= original_config.max_bpm) {
                out.push_back({cand, conf, m, mode_penalty(m)});
            }
        }
        return out;
    };
    auto relative_diff = [](double a, double b) {
        const double mean = 0.5 * (a + b);
        return mean > 0.0 ? (std::abs(a - b) / mean) : 1.0;
    };

    auto probe_beat_count = [](const AnalysisResult& r) -> std::size_t {
        const auto& beats = sparse_select_beats(r);
        return beats.size();
    };
    auto estimate_intro_phase_abs_ms =
        [&](const AnalysisResult& r, double bpm_hint) -> double {
        if (sample_rate_ <= 0.0 || bpm_hint <= 0.0 || !provider) {
            return std::numeric_limits<double>::infinity();
        }
        const auto& beats = sparse_select_beats(r);
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
        const std::size_t radius = sparse_waveform_radius(sample_rate_, bpm_hint);
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
                               sample_rate_,
                               &signed_offsets_ms,
                               &abs_offsets_ms);
        if (abs_offsets_ms.size() < 8) {
            return std::numeric_limits<double>::infinity();
        }
        return sparse_median_inplace(&abs_offsets_ms);
    };
    auto run_probe_result = [&](double start_s) {
        ProbeResult p;
        p.start = clamp_start(start_s);
        p.analysis = run_probe(p.start, probe_duration);
        p.bpm = p.analysis.estimated_bpm;
        p.conf = estimate_probe_confidence(p.analysis);
        p.phase_abs_ms = estimate_intro_phase_abs_ms(p.analysis, p.bpm);
        return p;
    };
    auto probe_quality_score = [&](const ProbeResult& p) {
        const std::size_t beat_count = probe_beat_count(p.analysis);
        if (!(p.bpm > 0.0) || beat_count < 4) {
            return 0.0;
        }
        const double beat_factor =
            std::min(1.0, static_cast<double>(beat_count) / 24.0);
        const double phase_factor = std::isfinite(p.phase_abs_ms)
            ? (1.0 / (1.0 + (p.phase_abs_ms / 120.0)))
            : 0.15;
        return p.conf * beat_factor * phase_factor;
    };
    auto probe_is_usable = [&](const ProbeResult& p) {
        const std::size_t beat_count = probe_beat_count(p.analysis);
        return (p.bpm > 0.0) &&
               (beat_count >= 16) &&
               (p.conf >= 0.55) &&
               (!std::isfinite(p.phase_abs_ms) || p.phase_abs_ms <= 120.0);
    };
    auto seek_quality_probe = [&](double seed_start, bool shift_right) {
        double start_s = std::clamp(seed_start, min_allowed_start, max_allowed_start);
        ProbeResult best = run_probe_result(start_s);
        double best_score = probe_quality_score(best);
        if (probe_is_usable(best)) {
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
            const double score = probe_quality_score(candidate);
            if (score > best_score) {
                best = candidate;
                best_score = score;
            }
            if (probe_is_usable(candidate)) {
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
        const double incoming_score = probe_quality_score(probe);
        for (auto& existing : probes) {
            if (std::abs(existing.start - probe.start) < 1.0) {
                if (incoming_score > probe_quality_score(existing)) {
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

    auto consensus_from_probes = [&](const std::vector<ProbeResult>& values) {
        struct ConsensusCand {
            double bpm = 0.0;
            double conf = 0.0;
            double mode = 1.0;
            double penalty = 1.0;
        };
        std::vector<ConsensusCand> all_modes;
        for (const auto& p : values) {
            const auto modes = expand_modes(p.bpm, p.conf);
            for (const auto& m : modes) {
                all_modes.push_back({m.bpm, m.conf, m.mode, m.penalty});
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
            for (const auto& p : values) {
                const auto modes = expand_modes(p.bpm, p.conf);
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
    };

    struct IntroPhaseMetrics {
        double median_abs_ms = std::numeric_limits<double>::infinity();
        double odd_even_gap_ms = std::numeric_limits<double>::infinity();
        std::size_t count = 0;
    };
    auto measure_intro_phase = [&](const AnalysisResult& r, double bpm_hint) -> IntroPhaseMetrics {
        IntroPhaseMetrics metrics;
        if (sample_rate_ <= 0.0 || bpm_hint <= 0.0 || !provider) {
            return metrics;
        }
        const auto& beats = sparse_select_beats(r);
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
        const std::size_t radius = sparse_waveform_radius(sample_rate_, bpm_hint);
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
                               sample_rate_,
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
    };

    struct WindowPhaseMetrics {
        double median_ms = std::numeric_limits<double>::infinity();
        double median_abs_ms = std::numeric_limits<double>::infinity();
        double odd_even_gap_ms = std::numeric_limits<double>::infinity();
        double abs_p90_ms = std::numeric_limits<double>::infinity();
        double abs_p95_ms = std::numeric_limits<double>::infinity();
        double abs_limit_exceed_ratio = std::numeric_limits<double>::infinity();
        double signed_limit_exceed_ratio = std::numeric_limits<double>::infinity();
        std::size_t count = 0;
    };
    auto measure_window_phase = [&](const AnalysisResult& r,
                                    double bpm_hint,
                                    double window_start_s) -> WindowPhaseMetrics {
        WindowPhaseMetrics metrics;
        if (sample_rate_ <= 0.0 || bpm_hint <= 0.0 || !provider) {
            return metrics;
        }
        const auto& beats = sparse_select_beats(r);
        if (beats.size() < 12) {
            return metrics;
        }

        const unsigned long long window_start_frame = static_cast<unsigned long long>(
            std::llround(std::max(0.0, window_start_s) * sample_rate_));
        const unsigned long long window_end_frame = static_cast<unsigned long long>(
            std::llround(std::max(0.0, window_start_s + probe_duration) * sample_rate_));
        auto begin_it = std::lower_bound(beats.begin(), beats.end(), window_start_frame);
        auto end_it = std::upper_bound(beats.begin(), beats.end(), window_end_frame);
        if (begin_it == end_it || std::distance(begin_it, end_it) < 8) {
            return metrics;
        }

        const std::size_t radius = sparse_waveform_radius(sample_rate_, bpm_hint);
        if (radius == 0) {
            return metrics;
        }

        const std::size_t margin = radius + static_cast<std::size_t>(std::llround(sample_rate_ * 1.5));
        const std::size_t first_frame = static_cast<std::size_t>(*begin_it);
        const std::size_t last_frame = static_cast<std::size_t>(*(end_it - 1));
        const std::size_t segment_start = first_frame > margin ? first_frame - margin : 0;
        const std::size_t segment_end = last_frame + margin;
        const double segment_start_s = static_cast<double>(segment_start) / sample_rate_;
        const double segment_duration_s =
            static_cast<double>(std::max<std::size_t>(1, segment_end - segment_start)) / sample_rate_;

        std::vector<float> samples;
        if (!sparse_load_samples(provider, segment_start_s, segment_duration_s, &samples)) {
            return metrics;
        }

        std::vector<double> signed_offsets_ms;
        std::vector<double> abs_offsets_ms;
        signed_offsets_ms.reserve(static_cast<std::size_t>(std::distance(begin_it, end_it)));
        abs_offsets_ms.reserve(static_cast<std::size_t>(std::distance(begin_it, end_it)));
        const std::size_t begin_idx =
            static_cast<std::size_t>(std::distance(beats.begin(), begin_it));
        const std::size_t end_idx =
            static_cast<std::size_t>(std::distance(beats.begin(), end_it));
        sparse_collect_offsets(beats,
                               begin_idx,
                               end_idx,
                               segment_start,
                               samples,
                               radius,
                               SparsePeakMode::ThresholdedLocalMax,
                               sample_rate_,
                               &signed_offsets_ms,
                               &abs_offsets_ms);

        if (abs_offsets_ms.size() < 8) {
            return metrics;
        }
        metrics.count = abs_offsets_ms.size();
        metrics.median_ms = sparse_median_inplace(&signed_offsets_ms);
        metrics.median_abs_ms = sparse_median_inplace(&abs_offsets_ms);
        const double beat_ms = bpm_hint > 0.0 ? (60000.0 / bpm_hint) : 500.0;
        const double signed_limit_ms = std::max(30.0, beat_ms * 0.12);
        const double abs_limit_ms = std::max(45.0, beat_ms * 0.18);
        std::size_t abs_exceed_count = 0;
        std::size_t signed_exceed_count = 0;
        for (double v : abs_offsets_ms) {
            if (v > abs_limit_ms) {
                ++abs_exceed_count;
            }
        }
        for (double v : signed_offsets_ms) {
            if (std::fabs(v) > signed_limit_ms) {
                ++signed_exceed_count;
            }
        }
        metrics.abs_limit_exceed_ratio =
            static_cast<double>(abs_exceed_count) / static_cast<double>(abs_offsets_ms.size());
        metrics.signed_limit_exceed_ratio =
            static_cast<double>(signed_exceed_count) / static_cast<double>(signed_offsets_ms.size());
        auto quantile = [](std::vector<double> values, double q) {
            if (values.empty()) {
                return std::numeric_limits<double>::infinity();
            }
            q = std::clamp(q, 0.0, 1.0);
            const std::size_t index = static_cast<std::size_t>(
                std::llround(q * static_cast<double>(values.size() - 1)));
            std::nth_element(values.begin(),
                             values.begin() + static_cast<long>(index),
                             values.end());
            return values[index];
        };
        metrics.abs_p90_ms = quantile(abs_offsets_ms, 0.90);
        metrics.abs_p95_ms = quantile(abs_offsets_ms, 0.95);

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
    };

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

    double consensus_bpm = consensus_from_probes(probes);
    if (probes.size() >= 2 && consensus_bpm > 0.0) {
        double max_err = 0.0;
        for (const auto& p : probes) {
            const auto modes = expand_modes(p.bpm, p.conf);
            double best_local = std::numeric_limits<double>::infinity();
            for (const auto& m : modes) {
                best_local = std::min(best_local, relative_diff(consensus_bpm, m.bpm) * m.penalty);
            }
            max_err = std::max(max_err, best_local);
        }
        if (max_err > 0.025 && probes.size() == 2) {
            push_unique_probe(run_probe_result(clamp_start(total * 0.5 - probe_duration * 0.5)));
        }
    }

    bool have_consensus = false;
    std::vector<double> probe_mode_errors;
    std::vector<IntroPhaseMetrics> probe_intro_metrics;
    std::vector<WindowPhaseMetrics> probe_middle_metrics;
    double left_probe_start = min_allowed_start;
    double right_probe_start = max_allowed_start;
    double middle_probe_start = clamp_start(total * 0.5 - probe_duration * 0.5);
    double between_probe_start = clamp_start(0.5 * (min_allowed_start + middle_probe_start));

    auto recompute_probe_scores = [&]() {
        consensus_bpm = consensus_from_probes(probes);
        have_consensus = consensus_bpm > 0.0;

        probe_mode_errors.assign(probes.size(), 1.0);
        for (std::size_t i = 0; i < probes.size(); ++i) {
            if (!have_consensus) {
                probe_mode_errors[i] = 0.0;
                continue;
            }
            const auto modes = expand_modes(probes[i].bpm, probes[i].conf);
            double mode_error = 1.0;
            for (const auto& m : modes) {
                mode_error = std::min(mode_error, relative_diff(consensus_bpm, m.bpm) * m.penalty);
            }
            probe_mode_errors[i] = mode_error;
        }

        const auto probe_extents = probe_start_extents();
        left_probe_start = probe_extents.first;
        right_probe_start = probe_extents.second;
        middle_probe_start = clamp_start(0.5 * (probe_extents.first + probe_extents.second));
        between_probe_start = clamp_start(0.5 * (probe_extents.first + middle_probe_start));

        probe_intro_metrics.assign(probes.size(), IntroPhaseMetrics{});
        probe_middle_metrics.assign(probes.size(), WindowPhaseMetrics{});
        for (std::size_t i = 0; i < probes.size(); ++i) {
            const double bpm_hint = have_consensus ? consensus_bpm : probes[i].bpm;
            probe_intro_metrics[i] = measure_intro_phase(probes[i].analysis, bpm_hint);
            probe_middle_metrics[i] = measure_window_phase(
                probes[i].analysis, bpm_hint, middle_probe_start);
        }
    };

    recompute_probe_scores();

    const double anchor_start = left_anchor_start;

    struct DecisionOutcome {
        std::size_t selected_index = 0;
        double selected_score = std::numeric_limits<double>::infinity();
        double score_margin = 0.0;
        bool low_confidence = true;
        std::string mode;
    };
    auto decide_unified = [&]() {
        DecisionOutcome out;
        out.mode = "unified";
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
            if (score < out.selected_score) {
                second_best = out.selected_score;
                out.selected_score = score;
                out.selected_index = i;
            } else if (score < second_best) {
                second_best = score;
            }
        }
        out.score_margin = std::isfinite(second_best)
            ? std::max(0.0, second_best - out.selected_score)
            : 0.0;
        const auto& intro = probe_intro_metrics[out.selected_index];
        const bool severe_intro_miss =
            std::isfinite(intro.median_abs_ms) && intro.median_abs_ms > 220.0;
        const bool intro_miss_with_weak_conf =
            std::isfinite(intro.median_abs_ms) &&
            intro.median_abs_ms > 170.0 &&
            probes[out.selected_index].conf < 0.80;
        out.low_confidence =
            (probes[out.selected_index].conf < 0.55) ||
            (probe_mode_errors[out.selected_index] > 0.03) ||
            severe_intro_miss ||
            intro_miss_with_weak_conf ||
            (std::isfinite(intro.odd_even_gap_ms) && intro.odd_even_gap_ms > 220.0);
        return out;
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
    auto evaluate_window_phase_gate = [&](const WindowPhaseMetrics& metrics,
                                          double bpm_hint) -> WindowPhaseGate {
        WindowPhaseGate gate;
        gate.has_data = metrics.count >= 10 &&
                        std::isfinite(metrics.median_ms) &&
                        std::isfinite(metrics.median_abs_ms);
        if (!gate.has_data) {
            return gate;
        }
        gate.beat_ms = bpm_hint > 0.0 ? (60000.0 / bpm_hint) : 500.0;
        gate.signed_limit_ms = std::max(30.0, gate.beat_ms * 0.12);
        gate.abs_limit_ms = std::max(45.0, gate.beat_ms * 0.18);
        gate.signed_exceeds = std::fabs(metrics.median_ms) > gate.signed_limit_ms;
        gate.abs_exceeds = metrics.median_abs_ms > gate.abs_limit_ms;
        gate.unstable_and = gate.signed_exceeds && gate.abs_exceeds;
        gate.unstable_or = gate.signed_exceeds || gate.abs_exceeds;
        return gate;
    };
    DecisionOutcome decision = decide_unified();
    std::size_t selected_index = decision.selected_index;
    double selected_score = decision.selected_score;
    double score_margin = decision.score_margin;
    bool low_confidence = decision.low_confidence;
    IntroPhaseMetrics selected_intro_metrics = probe_intro_metrics[selected_index];
    WindowPhaseMetrics selected_middle_metrics = probe_middle_metrics[selected_index];
    WindowPhaseMetrics selected_between_metrics;
    WindowPhaseMetrics selected_left_window_metrics;
    WindowPhaseMetrics selected_right_window_metrics;
    WindowPhaseGate selected_middle_gate;
    WindowPhaseGate selected_between_gate;
    WindowPhaseGate selected_left_gate;
    WindowPhaseGate selected_right_gate;
    bool middle_gate_triggered = false;
    bool consistency_gate_triggered = false;
    bool consistency_edges_calm = false;
    bool consistency_between_hot = false;
    bool consistency_middle_hot = false;
    bool interior_probe_added = false;

    auto refresh_selected_window_diagnostics = [&]() {
        const double selected_bpm_hint = have_consensus ? consensus_bpm : probes[selected_index].bpm;
        selected_middle_metrics = probe_middle_metrics[selected_index];
        selected_middle_gate = evaluate_window_phase_gate(selected_middle_metrics, selected_bpm_hint);
        selected_between_metrics = measure_window_phase(
            probes[selected_index].analysis, selected_bpm_hint, between_probe_start);
        selected_between_gate = evaluate_window_phase_gate(selected_between_metrics, selected_bpm_hint);
        selected_left_window_metrics = measure_window_phase(
            probes[selected_index].analysis, selected_bpm_hint, left_probe_start);
        selected_right_window_metrics = measure_window_phase(
            probes[selected_index].analysis, selected_bpm_hint, right_probe_start);
        selected_left_gate = evaluate_window_phase_gate(
            selected_left_window_metrics, selected_bpm_hint);
        selected_right_gate = evaluate_window_phase_gate(
            selected_right_window_metrics, selected_bpm_hint);
    };

    auto evaluate_consistency_gate = [&]() {
        constexpr double kHotAbsRatio = 0.75;
        constexpr double kHotSignedRatio = 0.85;
        constexpr double kCalmAbsRatio = 0.20;
        constexpr double kCalmSignedRatio = 0.25;

        const auto window_hot = [&](const WindowPhaseMetrics& metrics, const WindowPhaseGate& gate) {
            return gate.has_data &&
                   std::isfinite(metrics.abs_limit_exceed_ratio) &&
                   std::isfinite(metrics.signed_limit_exceed_ratio) &&
                   metrics.abs_limit_exceed_ratio >= kHotAbsRatio &&
                   metrics.signed_limit_exceed_ratio >= kHotSignedRatio;
        };
        const auto window_calm = [&](const WindowPhaseMetrics& metrics, const WindowPhaseGate& gate) {
            return gate.has_data &&
                   std::isfinite(metrics.abs_limit_exceed_ratio) &&
                   std::isfinite(metrics.signed_limit_exceed_ratio) &&
                   metrics.abs_limit_exceed_ratio <= kCalmAbsRatio &&
                   metrics.signed_limit_exceed_ratio <= kCalmSignedRatio;
        };

        consistency_between_hot = window_hot(selected_between_metrics, selected_between_gate);
        consistency_middle_hot = window_hot(selected_middle_metrics, selected_middle_gate);
        consistency_edges_calm = window_calm(selected_left_window_metrics, selected_left_gate) &&
                                 window_calm(selected_right_window_metrics, selected_right_gate);
        return consistency_edges_calm && (consistency_between_hot || consistency_middle_hot);
    };

    if (probes.size() == 2) {
        refresh_selected_window_diagnostics();
        middle_gate_triggered = selected_middle_gate.unstable_and;
        consistency_gate_triggered = evaluate_consistency_gate();
        if (middle_gate_triggered || consistency_gate_triggered) {
            push_unique_probe(run_probe_result(between_probe_start));
            recompute_probe_scores();
            decision = decide_unified();
            selected_index = decision.selected_index;
            selected_score = decision.selected_score;
            score_margin = decision.score_margin;
            low_confidence = decision.low_confidence;
            selected_intro_metrics = probe_intro_metrics[selected_index];
            refresh_selected_window_diagnostics();
            middle_gate_triggered = selected_middle_gate.unstable_and;
            consistency_gate_triggered = evaluate_consistency_gate();
            interior_probe_added = true;
        }
    }
    if (original_config.verbose) {
        refresh_selected_window_diagnostics();
        middle_gate_triggered = selected_middle_gate.unstable_and;
        consistency_gate_triggered = evaluate_consistency_gate();
    }

    // Sparse mode returns directly from the best probe to avoid an extra anchor pass.
    result = probes[selected_index].analysis;
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

    const bool needs_bounded_refit =
        low_confidence ||
        (std::isfinite(selected_intro_metrics.median_abs_ms) &&
         selected_intro_metrics.median_abs_ms > 60.0);
    if (needs_bounded_refit) {
        apply_sparse_bounded_grid_refit(&result, sample_rate_);
    }
    apply_sparse_anchor_state_refit(&result,
                                    sample_rate_,
                                    probe_duration,
                                    probes,
                                    original_config.verbose);

    SparseWaveformRefitParams waveform_refit_params;
    waveform_refit_params.config = &original_config;
    waveform_refit_params.provider = &provider;
    waveform_refit_params.estimate_bpm_from_beats = &estimate_bpm_from_beats_fn;
    waveform_refit_params.probes = &probes;
    waveform_refit_params.sample_rate = sample_rate_;
    waveform_refit_params.probe_duration = probe_duration;
    waveform_refit_params.between_probe_start = between_probe_start;
    waveform_refit_params.middle_probe_start = middle_probe_start;
    apply_sparse_waveform_edge_refit(&result, waveform_refit_params);

    {
        // Keep reported BPM consistent with the returned beat grid.
        const auto& bpm_frames = !result.coreml_beat_projected_sample_frames.empty()
            ? result.coreml_beat_projected_sample_frames
            : result.coreml_beat_sample_frames;
        const float grid_bpm = normalize_bpm_to_range_local(
            estimate_bpm_from_beats_local(bpm_frames, sample_rate_),
            std::max(1.0f, original_config.min_bpm),
            std::max(std::max(1.0f, original_config.min_bpm) + 1.0f, original_config.max_bpm));
        if (grid_bpm > 0.0f) {
            result.estimated_bpm = grid_bpm;
        } else if (have_consensus && consensus_bpm > 0.0) {
            result.estimated_bpm = static_cast<float>(consensus_bpm);
        }
    }
    {
        const auto& marker_feature_frames = result.coreml_beat_projected_feature_frames.empty()
            ? result.coreml_beat_feature_frames
            : result.coreml_beat_projected_feature_frames;
        const auto& marker_sample_frames = result.coreml_beat_projected_sample_frames.empty()
            ? result.coreml_beat_sample_frames
            : result.coreml_beat_projected_sample_frames;
        const auto& marker_downbeats = result.coreml_downbeat_projected_feature_frames.empty()
            ? result.coreml_downbeat_feature_frames
            : result.coreml_downbeat_projected_feature_frames;
        result.coreml_beat_events =
            build_shakespear_markers(marker_feature_frames,
                                     marker_sample_frames,
                                     marker_downbeats,
                                     &result.coreml_beat_activation,
                                     result.estimated_bpm,
                                     sample_rate_,
                                     original_config);
    }
    if (original_config.verbose) {
        std::cerr << "Sparse probes:";
        for (std::size_t i = 0; i < probes.size(); ++i) {
            const auto& p = probes[i];
            std::cerr << " start=" << p.start
                      << " bpm=" << p.bpm
                      << " conf=" << p.conf
                      << " mode_err=" << probe_mode_errors[i];
        }
        std::cerr << " consensus=" << consensus_bpm
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
                  << " consistency_edges_calm=" << (consistency_edges_calm ? 1 : 0)
                  << " consistency_between_hot=" << (consistency_between_hot ? 1 : 0)
                  << " consistency_middle_hot=" << (consistency_middle_hot ? 1 : 0)
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
                  << " repair=" << (low_confidence ? 1 : 0)
                  << "\n";
    }

    return result;
}

} // namespace detail
} // namespace beatit
