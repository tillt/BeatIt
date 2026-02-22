//
//  sparse_refinement.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

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

namespace {

double median_diff(const std::vector<unsigned long long>& frames) {
    if (frames.size() < 2) {
        return 0.0;
    }
    std::vector<double> diffs;
    diffs.reserve(frames.size() - 1);
    for (std::size_t i = 1; i < frames.size(); ++i) {
        if (frames[i] > frames[i - 1]) {
            diffs.push_back(static_cast<double>(frames[i] - frames[i - 1]));
        }
    }
    if (diffs.empty()) {
        return 0.0;
    }
    auto mid = diffs.begin() + static_cast<long>(diffs.size() / 2);
    std::nth_element(diffs.begin(), mid, diffs.end());
    return *mid;
}

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

struct EdgeOffsetMetrics {
    double median_ms = std::numeric_limits<double>::infinity();
    double mad_ms = std::numeric_limits<double>::infinity();
    std::size_t count = 0;
};

WindowPhaseMetrics measure_window_phase(const AnalysisResult& result,
                                        double bpm_hint,
                                        double window_start_s,
                                        double probe_duration,
                                        double sample_rate,
                                        const SparseSampleProvider& provider) {
    WindowPhaseMetrics metrics;
    if (sample_rate <= 0.0 || bpm_hint <= 0.0 || !provider) {
        return metrics;
    }
    const auto& beats = sparse_select_beats(result);
    if (beats.size() < 12) {
        return metrics;
    }

    const unsigned long long window_start_frame = static_cast<unsigned long long>(
        std::llround(std::max(0.0, window_start_s) * sample_rate));
    const unsigned long long window_end_frame = static_cast<unsigned long long>(
        std::llround(std::max(0.0, window_start_s + probe_duration) * sample_rate));
    auto begin_it = std::lower_bound(beats.begin(), beats.end(), window_start_frame);
    auto end_it = std::upper_bound(beats.begin(), beats.end(), window_end_frame);
    if (begin_it == end_it || std::distance(begin_it, end_it) < 8) {
        return metrics;
    }

    const std::size_t radius = sparse_waveform_radius(sample_rate, bpm_hint);
    if (radius == 0) {
        return metrics;
    }

    const std::size_t margin = radius + static_cast<std::size_t>(std::llround(sample_rate * 1.5));
    const std::size_t first_frame = static_cast<std::size_t>(*begin_it);
    const std::size_t last_frame = static_cast<std::size_t>(*(end_it - 1));
    const std::size_t segment_start = first_frame > margin ? first_frame - margin : 0;
    const std::size_t segment_end = last_frame + margin;
    const double segment_start_s = static_cast<double>(segment_start) / sample_rate;
    const double segment_duration_s =
        static_cast<double>(std::max<std::size_t>(1, segment_end - segment_start)) / sample_rate;

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
                           sample_rate,
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
}

EdgeOffsetMetrics measure_edge_offsets(const std::vector<unsigned long long>& beats,
                                       double bpm_hint,
                                       bool from_end,
                                       double sample_rate,
                                       const SparseSampleProvider& provider) {
    EdgeOffsetMetrics metrics;
    if (sample_rate <= 0.0 || bpm_hint <= 0.0 || !provider || beats.size() < 16) {
        return metrics;
    }
    const std::size_t probe_beats = std::min<std::size_t>(32, beats.size());
    std::vector<unsigned long long> edge_beats;
    edge_beats.reserve(probe_beats);
    if (from_end) {
        edge_beats.insert(edge_beats.end(),
                          beats.end() - static_cast<long>(probe_beats),
                          beats.end());
    } else {
        edge_beats.insert(edge_beats.end(),
                          beats.begin(),
                          beats.begin() + static_cast<long>(probe_beats));
    }
    if (edge_beats.empty()) {
        return metrics;
    }

    const std::size_t radius = sparse_waveform_radius(sample_rate, bpm_hint);
    if (radius == 0) {
        return metrics;
    }
    const std::size_t margin = radius + static_cast<std::size_t>(std::llround(sample_rate * 1.5));
    const std::size_t first_frame = static_cast<std::size_t>(edge_beats.front());
    const std::size_t last_frame = static_cast<std::size_t>(edge_beats.back());
    const std::size_t segment_start = first_frame > margin ? first_frame - margin : 0;
    const std::size_t segment_end = last_frame + margin;
    const double segment_start_s = static_cast<double>(segment_start) / sample_rate;
    const double segment_duration_s =
        static_cast<double>(std::max<std::size_t>(1, segment_end - segment_start)) / sample_rate;

    std::vector<float> samples;
    if (!sparse_load_samples(provider, segment_start_s, segment_duration_s, &samples)) {
        return metrics;
    }

    std::vector<double> offsets_ms;
    offsets_ms.reserve(edge_beats.size());
    sparse_collect_offsets(edge_beats,
                           0,
                           edge_beats.size(),
                           segment_start,
                           samples,
                           radius,
                           SparsePeakMode::ThresholdedLocalMax,
                           sample_rate,
                           &offsets_ms,
                           nullptr);
    if (offsets_ms.size() < 8) {
        return metrics;
    }
    metrics.count = offsets_ms.size();
    metrics.median_ms = sparse_median_inplace(&offsets_ms);
    std::vector<double> abs_dev;
    abs_dev.reserve(offsets_ms.size());
    for (double v : offsets_ms) {
        abs_dev.push_back(std::fabs(v - metrics.median_ms));
    }
    metrics.mad_ms = sparse_median_inplace(&abs_dev);
    return metrics;
}

} // namespace

void apply_sparse_bounded_grid_refit(AnalysisResult* result, double sample_rate) {
    if (!result) {
        return;
    }
    if (!(sample_rate > 0.0)) {
        return;
    }

    auto& projected = result->coreml_beat_projected_sample_frames;
    const auto& observed = result->coreml_beat_sample_frames;
    if (projected.size() < 32 || observed.size() < 32) {
        return;
    }

    const double base_step = median_diff(projected);
    if (!(base_step > 0.0)) {
        return;
    }

    std::vector<double> errors;
    errors.reserve(projected.size());
    for (unsigned long long frame : projected) {
        auto it = std::lower_bound(observed.begin(), observed.end(), frame);
        unsigned long long nearest = observed.front();
        if (it == observed.end()) {
            nearest = observed.back();
        } else {
            nearest = *it;
            if (it != observed.begin()) {
                const unsigned long long prev = *(it - 1);
                if (frame - prev < nearest - frame) {
                    nearest = prev;
                }
            }
        }
        errors.push_back(static_cast<double>(nearest) - static_cast<double>(frame));
    }

    const std::size_t edge = std::min<std::size_t>(64, errors.size() / 2);
    if (edge < 8) {
        return;
    }
    std::vector<double> head(errors.begin(), errors.begin() + static_cast<long>(edge));
    std::vector<double> tail(errors.end() - static_cast<long>(edge), errors.end());
    const double head_med = sparse_median_inplace(&head);
    const double tail_med = sparse_median_inplace(&tail);
    const double err_delta = tail_med - head_med;
    const double beats_span = static_cast<double>(projected.size() - 1);
    if (!(beats_span > 0.0)) {
        return;
    }
    const double per_beat_adjust = err_delta / beats_span;
    const double ratio = 1.0 + (per_beat_adjust / base_step);
    const double clamped_ratio = std::max(0.998, std::min(1.002, ratio));
    if (std::abs(clamped_ratio - 1.0) < 1e-4) {
        return;
    }

    const long long anchor = static_cast<long long>(projected.front());
    for (std::size_t i = 0; i < projected.size(); ++i) {
        const long long current = static_cast<long long>(projected[i]);
        const double rel = static_cast<double>(current - anchor);
        const long long adjusted = anchor + static_cast<long long>(std::llround(rel * clamped_ratio));
        projected[i] = static_cast<unsigned long long>(std::max<long long>(0, adjusted));
    }
}

void apply_sparse_anchor_state_refit(AnalysisResult* result,
                                     double sample_rate,
                                     double probe_duration,
                                     const std::vector<SparseProbeObservation>& probes,
                                     bool verbose) {
    if (!result || sample_rate <= 0.0 || probes.size() < 2) {
        return;
    }

    std::vector<unsigned long long>* projected = nullptr;
    if (!result->coreml_beat_projected_sample_frames.empty()) {
        projected = &result->coreml_beat_projected_sample_frames;
    } else if (!result->coreml_beat_sample_frames.empty()) {
        projected = &result->coreml_beat_sample_frames;
    }
    if (!projected || projected->size() < 64) {
        return;
    }

    struct AnchorObservation {
        double local_step = 0.0;
        double weight = 0.0;
        double start = 0.0;
    };

    auto nearest_index = [](const std::vector<unsigned long long>& beats,
                            unsigned long long frame) -> std::size_t {
        if (beats.empty()) {
            return 0;
        }
        auto it = std::lower_bound(beats.begin(), beats.end(), frame);
        if (it == beats.end()) {
            return beats.size() - 1;
        }
        const std::size_t right = static_cast<std::size_t>(it - beats.begin());
        if (right == 0) {
            return 0;
        }
        const std::size_t left = right - 1;
        const unsigned long long left_frame = beats[left];
        const unsigned long long right_frame = beats[right];
        return (frame - left_frame) <= (right_frame - frame) ? left : right;
    };

    auto local_step_around = [&](const std::vector<unsigned long long>& beats,
                                 std::size_t center) -> double {
        if (beats.size() < 8) {
            return 0.0;
        }
        const std::size_t left = center > 12 ? center - 12 : 1;
        const std::size_t right = std::min<std::size_t>(beats.size() - 1, center + 12);
        std::vector<double> diffs;
        diffs.reserve(right - left + 1);
        for (std::size_t i = left; i <= right; ++i) {
            if (beats[i] > beats[i - 1]) {
                diffs.push_back(static_cast<double>(beats[i] - beats[i - 1]));
            }
        }
        if (diffs.size() < 4) {
            return median_diff(beats);
        }
        return sparse_median_inplace(&diffs);
    };

    std::vector<AnchorObservation> anchors;
    anchors.reserve(probes.size());
    for (const auto& probe : probes) {
        const auto& probe_beats = !probe.analysis.coreml_beat_projected_sample_frames.empty()
            ? probe.analysis.coreml_beat_projected_sample_frames
            : probe.analysis.coreml_beat_sample_frames;
        if (probe_beats.size() < 64) {
            continue;
        }
        const double center_s = std::max(0.0, probe.start + (probe_duration * 0.5));
        const unsigned long long center_frame =
            static_cast<unsigned long long>(std::llround(center_s * sample_rate));
        const std::size_t src_idx = nearest_index(probe_beats, center_frame);
        if (src_idx >= probe_beats.size()) {
            continue;
        }
        const double local_step = local_step_around(probe_beats, src_idx);
        if (!(local_step > 0.0)) {
            continue;
        }
        const double weight = std::clamp(probe.conf, 0.15, 1.0);
        anchors.push_back({local_step, weight, probe.start});
    }
    if (anchors.size() < 2) {
        return;
    }

    const double base_step = median_diff(*projected);
    if (!(base_step > 0.0)) {
        return;
    }

    auto normalize_step = [&](double step) -> double {
        if (!(step > 0.0)) {
            return 0.0;
        }
        const double harmonics[] = {0.5, 1.0, 2.0, 1.5, (2.0 / 3.0), 3.0};
        double best = step;
        double best_err = std::numeric_limits<double>::infinity();
        for (double h : harmonics) {
            const double candidate = step * h;
            const double err = std::fabs(candidate - base_step);
            if (err < best_err) {
                best_err = err;
                best = candidate;
            }
        }
        return best;
    };

    std::vector<double> normalized_steps;
    normalized_steps.reserve(anchors.size());
    double weighted_step_sum = 0.0;
    double weighted_sum = 0.0;
    double step_min = std::numeric_limits<double>::infinity();
    double step_max = 0.0;
    for (const auto& a : anchors) {
        const double normalized = normalize_step(a.local_step);
        if (!(normalized > 0.0)) {
            continue;
        }
        normalized_steps.push_back(normalized);
        weighted_step_sum += a.weight * normalized;
        weighted_sum += a.weight;
        step_min = std::min(step_min, normalized);
        step_max = std::max(step_max, normalized);
    }
    if (normalized_steps.size() < 2 || !(weighted_sum > 0.0)) {
        return;
    }

    const double spread_ratio = (step_max - step_min) / std::max(1e-6, base_step);
    if (spread_ratio > 0.004) {
        return;
    }

    const double step_target = weighted_step_sum / weighted_sum;
    if (!(step_target > 0.0)) {
        return;
    }
    const double raw_ratio = step_target / base_step;
    const double ratio = std::clamp(raw_ratio, 0.9997, 1.0003);
    if (std::abs(ratio - 1.0) < 1e-5) {
        return;
    }

    const long long anchor = static_cast<long long>(projected->front());
    const double adjusted_step = base_step * ratio;
    if (!(adjusted_step > 0.0)) {
        return;
    }

    for (std::size_t i = 0; i < projected->size(); ++i) {
        const double rel = static_cast<double>(i) * adjusted_step;
        const long long adjusted = anchor + static_cast<long long>(std::llround(rel));
        (*projected)[i] =
            static_cast<unsigned long long>(std::max<long long>(0, adjusted));
    }

    if (verbose) {
        std::cerr << "Sparse anchor state refit:"
                  << " anchors=" << anchors.size()
                  << " spread_ratio=" << spread_ratio
                  << " step_target=" << step_target
                  << " ratio=" << raw_ratio
                  << " ratio_applied=" << ratio
                  << " base_step=" << base_step
                  << "\n";
    }
}

void apply_sparse_waveform_edge_refit(AnalysisResult* result,
                                      const SparseWaveformRefitParams& params) {
    if (!result || !params.config || !params.provider || !params.probes) {
        return;
    }
    const CoreMLConfig& original_config = *params.config;
    const SparseSampleProvider& provider = *params.provider;
    const std::vector<SparseProbeObservation>& probes = *params.probes;
    const double sample_rate = params.sample_rate;
    const double probe_duration = params.probe_duration;
    if (sample_rate <= 0.0 || probes.empty() || probe_duration <= 0.0) {
        return;
    }

    const bool second_pass_enabled = []() {
        const char* v = std::getenv("BEATIT_EDGE_REFIT_SECOND_PASS");
        if (!v || v[0] == '\0') {
            return true;
        }
        return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' ||
                 v[0] == 'n' || v[0] == 'N');
    }();

    std::vector<unsigned long long>* projected = nullptr;
    if (!result->coreml_beat_projected_sample_frames.empty()) {
        projected = &result->coreml_beat_projected_sample_frames;
    } else if (!result->coreml_beat_sample_frames.empty()) {
        projected = &result->coreml_beat_sample_frames;
    }
    if (!projected || projected->size() < 64) {
        return;
    }
    const std::vector<unsigned long long> projected_before_refit = *projected;

    double bpm_hint = result->estimated_bpm > 0.0f
        ? static_cast<double>(result->estimated_bpm)
        : 0.0;
    if (!(bpm_hint > 0.0) &&
        params.estimate_bpm_from_beats &&
        *params.estimate_bpm_from_beats) {
        bpm_hint = (*params.estimate_bpm_from_beats)(*projected, sample_rate);
    }
    if (!(bpm_hint > 0.0)) {
        return;
    }

    auto measure_middle_windows = [&](const std::vector<unsigned long long>& beats) {
        WindowPhaseMetrics between_metrics;
        WindowPhaseMetrics middle_metrics;
        if (beats.size() < 32) {
            return std::pair<WindowPhaseMetrics, WindowPhaseMetrics>{between_metrics, middle_metrics};
        }
        AnalysisResult tmp;
        tmp.coreml_beat_projected_sample_frames = beats;
        between_metrics = measure_window_phase(tmp,
                                               bpm_hint,
                                               params.between_probe_start,
                                               probe_duration,
                                               sample_rate,
                                               provider);
        middle_metrics = measure_window_phase(tmp,
                                              bpm_hint,
                                              params.middle_probe_start,
                                              probe_duration,
                                              sample_rate,
                                              provider);
        return std::pair<WindowPhaseMetrics, WindowPhaseMetrics>{between_metrics, middle_metrics};
    };

    std::size_t first_probe_index = 0;
    std::size_t last_probe_index = 0;
    for (std::size_t i = 1; i < probes.size(); ++i) {
        if (probes[i].start < probes[first_probe_index].start) {
            first_probe_index = i;
        }
        if (probes[i].start > probes[last_probe_index].start) {
            last_probe_index = i;
        }
    }
    const double first_probe_start = probes[first_probe_index].start;
    const double last_probe_start = probes[last_probe_index].start;
    if (std::abs(last_probe_start - first_probe_start) < 1.0) {
        return;
    }
    const auto clamp_window_start = [&](double s) {
        return std::clamp(s, first_probe_start, last_probe_start);
    };

    auto select_window_beats = [&](const std::vector<unsigned long long>& beats,
                                   double window_start_s) {
        std::vector<unsigned long long> selected;
        if (sample_rate <= 0.0 || beats.empty()) {
            return selected;
        }
        const unsigned long long window_start_frame = static_cast<unsigned long long>(
            std::llround(std::max(0.0, window_start_s) * sample_rate));
        const unsigned long long window_end_frame = static_cast<unsigned long long>(
            std::llround(std::max(0.0, window_start_s + probe_duration) * sample_rate));
        auto begin_it = std::lower_bound(beats.begin(), beats.end(), window_start_frame);
        auto end_it = std::upper_bound(beats.begin(), beats.end(), window_end_frame);
        if (begin_it < end_it) {
            selected.insert(selected.end(), begin_it, end_it);
        }
        return selected;
    };
    auto measure_window_pair = [&](const std::vector<unsigned long long>& beats,
                                   double first_window_start_s,
                                   double last_window_start_s) {
        const std::vector<unsigned long long> first_window_beats =
            select_window_beats(beats, first_window_start_s);
        const std::vector<unsigned long long> last_window_beats =
            select_window_beats(beats, last_window_start_s);
        return std::pair<EdgeOffsetMetrics, EdgeOffsetMetrics>{
            measure_edge_offsets(first_window_beats, bpm_hint, false, sample_rate, provider),
            measure_edge_offsets(last_window_beats, bpm_hint, true, sample_rate, provider)};
    };
    auto measure_window_phase_pair = [&](const std::vector<unsigned long long>& beats,
                                         double first_window_start_s,
                                         double last_window_start_s) {
        AnalysisResult tmp;
        tmp.coreml_beat_projected_sample_frames = beats;
        return std::pair<WindowPhaseMetrics, WindowPhaseMetrics>{
            measure_window_phase(tmp,
                                 bpm_hint,
                                 first_window_start_s,
                                 probe_duration,
                                 sample_rate,
                                 provider),
            measure_window_phase(tmp,
                                 bpm_hint,
                                 last_window_start_s,
                                 probe_duration,
                                 sample_rate,
                                 provider)};
    };
    auto window_usable = [](const EdgeOffsetMetrics& m) {
        return m.count >= 10 &&
               std::isfinite(m.median_ms) &&
               std::isfinite(m.mad_ms) &&
               m.mad_ms <= 120.0;
    };
    const double shift_step = std::clamp(probe_duration * 0.25, 5.0, 20.0);
    double first_window_start = first_probe_start;
    double last_window_start = last_probe_start;
    EdgeOffsetMetrics intro;
    EdgeOffsetMetrics outro;
    std::size_t quality_shift_rounds = 0;
    for (; quality_shift_rounds < 6; ++quality_shift_rounds) {
        auto pair = measure_window_pair(*projected, first_window_start, last_window_start);
        intro = pair.first;
        outro = pair.second;
        const bool intro_ok = window_usable(intro);
        const bool outro_ok = window_usable(outro);
        if (intro_ok && outro_ok) {
            break;
        }
        bool moved = false;
        if (!intro_ok) {
            const double next = clamp_window_start(first_window_start + shift_step);
            if (next > first_window_start + 0.5) {
                first_window_start = next;
                moved = true;
            }
        }
        if (!outro_ok) {
            const double next = clamp_window_start(last_window_start - shift_step);
            if (next + 0.5 < last_window_start) {
                last_window_start = next;
                moved = true;
            }
        }
        if (!moved || (last_window_start - first_window_start) < std::max(1.0, shift_step)) {
            break;
        }
    }
    if (intro.count < 8 || outro.count < 8 ||
        !std::isfinite(intro.median_ms) || !std::isfinite(outro.median_ms)) {
        return;
    }

    auto apply_scale = [&](std::vector<unsigned long long>* beats,
                           double ratio,
                           double min_ratio,
                           double max_ratio,
                           double min_delta) -> double {
        if (!beats || beats->size() < 2) {
            return 1.0;
        }
        const double clamped_ratio = std::clamp(ratio, min_ratio, max_ratio);
        if (std::abs(clamped_ratio - 1.0) < min_delta) {
            return 1.0;
        }
        const long long anchor = static_cast<long long>(beats->front());
        for (std::size_t i = 0; i < beats->size(); ++i) {
            const long long current = static_cast<long long>((*beats)[i]);
            const double rel = static_cast<double>(current - anchor);
            const long long adjusted =
                anchor + static_cast<long long>(std::llround(rel * clamped_ratio));
            (*beats)[i] =
                static_cast<unsigned long long>(std::max<long long>(0, adjusted));
        }
        return clamped_ratio;
    };

    const auto compute_ratio = [&](const std::vector<unsigned long long>& beats,
                                   const EdgeOffsetMetrics& a,
                                   const EdgeOffsetMetrics& b) -> double {
        const double base_step = median_diff(beats);
        const double beats_span = static_cast<double>(beats.size() - 1);
        if (!(base_step > 0.0) || !(beats_span > 0.0)) {
            return 1.0;
        }
        const double err_delta_frames = ((b.median_ms - a.median_ms) * sample_rate) / 1000.0;
        const double per_beat_adjust = err_delta_frames / beats_span;
        return 1.0 + (per_beat_adjust / base_step);
    };

    const double ratio = compute_ratio(*projected, intro, outro);
    const double applied_ratio =
        apply_scale(projected, ratio, 0.9995, 1.0005, 1e-5);
    if (applied_ratio == 1.0) {
        return;
    }

    EdgeOffsetMetrics post_intro = intro;
    EdgeOffsetMetrics post_outro = outro;
    if (second_pass_enabled) {
        auto measured = measure_window_pair(*projected, first_window_start, last_window_start);
        post_intro = measured.first;
        post_outro = measured.second;
        if (post_intro.count >= 8 && post_outro.count >= 8 &&
            std::isfinite(post_intro.median_ms) && std::isfinite(post_outro.median_ms)) {
            const double post_delta = std::abs(post_outro.median_ms - post_intro.median_ms);
            std::vector<unsigned long long> candidate = *projected;
            const double pass2_ratio = compute_ratio(candidate, post_intro, post_outro);
            const double pass2_applied =
                apply_scale(&candidate, pass2_ratio, 0.9997, 1.0003, 1e-6);
            if (pass2_applied != 1.0) {
                auto candidate_measured =
                    measure_window_pair(candidate, first_window_start, last_window_start);
                const EdgeOffsetMetrics cand_intro = candidate_measured.first;
                const EdgeOffsetMetrics cand_outro = candidate_measured.second;
                if (cand_intro.count >= 8 && cand_outro.count >= 8 &&
                    std::isfinite(cand_intro.median_ms) && std::isfinite(cand_outro.median_ms)) {
                    const double cand_delta = std::abs(cand_outro.median_ms - cand_intro.median_ms);
                    const bool improves_delta = cand_delta <= post_delta;
                    const bool keeps_intro =
                        std::abs(cand_intro.median_ms) <= (std::abs(post_intro.median_ms) + 3.0);
                    if (improves_delta && keeps_intro) {
                        *projected = std::move(candidate);
                        post_intro = cand_intro;
                        post_outro = cand_outro;
                    }
                }
            }
        }
    }

    auto try_uniform_shift_on_windows =
        [&](const EdgeOffsetMetrics& base_intro,
            const EdgeOffsetMetrics& base_outro,
            double max_beat_fraction) {
            if (base_intro.count < 8 || base_outro.count < 8 ||
                !std::isfinite(base_intro.median_ms) || !std::isfinite(base_outro.median_ms)) {
                return false;
            }
            if ((base_intro.median_ms * base_outro.median_ms) <= 0.0) {
                return false;
            }
            const double mean_ms = 0.5 * (base_intro.median_ms + base_outro.median_ms);
            const double beat_ms = 60000.0 / std::max(1e-6, bpm_hint);
            const double max_shift_ms = std::max(25.0, beat_ms * max_beat_fraction);
            const double clamped_shift_ms = std::clamp(mean_ms, -max_shift_ms, max_shift_ms);
            const long long shift_frames = static_cast<long long>(
                std::llround((clamped_shift_ms * sample_rate) / 1000.0));
            if (shift_frames == 0) {
                return false;
            }

            std::vector<unsigned long long> candidate = *projected;
            for (std::size_t i = 0; i < candidate.size(); ++i) {
                const long long shifted = static_cast<long long>(candidate[i]) + shift_frames;
                candidate[i] = static_cast<unsigned long long>(std::max<long long>(0, shifted));
            }
            const auto measured =
                measure_window_pair(candidate, first_window_start, last_window_start);
            const EdgeOffsetMetrics cand_intro = measured.first;
            const EdgeOffsetMetrics cand_outro = measured.second;
            if (cand_intro.count < 8 || cand_outro.count < 8 ||
                !std::isfinite(cand_intro.median_ms) || !std::isfinite(cand_outro.median_ms)) {
                return false;
            }

            const double base_worst =
                std::max(std::abs(base_intro.median_ms), std::abs(base_outro.median_ms));
            const double cand_worst =
                std::max(std::abs(cand_intro.median_ms), std::abs(cand_outro.median_ms));
            if (cand_worst + 5.0 < base_worst) {
                *projected = std::move(candidate);
                return true;
            }
            return false;
        };
    if (try_uniform_shift_on_windows(post_intro, post_outro, 0.30)) {
        auto measured = measure_window_pair(*projected, first_window_start, last_window_start);
        post_intro = measured.first;
        post_outro = measured.second;
    }

    EdgeOffsetMetrics global_intro = measure_edge_offsets(*projected, bpm_hint, false, sample_rate, provider);
    EdgeOffsetMetrics global_outro = measure_edge_offsets(*projected, bpm_hint, true, sample_rate, provider);
    double global_guard_ratio = 1.0;
    if (global_intro.count >= 8 && global_outro.count >= 8 &&
        std::isfinite(global_intro.median_ms) && std::isfinite(global_outro.median_ms)) {
        const double global_delta = std::abs(global_outro.median_ms - global_intro.median_ms);
        if (global_delta > 30.0) {
            std::vector<unsigned long long> candidate = *projected;
            const double guard_ratio =
                compute_ratio(candidate, global_intro, global_outro);
            const double guard_applied =
                apply_scale(&candidate, guard_ratio, 0.99985, 1.00015, 1e-6);
            if (guard_applied != 1.0) {
                const EdgeOffsetMetrics cand_intro =
                    measure_edge_offsets(candidate, bpm_hint, false, sample_rate, provider);
                const EdgeOffsetMetrics cand_outro =
                    measure_edge_offsets(candidate, bpm_hint, true, sample_rate, provider);
                if (cand_intro.count >= 8 && cand_outro.count >= 8 &&
                    std::isfinite(cand_intro.median_ms) && std::isfinite(cand_outro.median_ms)) {
                    const double cand_delta =
                        std::abs(cand_outro.median_ms - cand_intro.median_ms);
                    const bool improves_delta = cand_delta <= (global_delta - 1.0);
                    const bool keeps_intro =
                        std::abs(cand_intro.median_ms) <= (std::abs(global_intro.median_ms) + 4.0);
                    if (improves_delta && keeps_intro) {
                        *projected = std::move(candidate);
                        global_intro = cand_intro;
                        global_outro = cand_outro;
                        global_guard_ratio = guard_applied;
                    }
                }
            }
        }
    }
    if (global_intro.count >= 8 && global_outro.count >= 8 &&
        std::isfinite(global_intro.median_ms) && std::isfinite(global_outro.median_ms)) {
        const double global_delta = std::abs(global_outro.median_ms - global_intro.median_ms);
        if (global_delta > 60.0 && global_delta <= 120.0) {
            std::vector<unsigned long long> candidate = *projected;
            const double guard_ratio =
                compute_ratio(candidate, global_intro, global_outro);
            const double guard_applied =
                apply_scale(&candidate, guard_ratio, 0.9996, 1.0004, 1e-6);
            if (guard_applied != 1.0) {
                const EdgeOffsetMetrics cand_intro =
                    measure_edge_offsets(candidate, bpm_hint, false, sample_rate, provider);
                const EdgeOffsetMetrics cand_outro =
                    measure_edge_offsets(candidate, bpm_hint, true, sample_rate, provider);
                if (cand_intro.count >= 8 && cand_outro.count >= 8 &&
                    std::isfinite(cand_intro.median_ms) && std::isfinite(cand_outro.median_ms)) {
                    const double cand_delta =
                        std::abs(cand_outro.median_ms - cand_intro.median_ms);
                    const double base_worst =
                        std::max(std::abs(global_intro.median_ms), std::abs(global_outro.median_ms));
                    const double cand_worst =
                        std::max(std::abs(cand_intro.median_ms), std::abs(cand_outro.median_ms));
                    const bool improves_delta = cand_delta + 2.0 < global_delta;
                    const bool keeps_worst_reasonable = cand_worst <= (base_worst + 10.0);
                    if (improves_delta && keeps_worst_reasonable) {
                        *projected = std::move(candidate);
                        global_intro = cand_intro;
                        global_outro = cand_outro;
                        global_guard_ratio = guard_applied;
                    }
                }
            }
        }
    }
    if (global_intro.count >= 8 && global_outro.count >= 8 &&
        std::isfinite(global_intro.median_ms) && std::isfinite(global_outro.median_ms) &&
        (global_intro.median_ms * global_outro.median_ms) > 0.0) {
        const double mean_ms = 0.5 * (global_intro.median_ms + global_outro.median_ms);
        const double beat_ms = 60000.0 / std::max(1e-6, bpm_hint);
        const double max_shift_ms = std::max(40.0, beat_ms * 0.35);
        const double clamped_shift_ms = std::clamp(mean_ms, -max_shift_ms, max_shift_ms);
        const long long shift_frames = static_cast<long long>(
            std::llround((clamped_shift_ms * sample_rate) / 1000.0));
        if (shift_frames != 0) {
            std::vector<unsigned long long> candidate = *projected;
            for (std::size_t i = 0; i < candidate.size(); ++i) {
                const long long shifted =
                    static_cast<long long>(candidate[i]) + shift_frames;
                candidate[i] =
                    static_cast<unsigned long long>(std::max<long long>(0, shifted));
            }
            const EdgeOffsetMetrics cand_intro =
                measure_edge_offsets(candidate, bpm_hint, false, sample_rate, provider);
            const EdgeOffsetMetrics cand_outro =
                measure_edge_offsets(candidate, bpm_hint, true, sample_rate, provider);
            if (cand_intro.count >= 8 && cand_outro.count >= 8 &&
                std::isfinite(cand_intro.median_ms) && std::isfinite(cand_outro.median_ms)) {
                const double base_worst =
                    std::max(std::abs(global_intro.median_ms), std::abs(global_outro.median_ms));
                const double cand_worst =
                    std::max(std::abs(cand_intro.median_ms), std::abs(cand_outro.median_ms));
                if (cand_worst + 5.0 < base_worst) {
                    *projected = std::move(candidate);
                    global_intro = cand_intro;
                    global_outro = cand_outro;
                }
            }
        }
    }

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
    auto score_phase_candidate = [&](const std::vector<unsigned long long>& beats)
        -> PhaseScoreSummary {
        PhaseScoreSummary out;
        if (beats.size() < 64) {
            return out;
        }
        const EdgeOffsetMetrics intro_m = measure_edge_offsets(beats, bpm_hint, false, sample_rate, provider);
        const EdgeOffsetMetrics outro_m = measure_edge_offsets(beats, bpm_hint, true, sample_rate, provider);
        const auto middle_pair = measure_middle_windows(beats);
        const auto edge_phase_pair =
            measure_window_phase_pair(beats, first_window_start, last_window_start);
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
    };

    auto apply_ratio_candidate = [&](const std::vector<unsigned long long>& beats,
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
    };

    double phase_try_base_score = std::numeric_limits<double>::infinity();
    double phase_try_minus_score = std::numeric_limits<double>::infinity();
    double phase_try_plus_score = std::numeric_limits<double>::infinity();
    int phase_try_selected = 0;
    bool phase_try_applied = false;
    {
        const auto base_score = score_phase_candidate(*projected);
        phase_try_base_score = base_score.score;
        if (base_score.valid && projected->size() >= 128) {
            const double intervals = static_cast<double>(projected->size() - 1);
            const double minus_intervals = intervals - 1.0;
            const double plus_intervals = intervals + 1.0;
            const double minus_ratio =
                (minus_intervals > 0.0) ? (intervals / minus_intervals) : 1.0;
            const double plus_ratio =
                (plus_intervals > 0.0) ? (intervals / plus_intervals) : 1.0;

            const std::vector<unsigned long long> minus_candidate =
                apply_ratio_candidate(*projected, minus_ratio);
            const std::vector<unsigned long long> plus_candidate =
                apply_ratio_candidate(*projected, plus_ratio);
            const auto minus_score = score_phase_candidate(minus_candidate);
            const auto plus_score = score_phase_candidate(plus_candidate);
            phase_try_minus_score = minus_score.score;
            phase_try_plus_score = plus_score.score;

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
                phase_try_selected = best_choice;
                phase_try_applied = true;
            }
        }
    }

    const EdgeOffsetMetrics pre_global_intro =
        measure_edge_offsets(projected_before_refit, bpm_hint, false, sample_rate, provider);
    const EdgeOffsetMetrics pre_global_outro =
        measure_edge_offsets(projected_before_refit, bpm_hint, true, sample_rate, provider);
    const auto pre_middle_pair = measure_middle_windows(projected_before_refit);
    const auto post_middle_pair = measure_middle_windows(*projected);
    const double pre_global_delta_ms =
        (pre_global_intro.count >= 8 && pre_global_outro.count >= 8 &&
         std::isfinite(pre_global_intro.median_ms) && std::isfinite(pre_global_outro.median_ms))
            ? std::abs(pre_global_outro.median_ms - pre_global_intro.median_ms)
            : std::numeric_limits<double>::infinity();
    const double post_global_delta_ms =
        (global_intro.count >= 8 && global_outro.count >= 8 &&
         std::isfinite(global_intro.median_ms) && std::isfinite(global_outro.median_ms))
            ? std::abs(global_outro.median_ms - global_intro.median_ms)
            : std::numeric_limits<double>::infinity();
    const double pre_between_abs_ms = pre_middle_pair.first.median_abs_ms;
    const double pre_middle_abs_ms = pre_middle_pair.second.median_abs_ms;
    const double post_between_abs_ms = post_middle_pair.first.median_abs_ms;
    const double post_middle_abs_ms = post_middle_pair.second.median_abs_ms;
    const double pre_phase_score =
        (std::isfinite(pre_global_delta_ms) &&
         std::isfinite(pre_between_abs_ms) &&
         std::isfinite(pre_middle_abs_ms))
            ? ((0.40 * pre_global_delta_ms) + pre_between_abs_ms + pre_middle_abs_ms)
            : std::numeric_limits<double>::infinity();
    const double post_phase_score =
        (std::isfinite(post_global_delta_ms) &&
         std::isfinite(post_between_abs_ms) &&
         std::isfinite(post_middle_abs_ms))
            ? ((0.40 * post_global_delta_ms) + post_between_abs_ms + post_middle_abs_ms)
            : std::numeric_limits<double>::infinity();

    if (original_config.verbose) {
        const double err_delta_frames =
            ((outro.median_ms - intro.median_ms) * sample_rate) / 1000.0;
        std::cerr << "Sparse edge refit:"
                  << " second_pass=" << (second_pass_enabled ? 1 : 0)
                  << " first_probe_start_s=" << first_probe_start
                  << " last_probe_start_s=" << last_probe_start
                  << " first_window_start_s=" << first_window_start
                  << " last_window_start_s=" << last_window_start
                  << " quality_shift_rounds=" << quality_shift_rounds
                  << " intro_ms=" << intro.median_ms
                  << " outro_ms=" << outro.median_ms
                  << " post_intro_ms=" << post_intro.median_ms
                  << " post_outro_ms=" << post_outro.median_ms
                  << " global_intro_ms=" << global_intro.median_ms
                  << " global_outro_ms=" << global_outro.median_ms
                  << " pre_global_delta_ms=" << pre_global_delta_ms
                  << " post_global_delta_ms=" << post_global_delta_ms
                  << " pre_between_abs_ms=" << pre_between_abs_ms
                  << " pre_middle_abs_ms=" << pre_middle_abs_ms
                  << " post_between_abs_ms=" << post_between_abs_ms
                  << " post_middle_abs_ms=" << post_middle_abs_ms
                  << " pre_phase_score=" << pre_phase_score
                  << " post_phase_score=" << post_phase_score
                  << " phase_try_base_score=" << phase_try_base_score
                  << " phase_try_minus_score=" << phase_try_minus_score
                  << " phase_try_plus_score=" << phase_try_plus_score
                  << " phase_try_selected=" << phase_try_selected
                  << " phase_try_applied=" << (phase_try_applied ? 1 : 0)
                  << " global_ratio_applied=" << global_guard_ratio
                  << " delta_frames=" << err_delta_frames
                  << " ratio=" << ratio
                  << " ratio_applied=" << applied_ratio
                  << " beats=" << projected->size()
                  << "\n";
    }
}

} // namespace detail
} // namespace beatit
