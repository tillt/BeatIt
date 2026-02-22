//
//  phase_metrics.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/sparse/phase_metrics.h"
#include "beatit/sparse/waveform.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace beatit {
namespace detail {

double sparse_signed_phase_limit_ms(double bpm_hint) {
    const double beat_ms = bpm_hint > 0.0 ? (60000.0 / bpm_hint) : 500.0;
    return std::max(30.0, beat_ms * 0.12);
}

double sparse_abs_phase_limit_ms(double bpm_hint) {
    const double beat_ms = bpm_hint > 0.0 ? (60000.0 / bpm_hint) : 500.0;
    return std::max(45.0, beat_ms * 0.18);
}

SparseWindowPhaseMetrics measure_sparse_window_phase(const AnalysisResult& result,
                                                     double bpm_hint,
                                                     double window_start_s,
                                                     double probe_duration,
                                                     double sample_rate,
                                                     const SparseSampleProvider& provider) {
    SparseWindowPhaseMetrics metrics;
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

    const double signed_limit_ms = sparse_signed_phase_limit_ms(bpm_hint);
    const double abs_limit_ms = sparse_abs_phase_limit_ms(bpm_hint);

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

} // namespace detail
} // namespace beatit
