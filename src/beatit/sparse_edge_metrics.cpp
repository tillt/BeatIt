//
//  sparse_edge_metrics.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "sparse_edge_metrics.h"

#include "beatit/sparse_waveform.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace beatit {
namespace detail {

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

} // namespace detail
} // namespace beatit
