//
//  sparse_waveform.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/sparse_waveform.h"

#include <algorithm>
#include <cmath>

namespace beatit {
namespace detail {

const std::vector<unsigned long long>& sparse_select_beats(const AnalysisResult& result) {
    return !result.coreml_beat_projected_sample_frames.empty()
        ? result.coreml_beat_projected_sample_frames
        : result.coreml_beat_sample_frames;
}

bool sparse_load_samples(const SparseSampleProvider& provider,
                         double start_seconds,
                         double duration_seconds,
                         std::vector<float>* out_samples) {
    if (!provider || !out_samples) {
        return false;
    }
    out_samples->clear();
    const std::size_t received =
        provider(std::max(0.0, start_seconds), std::max(0.0, duration_seconds), out_samples);
    if (received == 0 || out_samples->empty()) {
        return false;
    }
    if (out_samples->size() > received) {
        out_samples->resize(received);
    }
    return !out_samples->empty();
}

std::size_t sparse_waveform_radius(double sample_rate, double bpm_hint) {
    if (!(sample_rate > 0.0) || !(bpm_hint > 0.0)) {
        return 0;
    }
    return static_cast<std::size_t>(
        std::llround(sample_rate * (60.0 / bpm_hint) * 0.6));
}

namespace {

std::size_t find_peak_absolute(const std::vector<float>& samples,
                               std::size_t start,
                               std::size_t end,
                               std::size_t fallback) {
    std::size_t best_peak = fallback;
    float best_value = 0.0f;
    for (std::size_t i = start; i <= end; ++i) {
        const float value = std::fabs(samples[i]);
        if (value > best_value) {
            best_value = value;
            best_peak = i;
        }
    }
    return best_value > 0.0f ? best_peak : fallback;
}

std::size_t find_peak_thresholded_localmax(const std::vector<float>& samples,
                                           std::size_t start,
                                           std::size_t end,
                                           std::size_t fallback) {
    float window_max = 0.0f;
    for (std::size_t i = start; i <= end; ++i) {
        window_max = std::max(window_max, std::fabs(samples[i]));
    }
    const float threshold = window_max * 0.6f;

    std::size_t best_peak = fallback;
    float best_value = 0.0f;
    for (std::size_t i = start + 1; i < end; ++i) {
        const float left = std::fabs(samples[i - 1]);
        const float value = std::fabs(samples[i]);
        const float right = std::fabs(samples[i + 1]);
        if (value < threshold) {
            continue;
        }
        if (value >= left && value > right && value > best_value) {
            best_value = value;
            best_peak = i;
        }
    }
    if (best_value > 0.0f) {
        return best_peak;
    }
    return find_peak_absolute(samples, start, end, fallback);
}

} // namespace

void sparse_collect_offsets(const std::vector<unsigned long long>& beat_frames,
                            std::size_t first_idx,
                            std::size_t last_idx_exclusive,
                            std::size_t segment_start_frame,
                            const std::vector<float>& samples,
                            std::size_t radius,
                            SparsePeakMode mode,
                            double sample_rate,
                            std::vector<double>* signed_offsets_ms,
                            std::vector<double>* abs_offsets_ms) {
    if (samples.empty() || radius == 0 || !(sample_rate > 0.0) || first_idx >= last_idx_exclusive) {
        return;
    }
    const std::size_t begin = std::min(first_idx, beat_frames.size());
    const std::size_t end_idx = std::min(last_idx_exclusive, beat_frames.size());
    if (begin >= end_idx) {
        return;
    }

    for (std::size_t k = begin; k < end_idx; ++k) {
        const std::size_t beat_frame = static_cast<std::size_t>(beat_frames[k]);
        if (beat_frame < segment_start_frame) {
            continue;
        }
        const std::size_t local_center =
            std::min<std::size_t>(samples.size() - 1, beat_frame - segment_start_frame);
        const std::size_t start = local_center > radius ? local_center - radius : 0;
        const std::size_t end = std::min(samples.size() - 1, local_center + radius);
        if (end <= start + 2) {
            continue;
        }

        std::size_t best_peak = local_center;
        if (mode == SparsePeakMode::AbsoluteMax) {
            best_peak = find_peak_absolute(samples, start, end, local_center);
            if (best_peak == local_center &&
                std::fabs(samples[best_peak]) <= 0.0f) {
                continue;
            }
        } else {
            best_peak = find_peak_thresholded_localmax(samples, start, end, local_center);
        }

        const double delta_frames = static_cast<double>(
            static_cast<long long>(best_peak) - static_cast<long long>(local_center));
        const double signed_ms = (delta_frames * 1000.0) / sample_rate;
        if (signed_offsets_ms) {
            signed_offsets_ms->push_back(signed_ms);
        }
        if (abs_offsets_ms) {
            abs_offsets_ms->push_back(std::fabs(signed_ms));
        }
    }
}

double sparse_median_inplace(std::vector<double>* values) {
    if (!values || values->empty()) {
        return 0.0;
    }
    auto mid = values->begin() + static_cast<long>(values->size() / 2);
    std::nth_element(values->begin(), mid, values->end());
    return *mid;
}

} // namespace detail
} // namespace beatit
