//
//  refinement.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/sparse/refinement.h"

#include "beatit/logging.hpp"
#include "beatit/sparse/waveform.h"
#include "refine_common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace beatit {
namespace detail {

namespace {

std::size_t nearest_index(const std::vector<unsigned long long>& beats,
                          unsigned long long frame) {
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
}

} // namespace

double sparse_median_frame_diff(const std::vector<unsigned long long>& frames) {
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

void apply_sparse_bounded_grid_refit(AnalysisResult& result, double sample_rate) {
    if (!(sample_rate > 0.0)) {
        return;
    }

    auto& projected = result.coreml_beat_projected_sample_frames;
    const auto& observed = result.coreml_beat_sample_frames;
    if (projected.size() < 32 || observed.size() < 32) {
        return;
    }

    const double base_step = sparse_median_frame_diff(projected);
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

void apply_sparse_anchor_state_refit(AnalysisResult& result,
                                     double sample_rate,
                                     double probe_duration,
                                     const std::vector<SparseProbeObservation>& probes) {
    if (sample_rate <= 0.0 || probes.size() < 2) {
        return;
    }

    std::vector<unsigned long long>* projected = nullptr;
    if (!result.coreml_beat_projected_sample_frames.empty()) {
        projected = &result.coreml_beat_projected_sample_frames;
    } else if (!result.coreml_beat_sample_frames.empty()) {
        projected = &result.coreml_beat_sample_frames;
    }
    if (!projected || projected->size() < 64) {
        return;
    }

    struct AnchorObservation {
        double local_step = 0.0;
        double weight = 0.0;
        double start = 0.0;
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
            return sparse_median_frame_diff(beats);
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

    const double base_step = sparse_median_frame_diff(*projected);
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

    BEATIT_LOG_DEBUG("Sparse anchor state refit:"
                     << " anchors=" << anchors.size()
                     << " spread_ratio=" << spread_ratio
                     << " step_target=" << step_target
                     << " ratio=" << raw_ratio
                     << " ratio_applied=" << ratio
                     << " base_step=" << base_step);
}

} // namespace detail
} // namespace beatit
