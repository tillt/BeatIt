//
//  usability_scan.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "usability_scan.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace beatit {
namespace detail {
namespace {

double clamp_unit(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double window_distance_penalty(const SparseUsabilityWindow& window,
                               const SparseUsabilityPickRequest& request) {
    return request.distance_weight * std::abs(window.start_seconds - request.target_seconds);
}

std::vector<double> build_window_starts(double total_duration_seconds,
                                        double window_duration_seconds,
                                        double hop_seconds) {
    std::vector<double> starts;
    if (!(total_duration_seconds > 0.0) ||
        !(window_duration_seconds > 0.0) ||
        !(hop_seconds > 0.0)) {
        return starts;
    }

    const double max_start = std::max(0.0, total_duration_seconds - window_duration_seconds);
    for (double start = 0.0; start <= max_start + 1e-9; start += hop_seconds) {
        starts.push_back(std::min(start, max_start));
        if (starts.back() >= max_start) {
            break;
        }
    }
    if (starts.empty()) {
        starts.push_back(0.0);
    }
    return starts;
}

std::vector<double> build_envelope(const std::vector<float>& samples, std::size_t hop) {
    std::vector<double> envelope;
    if (hop == 0) {
        return envelope;
    }
    envelope.reserve((samples.size() + hop - 1) / hop);
    for (std::size_t start = 0; start < samples.size(); start += hop) {
        const std::size_t end = std::min(samples.size(), start + hop);
        double sum = 0.0;
        for (std::size_t i = start; i < end; ++i) {
            sum += std::abs(static_cast<double>(samples[i]));
        }
        envelope.push_back(sum / static_cast<double>(std::max<std::size_t>(1, end - start)));
    }
    return envelope;
}

std::vector<double> build_onset_curve(const std::vector<double>& envelope) {
    std::vector<double> onset(envelope.size(), 0.0);
    for (std::size_t i = 1; i < envelope.size(); ++i) {
        onset[i] = std::max(0.0, envelope[i] - envelope[i - 1]);
    }
    return onset;
}

double normalized_autocorr_peak(const std::vector<double>& onset_curve,
                                std::size_t min_lag,
                                std::size_t max_lag) {
    if (onset_curve.size() < 8 || min_lag == 0 || min_lag > max_lag) {
        return 0.0;
    }

    double zero_lag = 0.0;
    for (double value : onset_curve) {
        zero_lag += value * value;
    }
    if (!(zero_lag > 1e-12)) {
        return 0.0;
    }

    double best = 0.0;
    const std::size_t clamped_max_lag = std::min(max_lag, onset_curve.size() - 1);
    for (std::size_t lag = min_lag; lag <= clamped_max_lag; ++lag) {
        double corr = 0.0;
        for (std::size_t i = lag; i < onset_curve.size(); ++i) {
            corr += onset_curve[i] * onset_curve[i - lag];
        }
        best = std::max(best, corr / zero_lag);
    }
    return clamp_unit(best);
}

double onset_interval_instability(const std::vector<double>& onset_curve,
                                  double threshold) {
    std::vector<std::size_t> peaks;
    for (std::size_t i = 1; i + 1 < onset_curve.size(); ++i) {
        if (onset_curve[i] >= threshold &&
            onset_curve[i] >= onset_curve[i - 1] &&
            onset_curve[i] >= onset_curve[i + 1]) {
            peaks.push_back(i);
        }
    }
    if (peaks.size() < 3) {
        return 1.0;
    }

    std::vector<double> intervals;
    intervals.reserve(peaks.size() - 1);
    for (std::size_t i = 1; i < peaks.size(); ++i) {
        intervals.push_back(static_cast<double>(peaks[i] - peaks[i - 1]));
    }

    const double mean =
        std::accumulate(intervals.begin(), intervals.end(), 0.0) / static_cast<double>(intervals.size());
    if (!(mean > 0.0)) {
        return 1.0;
    }

    double var = 0.0;
    for (double interval : intervals) {
        const double delta = interval - mean;
        var += delta * delta;
    }
    var /= static_cast<double>(intervals.size());
    return clamp_unit(std::sqrt(var) / mean);
}

} // namespace

double score_sparse_usability_window(const SparseUsabilityFeatures& features) {
    const double silence_ratio = clamp_unit(features.silence_ratio);
    const double onset_density = clamp_unit(features.onset_density);
    const double periodicity = clamp_unit(features.periodicity);
    const double transient_strength = clamp_unit(features.transient_strength);
    const double instability = clamp_unit(features.instability);

    const double positive =
        (0.30 * onset_density) +
        (0.40 * periodicity) +
        (0.30 * transient_strength);
    const double penalty =
        (0.20 * silence_ratio) +
        (0.35 * instability);

    return positive - penalty;
}

bool sparse_window_is_usable(const SparseUsabilityFeatures& features) {
    const double silence_ratio = clamp_unit(features.silence_ratio);
    const double periodicity = clamp_unit(features.periodicity);
    const double transient_strength = clamp_unit(features.transient_strength);
    const double instability = clamp_unit(features.instability);

    if (silence_ratio >= 0.98 && periodicity < 0.50) {
        return false;
    }

    return periodicity >= 0.35 &&
           transient_strength >= 0.25 &&
           instability <= 0.70;
}

SparseUsabilityWindow build_sparse_usability_window(double start_seconds,
                                                    double duration_seconds,
                                                    const SparseUsabilityFeatures& features) {
    SparseUsabilityWindow window;
    window.start_seconds = start_seconds;
    window.duration_seconds = duration_seconds;
    window.features = features;
    window.score = score_sparse_usability_window(features);
    window.usable = sparse_window_is_usable(features);
    return window;
}

SparseUsabilityFeatures measure_sparse_usability_features(const std::vector<float>& samples,
                                                          double sample_rate,
                                                          double min_bpm,
                                                          double max_bpm) {
    SparseUsabilityFeatures features;
    if (samples.empty() || !(sample_rate > 0.0) || !(min_bpm > 0.0) || !(max_bpm >= min_bpm)) {
        return features;
    }

    const std::size_t hop = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(sample_rate * 0.02)));
    const std::vector<double> envelope = build_envelope(samples, hop);
    if (envelope.size() < 8) {
        return features;
    }

    const auto envelope_peak_it = std::max_element(envelope.begin(), envelope.end());
    const double envelope_peak = (envelope_peak_it != envelope.end()) ? *envelope_peak_it : 0.0;
    const double silence_threshold = std::max(1e-4, envelope_peak * 0.08);
    const std::size_t silent_frames = static_cast<std::size_t>(std::count_if(
        envelope.begin(),
        envelope.end(),
        [silence_threshold](double value) { return value <= silence_threshold; }));
    features.silence_ratio =
        static_cast<double>(silent_frames) / static_cast<double>(envelope.size());

    const std::vector<double> onset_curve = build_onset_curve(envelope);
    const auto onset_peak_it = std::max_element(onset_curve.begin(), onset_curve.end());
    const double onset_peak = (onset_peak_it != onset_curve.end()) ? *onset_peak_it : 0.0;
    const double onset_threshold = std::max(1e-6, onset_peak * 0.25);

    std::size_t onset_count = 0;
    for (std::size_t i = 1; i + 1 < onset_curve.size(); ++i) {
        if (onset_curve[i] >= onset_threshold &&
            onset_curve[i] >= onset_curve[i - 1] &&
            onset_curve[i] >= onset_curve[i + 1]) {
            ++onset_count;
        }
    }

    const double duration_seconds = static_cast<double>(samples.size()) / sample_rate;
    const double onsets_per_second =
        duration_seconds > 0.0 ? static_cast<double>(onset_count) / duration_seconds : 0.0;
    features.onset_density = clamp_unit(onsets_per_second / 3.0);

    const double hop_seconds = static_cast<double>(hop) / sample_rate;
    const std::size_t min_lag = std::max<std::size_t>(1, static_cast<std::size_t>(
        std::floor((60.0 / max_bpm) / hop_seconds)));
    const std::size_t max_lag = std::max(min_lag, static_cast<std::size_t>(
        std::ceil((60.0 / min_bpm) / hop_seconds)));
    features.periodicity = normalized_autocorr_peak(onset_curve, min_lag, max_lag);

    const double mean_envelope =
        std::accumulate(envelope.begin(), envelope.end(), 0.0) / static_cast<double>(envelope.size());
    const double transient_ratio = onset_peak / std::max(1e-6, mean_envelope);
    features.transient_strength = clamp_unit(transient_ratio / 4.0);

    features.instability = onset_interval_instability(onset_curve, onset_threshold);
    return features;
}

std::size_t pick_sparse_usability_window(const std::vector<SparseUsabilityWindow>& windows,
                                         const SparseUsabilityPickRequest& request) {
    std::size_t best_index = windows.size();
    double best_adjusted_score = -std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < windows.size(); ++i) {
        const auto& window = windows[i];
        if (!window.usable || window.score < request.min_score) {
            continue;
        }

        const double distance = std::abs(window.start_seconds - request.target_seconds);
        if (distance > request.max_snap_seconds) {
            continue;
        }

        const double adjusted_score = window.score - window_distance_penalty(window, request);
        if (adjusted_score > best_adjusted_score) {
            best_adjusted_score = adjusted_score;
            best_index = i;
        }
    }

    return best_index;
}

std::vector<SparseUsabilitySpan> build_sparse_usability_spans(
    const std::vector<SparseUsabilityWindow>& windows,
    double min_score) {
    std::vector<SparseUsabilitySpan> spans;
    if (windows.empty()) {
        return spans;
    }

    bool in_span = false;
    SparseUsabilitySpan current;
    double score_sum = 0.0;
    std::size_t score_count = 0;

    for (std::size_t i = 0; i < windows.size(); ++i) {
        const auto& window = windows[i];
        const bool eligible = window.usable && window.score >= min_score;
        if (!eligible) {
            if (in_span) {
                current.mean_score = score_sum / static_cast<double>(score_count);
                spans.push_back(current);
                in_span = false;
                score_sum = 0.0;
                score_count = 0;
            }
            continue;
        }

        if (!in_span) {
            in_span = true;
            current = SparseUsabilitySpan{};
            current.start_seconds = window.start_seconds;
            current.first_index = i;
        }

        current.end_seconds = window.start_seconds + window.duration_seconds;
        current.last_index = i;
        score_sum += window.score;
        ++score_count;
    }

    if (in_span) {
        current.mean_score = score_sum / static_cast<double>(score_count);
        spans.push_back(current);
    }

    return spans;
}

std::vector<SparseUsabilityWindow> scan_sparse_usability_windows(
    const SparseUsabilityScanRequest& request) {
    std::vector<SparseUsabilityWindow> windows;
    if (!(request.total_duration_seconds > 0.0) ||
        !(request.window_duration_seconds > 0.0) ||
        !(request.hop_seconds > 0.0) ||
        !(request.sample_rate > 0.0) ||
        !(request.min_bpm > 0.0) ||
        !(request.max_bpm >= request.min_bpm) ||
        !request.provider) {
        return windows;
    }

    const auto starts = build_window_starts(request.total_duration_seconds,
                                            request.window_duration_seconds,
                                            request.hop_seconds);
    windows.reserve(starts.size());
    for (double start_seconds : starts) {
        std::vector<float> samples;
        if ((*request.provider)(start_seconds, request.window_duration_seconds, &samples) == 0 ||
            samples.empty()) {
            windows.push_back(build_sparse_usability_window(start_seconds,
                                                            request.window_duration_seconds,
                                                            SparseUsabilityFeatures{}));
            continue;
        }

        const auto features = measure_sparse_usability_features(samples,
                                                                request.sample_rate,
                                                                request.min_bpm,
                                                                request.max_bpm);
        windows.push_back(build_sparse_usability_window(start_seconds,
                                                        request.window_duration_seconds,
                                                        features));
    }
    return windows;
}

} // namespace detail
} // namespace beatit
