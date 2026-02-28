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
#include <initializer_list>
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

std::size_t pick_required_window(const std::vector<SparseUsabilityWindow>& windows,
                                 double target_seconds,
                                 double min_start_seconds,
                                 double max_start_seconds,
                                 double max_snap_seconds,
                                 double min_score,
                                 double distance_weight,
                                 std::initializer_list<std::size_t> excluded_indices = {}) {
    std::vector<SparseUsabilityWindow> filtered;
    filtered.reserve(windows.size());
    std::vector<std::size_t> index_map;
    index_map.reserve(windows.size());
    for (std::size_t i = 0; i < windows.size(); ++i) {
        if (std::find(excluded_indices.begin(), excluded_indices.end(), i) != excluded_indices.end()) {
            continue;
        }
        if (windows[i].start_seconds < min_start_seconds ||
            windows[i].start_seconds > max_start_seconds) {
            continue;
        }
        filtered.push_back(windows[i]);
        index_map.push_back(i);
    }
    if (filtered.empty()) {
        return windows.size();
    }

    const std::size_t filtered_index =
        pick_sparse_usability_window(filtered,
                                     SparseUsabilityPickRequest{
                                         target_seconds,
                                         max_snap_seconds,
                                         distance_weight,
                                         min_score
                                     });
    return filtered_index < index_map.size() ? index_map[filtered_index] : windows.size();
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

std::size_t find_covering_sparse_usability_window(const std::vector<SparseUsabilityWindow>& windows,
                                                  double target_seconds) {
    std::size_t best_index = windows.size();
    double best_score = -std::numeric_limits<double>::infinity();
    double best_center_distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < windows.size(); ++i) {
        const double start_seconds = windows[i].start_seconds;
        const double end_seconds = start_seconds + windows[i].duration_seconds;
        if (target_seconds < start_seconds || target_seconds >= end_seconds) {
            continue;
        }

        const double center_seconds = start_seconds + (0.5 * windows[i].duration_seconds);
        const double center_distance = std::abs(target_seconds - center_seconds);
        if (best_index == windows.size() ||
            windows[i].score > best_score + 1e-9 ||
            (std::abs(windows[i].score - best_score) <= 1e-9 &&
             center_distance < best_center_distance)) {
            best_index = i;
            best_score = windows[i].score;
            best_center_distance = center_distance;
        }
    }
    return best_index;
}

SparseUsabilityTargets pick_sparse_usability_targets(
    const std::vector<SparseUsabilityWindow>& windows,
    double min_score) {
    SparseUsabilityTargets targets;
    if (windows.empty()) {
        return targets;
    }

    double start_seconds = windows.front().start_seconds;
    double end_seconds = windows.front().start_seconds + windows.front().duration_seconds;
    for (const auto& window : windows) {
        start_seconds = std::min(start_seconds, window.start_seconds);
        end_seconds = std::max(end_seconds, window.start_seconds + window.duration_seconds);
    }
    const double total_duration_seconds = std::max(0.0, end_seconds - start_seconds);
    if (!(total_duration_seconds > 0.0)) {
        return targets;
    }

    const double left_target = start_seconds;
    const double right_target = end_seconds;
    const double middle_target = start_seconds + (0.5 * total_duration_seconds);
    const double between_target = start_seconds + (0.25 * total_duration_seconds);
    const double max_snap_seconds = total_duration_seconds;
    constexpr double kDistanceWeight = 0.01;

    targets.left_index = pick_required_window(windows,
                                              left_target,
                                              start_seconds,
                                              middle_target,
                                              max_snap_seconds,
                                              min_score,
                                              kDistanceWeight);
    targets.right_index = pick_required_window(windows,
                                               right_target,
                                               middle_target,
                                               end_seconds,
                                               max_snap_seconds,
                                               min_score,
                                               kDistanceWeight);
    targets.middle_index = pick_required_window(windows,
                                                middle_target,
                                                between_target,
                                                right_target,
                                                max_snap_seconds,
                                                min_score,
                                                kDistanceWeight,
                                                {targets.left_index, targets.right_index});
    targets.between_index = pick_required_window(windows,
                                                 between_target,
                                                 start_seconds,
                                                 middle_target,
                                                 max_snap_seconds,
                                                 min_score,
                                                 kDistanceWeight,
                                                 {targets.middle_index, targets.right_index});

    if (targets.between_index == windows.size()) {
        targets.between_index = pick_required_window(windows,
                                                     between_target,
                                                     start_seconds,
                                                     middle_target,
                                                     max_snap_seconds,
                                                     min_score,
                                                     kDistanceWeight);
    }
    return targets;
}

SparseInteriorWindowTargets resolve_sparse_interior_targets(
    const std::vector<SparseUsabilityWindow>& windows,
    double current_middle_start_seconds,
    double current_between_start_seconds,
    double min_score) {
    SparseInteriorWindowTargets targets;
    targets.middle_start_seconds = current_middle_start_seconds;
    targets.between_start_seconds = current_between_start_seconds;
    if (windows.empty()) {
        return targets;
    }

    double start_seconds = windows.front().start_seconds;
    double end_seconds = windows.front().start_seconds + windows.front().duration_seconds;
    for (const auto& window : windows) {
        start_seconds = std::min(start_seconds, window.start_seconds);
        end_seconds = std::max(end_seconds, window.start_seconds + window.duration_seconds);
    }
    const SparseUsabilityTargets suggested_targets =
        pick_sparse_usability_targets(windows, min_score);
    const double max_snap_seconds = std::max(0.0, end_seconds - start_seconds);
    const auto is_current_usable = [&](double current_start_seconds) {
        const std::size_t index =
            find_covering_sparse_usability_window(windows, current_start_seconds);
        return index < windows.size() && windows[index].usable;
    };
    const auto start_for_index = [&](std::size_t index, double fallback_start) {
        if (index >= windows.size()) {
            return fallback_start;
        }
        return windows[index].start_seconds;
    };
    const auto pick_replacement_start = [&](double target_start_seconds,
                                            double fallback_start,
                                            std::initializer_list<std::size_t> excluded_indices = {}) {
        const std::size_t index = pick_required_window(windows,
                                                       target_start_seconds,
                                                       start_seconds,
                                                       end_seconds,
                                                       max_snap_seconds,
                                                       min_score,
                                                       0.01,
                                                       excluded_indices);
        if (index >= windows.size()) {
            return fallback_start;
        }
        return windows[index].start_seconds;
    };

    if (!is_current_usable(current_middle_start_seconds)) {
        const double suggested_middle_start =
            start_for_index(suggested_targets.middle_index, current_middle_start_seconds);
        const double middle_start =
            pick_replacement_start(suggested_middle_start, current_middle_start_seconds);
        targets.middle_overridden = middle_start != current_middle_start_seconds;
        targets.middle_start_seconds = middle_start;
    }

    const std::size_t middle_index =
        find_covering_sparse_usability_window(windows, targets.middle_start_seconds);
    const bool between_usable = is_current_usable(current_between_start_seconds);
    if (targets.middle_overridden || !between_usable) {
        const double suggested_between_start =
            start_for_index(suggested_targets.between_index, current_between_start_seconds);
        double between_start =
            pick_replacement_start(suggested_between_start,
                                   targets.middle_start_seconds,
                                   {middle_index});
        if (between_start == targets.middle_start_seconds && between_usable &&
            current_between_start_seconds != targets.middle_start_seconds) {
            between_start = current_between_start_seconds;
        }
        targets.between_overridden =
            between_start != current_between_start_seconds ||
            between_start == targets.middle_start_seconds;
        targets.between_start_seconds = between_start;
    }

    return targets;
}

} // namespace detail
} // namespace beatit
