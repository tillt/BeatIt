//
//  bpm_activation.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/analysis.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace beatit {
namespace {

struct BpmEstimationContext {
    double fps = 0.0;
    double min_bpm = 0.0;
    double max_bpm = 0.0;
};

struct IntervalStats {
    double mean = 0.0;
    double stddev = 0.0;
    double median = 0.0;
};

bool make_bpm_estimation_context(const std::vector<float>& activation,
                                 const BeatitConfig& config,
                                 double sample_rate,
                                 BpmEstimationContext& context) {
    if (activation.size() < 8 || sample_rate <= 0.0 || config.hop_size <= 0) {
        return false;
    }

    context.fps =
        static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size);
    if (context.fps <= 0.0) {
        return false;
    }

    context.min_bpm = std::max(1.0, static_cast<double>(config.min_bpm));
    context.max_bpm = std::max(context.min_bpm + 1.0, static_cast<double>(config.max_bpm));
    return true;
}

IntervalStats compute_interval_stats(const std::vector<std::size_t>& peaks) {
    IntervalStats stats;
    if (peaks.size() < 2) {
        return stats;
    }

    std::vector<double> intervals;
    intervals.reserve(peaks.size() - 1);
    for (std::size_t i = 1; i < peaks.size(); ++i) {
        intervals.push_back(static_cast<double>(peaks[i] - peaks[i - 1]));
    }
    if (intervals.empty()) {
        return stats;
    }

    stats.mean = std::accumulate(intervals.begin(), intervals.end(), 0.0) /
                 static_cast<double>(intervals.size());

    double var = 0.0;
    for (double value : intervals) {
        const double delta = value - stats.mean;
        var += delta * delta;
    }
    stats.stddev = std::sqrt(var / static_cast<double>(intervals.size()));

    std::vector<double> tmp = intervals;
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
    stats.median = tmp[tmp.size() / 2];
    return stats;
}

} // namespace

float estimate_bpm_from_activation(const std::vector<float>& activation,
                                   const BeatitConfig& config,
                                   double sample_rate) {
    BpmEstimationContext context;
    if (!make_bpm_estimation_context(activation, config, sample_rate, context)) {
        return 0.0f;
    }

    const double min_interval = (60.0 * context.fps) / context.max_bpm;
    const double max_interval = (60.0 * context.fps) / context.min_bpm;
    const std::size_t min_sep = static_cast<std::size_t>(std::max(1.0, std::floor(min_interval)));

    float max_val = 0.0f;
    for (float v : activation) {
        max_val = std::max(max_val, v);
    }
    if (max_val <= 0.0f) {
        return 0.0f;
    }

    auto collect_peaks = [&](float threshold) {
        std::vector<std::size_t> peaks;
        peaks.reserve(activation.size() / 8);
        std::size_t last_peak = 0;
        bool has_last = false;
        for (std::size_t i = 1; i + 1 < activation.size(); ++i) {
            const float v = activation[i];
            if (v < threshold) {
                continue;
            }
            if (v < activation[i - 1] || v < activation[i + 1]) {
                continue;
            }
            if (!has_last || (i - last_peak) >= min_sep) {
                peaks.push_back(i);
                last_peak = i;
                has_last = true;
            } else if (v > activation[last_peak]) {
                peaks.back() = i;
                last_peak = i;
            }
        }
        return peaks;
    };

    float threshold = std::max(config.dbn_activation_floor, max_val * 0.25f);
    std::vector<std::size_t> peaks = collect_peaks(threshold);
    if (peaks.size() < 8) {
        threshold = std::max(config.dbn_activation_floor, max_val * 0.10f);
        peaks = collect_peaks(threshold);
    }
    if (peaks.size() < 4) {
        return 0.0f;
    }

    constexpr double kBpmBin = 0.05;
    const std::size_t bins =
        static_cast<std::size_t>(std::floor((context.max_bpm - context.min_bpm) / kBpmBin)) + 1;
    std::vector<double> hist(bins, 0.0);

    const std::size_t max_skip = 4;
    for (std::size_t i = 0; i < peaks.size(); ++i) {
        for (std::size_t k = 1; k <= max_skip && (i + k) < peaks.size(); ++k) {
            const std::size_t delta = peaks[i + k] - peaks[i];
            if (delta == 0) {
                continue;
            }
            const double per_beat = static_cast<double>(delta) / static_cast<double>(k);
            if (per_beat < min_interval || per_beat > max_interval) {
                continue;
            }
            const double bpm = (60.0 * context.fps) / per_beat;
            if (bpm < context.min_bpm || bpm > context.max_bpm) {
                continue;
            }
            const std::size_t idx =
                static_cast<std::size_t>(std::floor((bpm - context.min_bpm) / kBpmBin));
            const float weight = 0.5f * (activation[peaks[i]] + activation[peaks[i + k]]);
            hist[idx] += static_cast<double>(weight);
        }
    }

    const IntervalStats interval_stats = compute_interval_stats(peaks);
    const double bpm_mean =
        (interval_stats.mean > 0.0) ? (60.0 * context.fps / interval_stats.mean) : 0.0;
    const double bpm_median =
        (interval_stats.median > 0.0) ? (60.0 * context.fps / interval_stats.median) : 0.0;
    {
        auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
        debug_stream << "BPM debug: activation peaks=" << peaks.size()
                     << " threshold=" << threshold
                     << " bins=" << bins
                     << " interval_mean=" << interval_stats.mean
                     << " interval_median=" << interval_stats.median
                     << " interval_std=" << interval_stats.stddev
                     << " bpm_mean=" << bpm_mean
                     << " bpm_median=" << bpm_median;
    }
    BEATIT_LOG_DEBUG("BPM debug: peak_median_bpm=" << bpm_median);

    double best_score = 0.0;
    std::size_t best_idx = 0;
    for (std::size_t i = 0; i < bins; ++i) {
        const double bpm = context.min_bpm + static_cast<double>(i) * kBpmBin;
        double score = hist[i];
        const double half = bpm * 0.5;
        const double twice = bpm * 2.0;
        if (half >= context.min_bpm) {
            const std::size_t idx =
                static_cast<std::size_t>(std::floor((half - context.min_bpm) / kBpmBin));
            if (idx < bins) {
                score += 0.5 * hist[idx];
            }
        }
        if (twice <= context.max_bpm) {
            const std::size_t idx =
                static_cast<std::size_t>(std::floor((twice - context.min_bpm) / kBpmBin));
            if (idx < bins) {
                score += 0.5 * hist[idx];
            }
        }
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_score <= 0.0) {
        return 0.0f;
    }

    const std::size_t start = (best_idx > 2) ? (best_idx - 2) : 0;
    const std::size_t end = std::min(bins - 1, best_idx + 2);
    double weighted_sum = 0.0;
    double weight_total = 0.0;
    for (std::size_t i = start; i <= end; ++i) {
        const double bpm = context.min_bpm + static_cast<double>(i) * kBpmBin;
        const double weight = hist[i];
        weighted_sum += bpm * weight;
        weight_total += weight;
    }
    const double refined = (weight_total > 0.0)
        ? (weighted_sum / weight_total)
        : (context.min_bpm + static_cast<double>(best_idx) * kBpmBin);
    {
        auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
        debug_stream << "BPM debug: activation_bpm=" << refined
                     << " best_idx=" << best_idx
                     << " score=" << best_score;
    }

    return static_cast<float>(refined);
}

float estimate_bpm_from_activation_autocorr(const std::vector<float>& activation,
                                            const BeatitConfig& config,
                                            double sample_rate) {
    BpmEstimationContext context;
    if (!make_bpm_estimation_context(activation, config, sample_rate, context)) {
        return 0.0f;
    }

    const std::size_t target_frames = 3000;
    const std::size_t stride =
        std::max<std::size_t>(1, activation.size() / target_frames);

    std::vector<double> series;
    series.reserve((activation.size() + stride - 1) / stride);
    double mean = 0.0;
    for (std::size_t i = 0; i < activation.size(); i += stride) {
        const double v = activation[i];
        series.push_back(v);
        mean += v;
    }
    if (series.size() < 8) {
        return 0.0f;
    }
    mean /= static_cast<double>(series.size());
    for (double& v : series) {
        v -= mean;
    }

    const double fps_decim = context.fps / static_cast<double>(stride);
    const double min_lag = (60.0 * fps_decim) / context.max_bpm;
    const double max_lag = (60.0 * fps_decim) / context.min_bpm;
    const std::size_t lag_min = static_cast<std::size_t>(std::max(1.0, std::floor(min_lag)));
    const std::size_t lag_max =
        static_cast<std::size_t>(std::min<double>(series.size() - 1, std::ceil(max_lag)));
    if (lag_max <= lag_min) {
        return 0.0f;
    }

    std::vector<double> corr(lag_max + 1, 0.0);
    for (std::size_t lag = lag_min; lag <= lag_max; ++lag) {
        double sum = 0.0;
        for (std::size_t i = 0; i + lag < series.size(); ++i) {
            sum += series[i] * series[i + lag];
        }
        corr[lag] = sum;
    }

    double best_score = 0.0;
    std::size_t best_lag = lag_min;
    for (std::size_t lag = lag_min; lag <= lag_max; ++lag) {
        double score = corr[lag];
        const std::size_t half = lag / 2;
        const std::size_t twice = lag * 2;
        if (half >= lag_min) {
            score += 0.5 * corr[half];
        }
        if (twice <= lag_max) {
            score += 0.5 * corr[twice];
        }
        if (score > best_score) {
            best_score = score;
            best_lag = lag;
        }
    }

    if (best_score <= 0.0) {
        return 0.0f;
    }

    double refined_lag = static_cast<double>(best_lag);
    if (best_lag > lag_min && best_lag + 1 <= lag_max) {
        const double y0 = corr[best_lag - 1];
        const double y1 = corr[best_lag];
        const double y2 = corr[best_lag + 1];
        const double denom = (y0 - 2.0 * y1 + y2);
        if (std::abs(denom) > 1e-9) {
            const double delta = 0.5 * (y0 - y2) / denom;
            refined_lag = static_cast<double>(best_lag) + delta;
        }
    }

    const double bpm = (60.0 * fps_decim) / refined_lag;
    auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
    debug_stream << "BPM debug: autocorr frames=" << series.size()
                 << " stride=" << stride
                 << " lag=" << refined_lag
                 << " bpm=" << bpm
                 << " score=" << best_score;
    return static_cast<float>(bpm);
}

float estimate_bpm_from_activation_comb(const std::vector<float>& activation,
                                        const BeatitConfig& config,
                                        double sample_rate) {
    BpmEstimationContext context;
    if (!make_bpm_estimation_context(activation, config, sample_rate, context)) {
        return 0.0f;
    }

    const std::size_t min_lag =
        static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * context.fps) / context.max_bpm)));
    const std::size_t max_lag =
        static_cast<std::size_t>(std::max(min_lag + 1.0,
                                          std::ceil((60.0 * context.fps) / context.min_bpm)));
    if (max_lag >= activation.size()) {
        return 0.0f;
    }

    std::vector<double> series;
    series.reserve(activation.size());
    double mean = 0.0;
    for (float v : activation) {
        series.push_back(static_cast<double>(v));
        mean += v;
    }
    mean /= static_cast<double>(series.size());
    for (double& v : series) {
        v = std::max(0.0, v - mean);
    }

    std::vector<double> ac(max_lag + 1, 0.0);
    for (std::size_t lag = min_lag; lag <= max_lag; ++lag) {
        double sum = 0.0;
        for (std::size_t i = 0; i + lag < series.size(); ++i) {
            sum += series[i] * series[i + lag];
        }
        ac[lag] = sum;
    }

    double best_score = 0.0;
    std::size_t best_lag = min_lag;
    std::vector<double> comb_scores(max_lag + 1, 0.0);
    for (std::size_t lag = min_lag; lag <= max_lag; ++lag) {
        double score = 0.0;
        for (std::size_t k = 1; k <= 4; ++k) {
            const std::size_t idx = lag * k;
            if (idx > max_lag) {
                break;
            }
            score += ac[idx] / static_cast<double>(k);
        }
        comb_scores[lag] = score;
        if (score > best_score) {
            best_score = score;
            best_lag = lag;
        }
    }
    if (best_score <= 0.0) {
        return 0.0f;
    }

    const double bpm = (60.0 * context.fps) / static_cast<double>(best_lag);
    {
        auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
        debug_stream << "BPM debug: comb_bpm=" << bpm
                     << " comb_lag=" << best_lag
                     << " comb_score=" << best_score;
    }

    return static_cast<float>(bpm);
}

} // namespace beatit
