//
//  analysis_bpm_activation.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/analysis.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace beatit {

float estimate_bpm_from_activation(const std::vector<float>& activation,
                                   const CoreMLConfig& config,
                                   double sample_rate) {
    if (activation.size() < 8 || sample_rate <= 0.0 || config.hop_size <= 0) {
        return 0.0f;
    }

    const bool debug_bpm = std::getenv("BEATIT_DEBUG_BPM") != nullptr;
    const double fps =
        static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size);
    if (fps <= 0.0) {
        return 0.0f;
    }

    const double min_bpm = std::max(1.0, static_cast<double>(config.min_bpm));
    const double max_bpm = std::max(min_bpm + 1.0, static_cast<double>(config.max_bpm));
    const double min_interval = (60.0 * fps) / max_bpm;
    const double max_interval = (60.0 * fps) / min_bpm;
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
        static_cast<std::size_t>(std::floor((max_bpm - min_bpm) / kBpmBin)) + 1;
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
            const double bpm = (60.0 * fps) / per_beat;
            if (bpm < min_bpm || bpm > max_bpm) {
                continue;
            }
            const std::size_t idx =
                static_cast<std::size_t>(std::floor((bpm - min_bpm) / kBpmBin));
            const float weight = 0.5f * (activation[peaks[i]] + activation[peaks[i + k]]);
            hist[idx] += static_cast<double>(weight);
        }
    }

    if (debug_bpm) {
        std::vector<double> intervals;
        intervals.reserve(peaks.size() > 1 ? (peaks.size() - 1) : 0);
        for (std::size_t i = 1; i < peaks.size(); ++i) {
            intervals.push_back(static_cast<double>(peaks[i] - peaks[i - 1]));
        }
        double mean = 0.0;
        double stddev = 0.0;
        double median = 0.0;
        if (!intervals.empty()) {
            double sum = 0.0;
            for (double v : intervals) {
                sum += v;
            }
            mean = sum / static_cast<double>(intervals.size());
            double var = 0.0;
            for (double v : intervals) {
                const double d = v - mean;
                var += d * d;
            }
            stddev = std::sqrt(var / static_cast<double>(intervals.size()));
            std::vector<double> tmp = intervals;
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
            median = tmp[tmp.size() / 2];
        }
        const double bpm_mean = (mean > 0.0) ? (60.0 * fps / mean) : 0.0;
        const double bpm_median = (median > 0.0) ? (60.0 * fps / median) : 0.0;
        std::cerr << "BPM debug: activation peaks=" << peaks.size()
                  << " threshold=" << threshold
                  << " bins=" << bins
                  << " interval_mean=" << mean
                  << " interval_median=" << median
                  << " interval_std=" << stddev
                  << " bpm_mean=" << bpm_mean
                  << " bpm_median=" << bpm_median
                  << "\n";
        std::cerr << "BPM debug: peak_median_bpm=" << bpm_median << "\n";
    }

    double best_score = 0.0;
    std::size_t best_idx = 0;
    for (std::size_t i = 0; i < bins; ++i) {
        const double bpm = min_bpm + static_cast<double>(i) * kBpmBin;
        double score = hist[i];
        const double half = bpm * 0.5;
        const double twice = bpm * 2.0;
        if (half >= min_bpm) {
            const std::size_t idx =
                static_cast<std::size_t>(std::floor((half - min_bpm) / kBpmBin));
            if (idx < bins) {
                score += 0.5 * hist[idx];
            }
        }
        if (twice <= max_bpm) {
            const std::size_t idx =
                static_cast<std::size_t>(std::floor((twice - min_bpm) / kBpmBin));
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
        const double bpm = min_bpm + static_cast<double>(i) * kBpmBin;
        const double weight = hist[i];
        weighted_sum += bpm * weight;
        weight_total += weight;
    }
    const double refined = (weight_total > 0.0)
        ? (weighted_sum / weight_total)
        : (min_bpm + static_cast<double>(best_idx) * kBpmBin);

    if (debug_bpm) {
        std::vector<std::size_t> top_idx(hist.size());
        for (std::size_t i = 0; i < top_idx.size(); ++i) {
            top_idx[i] = i;
        }
        const std::size_t top_count = std::min<std::size_t>(5, top_idx.size());
        std::partial_sort(
            top_idx.begin(),
            top_idx.begin() + top_count,
            top_idx.end(),
            [&](std::size_t a, std::size_t b) { return hist[a] > hist[b]; }
        );
        std::ostringstream oss;
        double top_sum = 0.0;
        double top1_weight = 0.0;
        double top2_weight = 0.0;
        double top1_bpm = 0.0;
        double top2_bpm = 0.0;
        double top_range = 0.0;
        for (std::size_t i = 0; i < top_count; ++i) {
            const std::size_t idx = top_idx[i];
            const double bin_bpm = min_bpm + static_cast<double>(idx) * kBpmBin;
            if (i != 0) {
                oss << " ";
            }
            oss << bin_bpm << "(" << hist[idx] << ")";
            top_sum += hist[idx];
            if (i == 0) {
                top1_weight = hist[idx];
                top1_bpm = bin_bpm;
            } else if (i == 1) {
                top2_weight = hist[idx];
                top2_bpm = bin_bpm;
            }
            if (i == top_count - 1) {
                top_range = std::abs(bin_bpm - top1_bpm);
            }
        }
        std::cerr << "BPM debug: peak_hist_top=" << oss.str() << "\n";
        const double top_ratio = (top_sum > 0.0) ? (top1_weight / top_sum) : 0.0;
        const double top_gap = top1_weight - top2_weight;
        std::cerr << "BPM debug: peak_hist_dom "
                  << "top1_bpm=" << top1_bpm
                  << " top1_w=" << top1_weight
                  << " top2_bpm=" << top2_bpm
                  << " top2_w=" << top2_weight
                  << " top1_ratio=" << top_ratio
                  << " top_gap=" << top_gap
                  << " top_range_bpm=" << top_range
                  << "\n";
        std::cerr << "BPM debug: activation_bpm=" << refined
                  << " best_idx=" << best_idx
                  << " score=" << best_score << "\n";
    }

    return static_cast<float>(refined);
}

float estimate_bpm_from_activation_autocorr(const std::vector<float>& activation,
                                            const CoreMLConfig& config,
                                            double sample_rate) {
    if (activation.size() < 8 || sample_rate <= 0.0 || config.hop_size <= 0) {
        return 0.0f;
    }

    const bool debug_bpm = std::getenv("BEATIT_DEBUG_BPM") != nullptr;
    const double fps =
        static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size);
    if (fps <= 0.0) {
        return 0.0f;
    }

    const double min_bpm = std::max(1.0, static_cast<double>(config.min_bpm));
    const double max_bpm = std::max(min_bpm + 1.0, static_cast<double>(config.max_bpm));
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

    const double fps_decim = fps / static_cast<double>(stride);
    const double min_lag = (60.0 * fps_decim) / max_bpm;
    const double max_lag = (60.0 * fps_decim) / min_bpm;
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
    if (debug_bpm) {
        std::cerr << "BPM debug: autocorr frames=" << series.size()
                  << " stride=" << stride
                  << " lag=" << refined_lag
                  << " bpm=" << bpm
                  << " score=" << best_score
                  << "\n";
    }
    return static_cast<float>(bpm);
}

float estimate_bpm_from_activation_comb(const std::vector<float>& activation,
                                        const CoreMLConfig& config,
                                        double sample_rate) {
    if (activation.size() < 8 || sample_rate <= 0.0 || config.hop_size <= 0) {
        return 0.0f;
    }

    const bool debug_bpm = std::getenv("BEATIT_DEBUG_BPM") != nullptr;
    const double fps =
        static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size);
    if (fps <= 0.0) {
        return 0.0f;
    }

    const double min_bpm = std::max(1.0, static_cast<double>(config.min_bpm));
    const double max_bpm = std::max(min_bpm + 1.0, static_cast<double>(config.max_bpm));
    const std::size_t min_lag =
        static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm)));
    const std::size_t max_lag =
        static_cast<std::size_t>(std::max(min_lag + 1.0, std::ceil((60.0 * fps) / min_bpm)));
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

    const double bpm = (60.0 * fps) / static_cast<double>(best_lag);
    if (debug_bpm) {
        std::vector<std::size_t> top_idx;
        top_idx.reserve(max_lag - min_lag + 1);
        for (std::size_t lag = min_lag; lag <= max_lag; ++lag) {
            top_idx.push_back(lag);
        }
        const std::size_t top_count = std::min<std::size_t>(3, top_idx.size());
        std::partial_sort(
            top_idx.begin(),
            top_idx.begin() + top_count,
            top_idx.end(),
            [&](std::size_t a, std::size_t b) { return comb_scores[a] > comb_scores[b]; }
        );
        const std::size_t top1 = top_idx[0];
        const std::size_t top2 = (top_count > 1) ? top_idx[1] : top1;
        const double top1_bpm = (60.0 * fps) / static_cast<double>(top1);
        const double top2_bpm = (60.0 * fps) / static_cast<double>(top2);
        const double top1_score = comb_scores[top1];
        const double top2_score = comb_scores[top2];
        double top_sum = 0.0;
        for (std::size_t i = 0; i < top_count; ++i) {
            top_sum += comb_scores[top_idx[i]];
        }
        const double top_ratio = (top_sum > 0.0) ? (top1_score / top_sum) : 0.0;
        const double top_gap = top1_score - top2_score;
        std::ostringstream oss;
        for (std::size_t i = 0; i < top_count; ++i) {
            const std::size_t lag = top_idx[i];
            const double bin_bpm = (60.0 * fps) / static_cast<double>(lag);
            if (i != 0) {
                oss << " ";
            }
            oss << bin_bpm << "(" << comb_scores[lag] << ")";
        }
        std::cerr << "BPM debug: comb_bpm=" << bpm
                  << " comb_lag=" << best_lag
                  << " comb_score=" << best_score
                  << "\n";
        std::cerr << "BPM debug: comb_top=" << oss.str() << "\n";
        std::cerr << "BPM debug: comb_dom "
                  << "top1_bpm=" << top1_bpm
                  << " top1_w=" << top1_score
                  << " top2_bpm=" << top2_bpm
                  << " top2_w=" << top2_score
                  << " top1_ratio=" << top_ratio
                  << " top_gap=" << top_gap
                  << "\n";
    }

    return static_cast<float>(bpm);
}

} // namespace beatit
