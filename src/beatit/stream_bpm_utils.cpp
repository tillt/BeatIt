//
//  stream_bpm_utils.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#include "beatit/stream_bpm_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace beatit {
namespace detail {

float estimate_bpm_from_beats_local(const std::vector<unsigned long long>& beat_samples,
                                    double sample_rate) {
    if (beat_samples.size() < 2 || sample_rate <= 0.0) {
        return 0.0f;
    }

    const bool debug_bpm = std::getenv("BEATIT_DEBUG_BPM") != nullptr;

    std::vector<double> intervals;
    intervals.reserve(beat_samples.size() - 1);
    for (std::size_t i = 1; i < beat_samples.size(); ++i) {
        const unsigned long long prev = beat_samples[i - 1];
        const unsigned long long next = beat_samples[i];
        if (next > prev) {
            const double interval = static_cast<double>(next - prev) / sample_rate;
            if (interval > 0.0) {
                intervals.push_back(interval);
            }
        }
    }

    std::vector<double> bar_intervals;
    if (beat_samples.size() >= 5) {
        bar_intervals.reserve(beat_samples.size() - 4);
        for (std::size_t i = 4; i < beat_samples.size(); ++i) {
            const unsigned long long prev = beat_samples[i - 4];
            const unsigned long long next = beat_samples[i];
            if (next > prev) {
                const double interval = static_cast<double>(next - prev) / sample_rate;
                if (interval > 0.0) {
                    bar_intervals.push_back(interval);
                }
            }
        }
    }

    if (intervals.empty()) {
        return 0.0f;
    }

    double beat_median = 0.0;
    if (debug_bpm) {
        std::vector<double> tmp = intervals;
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        beat_median = tmp[tmp.size() / 2];
    }

    if (bar_intervals.size() >= 3) {
        std::sort(bar_intervals.begin(), bar_intervals.end());
        const double median = bar_intervals[bar_intervals.size() / 2];
        if (median > 0.0) {
            if (debug_bpm) {
                std::cerr << "BPM debug: bar_median=" << median
                          << " bpm=" << (240.0 / median)
                          << " beat_median=" << beat_median
                          << " beat_bpm=" << (beat_median > 0.0 ? (60.0 / beat_median) : 0.0)
                          << " bars=" << bar_intervals.size()
                          << " beats=" << intervals.size()
                          << " sample_rate=" << sample_rate << "\n";
            }
            return static_cast<float>(240.0 / median);
        }
    }

    std::sort(intervals.begin(), intervals.end());
    const std::size_t count = intervals.size();
    std::size_t trim = 0;
    if (count > 20) {
        trim = count / 10;
    }
    const std::size_t start = trim;
    const std::size_t end = (count > trim) ? (count - trim) : count;
    if (end <= start) {
        const double median = intervals[count / 2];
        if (debug_bpm) {
            std::cerr << "BPM debug: beat_median=" << median
                      << " bpm=" << (60.0 / median)
                      << " beats=" << intervals.size()
                      << " sample_rate=" << sample_rate << "\n";
        }
        return median > 0.0 ? static_cast<float>(60.0 / median) : 0.0f;
    }
    double sum = 0.0;
    for (std::size_t i = start; i < end; ++i) {
        sum += intervals[i];
    }
    const double avg = sum / static_cast<double>(end - start);
    if (debug_bpm) {
        std::cerr << "BPM debug: beat_trimmed_mean=" << avg
                  << " bpm=" << (avg > 0.0 ? (60.0 / avg) : 0.0)
                  << " trim=" << trim
                  << " beats=" << intervals.size()
                  << " sample_rate=" << sample_rate << "\n";
    }
    if (avg <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(60.0 / avg);
}

float estimate_bpm_from_activation_peaks_local(const std::vector<float>& activation,
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
        std::cerr << "BPM debug: activation_bpm=" << refined
                  << " best_idx=" << best_idx
                  << " score=" << best_score << "\n";
    }

    return static_cast<float>(refined);
}

float estimate_bpm_from_activation_autocorr_local(const std::vector<float>& activation,
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

float normalize_bpm_to_range_local(float bpm, float min_bpm, float max_bpm) {
    if (!(bpm > 0.0f)) {
        return bpm;
    }
    const float lo = std::max(1.0f, min_bpm);
    const float hi = std::max(lo + 1.0f, max_bpm);
    while (bpm < lo && (bpm * 2.0f) <= hi) {
        bpm *= 2.0f;
    }
    while (bpm > hi && (bpm * 0.5f) >= lo) {
        bpm *= 0.5f;
    }
    if (bpm < lo) {
        bpm = lo;
    } else if (bpm > hi) {
        bpm = hi;
    }
    return bpm;
}

} // namespace detail
} // namespace beatit
