//
//  analysis.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/analysis.h"
#include "beatit/coreml.h"
#if defined(BEATIT_USE_TORCH)
#include "beatit/torch_mel.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(BEATIT_USE_TORCH)
#include <c10/core/InferenceMode.h>
#include <torch/script.h>
#endif

namespace beatit {

float estimate_bpm_from_beats(const std::vector<unsigned long long>& beat_samples,
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

namespace {

std::string shell_escape(const std::string& value) {
    std::string escaped = "'";
    for (char ch : value) {
        if (ch == '\'') {
            escaped += "'\\''";
        } else {
            escaped += ch;
        }
    }
    escaped += "'";
    return escaped;
}

bool write_wav_mono_16(const std::string& path,
                       const std::vector<float>& samples,
                       double sample_rate,
                       std::string* error) {
    if (samples.empty() || sample_rate <= 0.0) {
        if (error) {
            *error = "Empty samples or invalid sample rate.";
        }
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        if (error) {
            *error = "Failed to open WAV output.";
        }
        return false;
    }

    const std::uint16_t channels = 1;
    const std::uint16_t bits_per_sample = 16;
    const std::uint32_t sample_rate_u = static_cast<std::uint32_t>(std::lround(sample_rate));
    const std::uint32_t byte_rate = sample_rate_u * channels * (bits_per_sample / 8);
    const std::uint16_t block_align = channels * (bits_per_sample / 8);
    const std::uint32_t data_size = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    const std::uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riff_size), sizeof(riff_size));
    out.write("WAVE", 4);

    out.write("fmt ", 4);
    const std::uint32_t fmt_size = 16;
    const std::uint16_t audio_format = 1;
    out.write(reinterpret_cast<const char*>(&fmt_size), sizeof(fmt_size));
    out.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
    out.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    out.write(reinterpret_cast<const char*>(&sample_rate_u), sizeof(sample_rate_u));
    out.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    out.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
    out.write(reinterpret_cast<const char*>(&bits_per_sample), sizeof(bits_per_sample));

    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

    for (float sample : samples) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const std::int16_t value = static_cast<std::int16_t>(std::lround(clamped * 32767.0f));
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    if (!out.good()) {
        if (error) {
            *error = "Failed to write WAV data.";
        }
        return false;
    }

    return true;
}

bool parse_beatthis_output(const std::string& output,
                           std::vector<double>* beats,
                           std::vector<double>* downbeats,
                           std::string* error) {
    if (!beats || !downbeats) {
        if (error) {
            *error = "Invalid output buffers.";
        }
        return false;
    }

    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream line_stream(line);
        std::string label;
        line_stream >> label;
        if (label != "beats" && label != "downbeats") {
            continue;
        }
        std::size_t count = 0;
        line_stream >> count;
        std::vector<double>& target = (label == "beats") ? *beats : *downbeats;
        target.clear();
        target.reserve(count);
        double value = 0.0;
        while (line_stream >> value) {
            target.push_back(value);
        }
    }

    return true;
}

bool run_beatthis_external(const std::vector<float>& samples,
                           double sample_rate,
                           const CoreMLConfig& config,
                           std::vector<double>* beats,
                           std::vector<double>* downbeats,
                           std::string* error) {
    if (!beats || !downbeats) {
        if (error) {
            *error = "Invalid output buffers.";
        }
        return false;
    }
    if (config.beatthis_script.empty() || config.beatthis_checkpoint.empty()) {
        if (error) {
            *error = "BeatThis script or checkpoint path missing.";
        }
        return false;
    }

    const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
    const std::string tmp_name =
        "beatthis_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".wav";
    const std::filesystem::path tmp_path = tmp_dir / tmp_name;

    std::string wav_error;
    if (!write_wav_mono_16(tmp_path.string(), samples, sample_rate, &wav_error)) {
        if (error) {
            *error = wav_error;
        }
        return false;
    }

    std::ostringstream command;
    command << shell_escape(config.beatthis_python)
            << " "
            << shell_escape(config.beatthis_script)
            << " --input "
            << shell_escape(tmp_path.string())
            << " --checkpoint "
            << shell_escape(config.beatthis_checkpoint)
            << " --device cpu";
    if (config.beatthis_use_dbn) {
        command << " --dbn";
    }

    FILE* pipe = popen(command.str().c_str(), "r");
    if (!pipe) {
        if (error) {
            *error = "Failed to launch BeatThis subprocess.";
        }
        return false;
    }

    std::string output;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }
    const int status = pclose(pipe);
    std::error_code remove_error;
    std::filesystem::remove(tmp_path, remove_error);

    if (status != 0) {
        if (error) {
            *error = "BeatThis subprocess failed.";
        }
        return false;
    }

    if (!parse_beatthis_output(output, beats, downbeats, error)) {
        return false;
    }

    return true;
}

std::vector<float> resample_linear(const std::vector<float>& input,
                                   double input_rate,
                                   std::size_t target_rate) {
    if (input_rate <= 0.0 || target_rate == 0 || input.empty()) {
        return {};
    }
    if (static_cast<std::size_t>(std::lround(input_rate)) == target_rate) {
        return input;
    }

    const double ratio = static_cast<double>(target_rate) / input_rate;
    const std::size_t output_size = static_cast<std::size_t>(std::lround(input.size() * ratio));
    std::vector<float> output(output_size, 0.0f);

    for (std::size_t i = 0; i < output_size; ++i) {
        const double position = static_cast<double>(i) / ratio;
        const std::size_t index = static_cast<std::size_t>(position);
        const double frac = position - static_cast<double>(index);
        if (index + 1 < input.size()) {
            const float a = input[index];
            const float b = input[index + 1];
            output[i] = static_cast<float>((1.0 - frac) * a + frac * b);
        } else if (index < input.size()) {
            output[i] = input[index];
        }
    }

    return output;
}

std::vector<float> compute_phase_energy(const std::vector<float>& samples,
                                        double sample_rate,
                                        const CoreMLConfig& config) {
    if (samples.empty() || sample_rate <= 0.0 || config.sample_rate == 0 || config.hop_size == 0) {
        return {};
    }

    std::vector<float> resampled = resample_linear(samples, sample_rate, config.sample_rate);
    if (resampled.empty()) {
        return {};
    }

    const double cutoff_hz = 150.0;
    const double dt = 1.0 / static_cast<double>(config.sample_rate);
    const double rc = 1.0 / (2.0 * 3.141592653589793 * cutoff_hz);
    const double alpha = dt / (rc + dt);

    double state = 0.0;
    double sum_sq = 0.0;
    std::size_t count = 0;
    std::vector<float> energy;
    energy.reserve(resampled.size() / config.hop_size + 1);

    for (float sample : resampled) {
        state += alpha * (static_cast<double>(sample) - state);
        sum_sq += state * state;
        count++;
        if (count >= config.hop_size) {
            const double rms = std::sqrt(sum_sq / static_cast<double>(count));
            energy.push_back(static_cast<float>(rms));
            sum_sq = 0.0;
            count = 0;
        }
    }

    return energy;
}

AnalysisResult analyze_with_beatthis(const std::vector<float>& samples,
                                     double sample_rate,
                                     const CoreMLConfig& config) {
    AnalysisResult result;
    std::vector<double> beat_times;
    std::vector<double> downbeat_times;
    std::string error;

    if (!run_beatthis_external(samples,
                               sample_rate,
                               config,
                               &beat_times,
                               &downbeat_times,
                               &error)) {
        if (config.verbose) {
            std::cerr << "BeatThis failed: " << error << "\n";
        }
        return result;
    }

    const double duration = static_cast<double>(samples.size()) / sample_rate;
    const float fallback_fps = config.hop_size > 0
        ? static_cast<float>(config.sample_rate) / static_cast<float>(config.hop_size)
        : 100.0f;
    const float fps = config.beatthis_fps > 0.0f ? config.beatthis_fps : fallback_fps;
    const std::size_t total_frames =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(duration * fps)));

    result.coreml_beat_activation.assign(total_frames, 0.0f);
    result.coreml_downbeat_activation.assign(total_frames, 0.0f);

    result.coreml_beat_feature_frames.clear();
    result.coreml_beat_sample_frames.clear();
    result.coreml_beat_strengths.clear();

    for (double time : beat_times) {
        const std::size_t frame = static_cast<std::size_t>(std::llround(time * fps));
        if (frame < total_frames) {
            result.coreml_beat_activation[frame] = 1.0f;
            result.coreml_beat_feature_frames.push_back(static_cast<unsigned long long>(frame));
            const auto sample_frame =
                static_cast<unsigned long long>(std::llround(time * sample_rate));
            result.coreml_beat_sample_frames.push_back(sample_frame);
            result.coreml_beat_strengths.push_back(1.0f);
        }
    }

    result.coreml_downbeat_feature_frames.clear();
    for (double time : downbeat_times) {
        const std::size_t frame = static_cast<std::size_t>(std::llround(time * fps));
        if (frame < total_frames) {
            result.coreml_downbeat_activation[frame] = 1.0f;
            result.coreml_downbeat_feature_frames.push_back(static_cast<unsigned long long>(frame));
        }
    }

    result.estimated_bpm =
        estimate_bpm_from_beats(result.coreml_beat_sample_frames, sample_rate);
    return result;
}

#if defined(BEATIT_USE_TORCH)
CoreMLResult analyze_with_torch_activations(const std::vector<float>& samples,
                                            double sample_rate,
                                            const CoreMLConfig& config) {
    CoreMLResult result;
    if (config.torch_model_path.empty()) {
        if (config.verbose) {
            std::cerr << "Torch backend: missing model path.\n";
        }
        return result;
    }

    const std::filesystem::path model_path(config.torch_model_path);
    if (!std::filesystem::exists(model_path)) {
        if (config.verbose) {
            std::cerr << "Torch backend: model not found: " << config.torch_model_path << "\n";
        }
        return result;
    }

    torch::Device device(torch::kCPU);
    if (config.torch_device == "mps") {
        device = torch::Device(torch::kMPS);
    }

    torch::jit::script::Module module;
    try {
        if (config.verbose) {
            std::cerr << "Torch backend: loading model=" << config.torch_model_path
                      << " device=" << config.torch_device << "\n";
        }
        module = torch::jit::load(config.torch_model_path, torch::kCPU);
        module.to(torch::kFloat32);
        if (device.type() != torch::kCPU) {
            try {
                module.to(device);
            } catch (const c10::Error& err) {
                if (config.verbose) {
                    std::string message = err.what();
                    const std::size_t newline = message.find('\n');
                    if (newline != std::string::npos) {
                        message = message.substr(0, newline);
                    }
                    std::cerr << "Torch backend: device move failed, falling back to cpu: "
                              << message << "\n";
                }
                device = torch::kCPU;
            }
        }
    } catch (const c10::Error& err) {
        if (config.verbose) {
            std::cerr << "Torch backend: failed to load model: " << err.what() << "\n";
        }
        return result;
    }

    std::size_t frames = 0;
    std::vector<float> features;
    if (config.mel_backend == CoreMLConfig::MelBackend::Torch) {
#if defined(BEATIT_USE_TORCH)
        std::string mel_error;
        features = compute_mel_features_torch(samples, sample_rate, config, device, &frames, &mel_error);
#else
        if (config.verbose) {
            std::cerr << "Torch backend: mel backend requested but Torch is disabled.\n";
        }
#endif
    } else {
        features = compute_mel_features(samples, sample_rate, config, &frames);
    }
    if (features.empty() || frames == 0) {
        if (config.verbose) {
            std::cerr << "Torch backend: mel feature extraction failed.\n";
        }
        return result;
    }

    const std::size_t window_frames = config.fixed_frames > 0 ? config.fixed_frames : frames;
    const std::size_t hop_frames =
        config.window_hop_frames > 0 ? config.window_hop_frames : window_frames;
    const std::size_t border =
        std::min(config.window_border_frames, window_frames / 2);

    if (config.verbose) {
        std::cerr << "Torch backend: mel_frames=" << frames
                  << " window_frames=" << window_frames
                  << " hop_frames=" << hop_frames
                  << " border_frames=" << border
                  << " mel_bins=" << config.mel_bins
                  << " sample_rate=" << config.sample_rate
                  << " hop_size=" << config.hop_size
                  << " mel_backend="
                  << (config.mel_backend == CoreMLConfig::MelBackend::Torch ? "torch" : "cpu")
                  << "\n";
    }

    result.beat_activation.assign(frames, -1.0f);
    result.downbeat_activation.assign(frames, -1.0f);

    std::vector<char> filled(frames, 0);
    std::vector<char> downbeat_filled(frames, 0);

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const std::size_t batch_size = std::max<std::size_t>(1, config.torch_batch_size);

    struct BatchItem {
        std::size_t start = 0;
        std::size_t write_start = 0;
        std::size_t write_end = 0;
        std::size_t copy_frames = 0;
    };

    std::vector<BatchItem> batch_items;
    std::vector<float> batch_buffer;

    for (std::size_t start = 0; start < frames; ) {
        batch_items.clear();
        for (std::size_t slot = 0; slot < batch_size && start < frames; ++slot) {
            const std::size_t available = frames - start;
            if (available < window_frames && !config.pad_final_window) {
                break;
            }

            const std::size_t copy_frames = std::min(window_frames, available);
            const std::size_t write_start = start + border;
            const std::size_t write_end =
                std::min(frames, start + window_frames - border);
            if (write_end > write_start) {
                batch_items.push_back({start, write_start, write_end, copy_frames});
            }

            start += hop_frames;
        }

        if (batch_items.empty()) {
            break;
        }

        batch_buffer.assign(batch_items.size() * window_frames * config.mel_bins, 0.0f);
        for (std::size_t b = 0; b < batch_items.size(); ++b) {
            const auto& item = batch_items[b];
            for (std::size_t f = 0; f < item.copy_frames; ++f) {
                const std::size_t src = (item.start + f) * config.mel_bins;
                const std::size_t dst =
                    (b * window_frames + f) * config.mel_bins;
                std::copy(features.begin() + src,
                          features.begin() + src + config.mel_bins,
                          batch_buffer.begin() + dst);
            }
        }

        torch::Tensor input =
            torch::from_blob(batch_buffer.data(),
                             {static_cast<long long>(batch_items.size()),
                              static_cast<long long>(window_frames),
                              static_cast<long long>(config.mel_bins)},
                             torch::kFloat32)
                .to(options)
                .clone();

        torch::IValue output;
        std::vector<torch::IValue> inputs;
        inputs.reserve(1);
        inputs.emplace_back(input);
        try {
            c10::InferenceMode inference_guard(true);
            if (config.verbose) {
                std::cerr << "Torch backend: forward batch=" << batch_items.size()
                          << " start_frame=" << batch_items.front().start << "\n";
            }
            output = module.forward(inputs);
        } catch (const c10::Error& err) {
            if (config.verbose) {
                std::cerr << "Torch backend: forward failed at frame=" << batch_items.front().start
                          << " err=" << err.what() << "\n";
            }
            return CoreMLResult{};
        } catch (const std::exception& err) {
            if (config.verbose) {
                std::cerr << "Torch backend: forward exception at frame=" << batch_items.front().start
                          << " err=" << err.what() << "\n";
            }
            return CoreMLResult{};
        } catch (...) {
            if (config.verbose) {
                std::cerr << "Torch backend: forward unknown exception at frame="
                          << batch_items.front().start << "\n";
            }
            return CoreMLResult{};
        }

        torch::Tensor beat_tensor;
        torch::Tensor downbeat_tensor;
        if (output.isTuple()) {
            const auto tuple = output.toTuple();
            const auto& elements = tuple->elements();
            if (elements.size() >= 2 && elements[0].isTensor() && elements[1].isTensor()) {
                beat_tensor = elements[0].toTensor();
                downbeat_tensor = elements[1].toTensor();
            }
        } else if (output.isGenericDict()) {
            auto dict = output.toGenericDict();
            if (dict.contains("beat")) {
                beat_tensor = dict.at("beat").toTensor();
            }
            if (dict.contains("downbeat")) {
                downbeat_tensor = dict.at("downbeat").toTensor();
            }
        }

        if (!beat_tensor.defined()) {
            if (config.verbose) {
                std::cerr << "Torch backend: unexpected output signature.\n";
            }
            return CoreMLResult{};
        }

        beat_tensor = torch::sigmoid(beat_tensor).to(torch::kCPU);
        if (downbeat_tensor.defined()) {
            downbeat_tensor = torch::sigmoid(downbeat_tensor).to(torch::kCPU);
        }

        if (beat_tensor.dim() == 3 && beat_tensor.size(1) == 1) {
            beat_tensor = beat_tensor.squeeze(1);
        }
        if (downbeat_tensor.defined() && downbeat_tensor.dim() == 3 && downbeat_tensor.size(1) == 1) {
            downbeat_tensor = downbeat_tensor.squeeze(1);
        }

        if (beat_tensor.dim() == 1) {
            beat_tensor = beat_tensor.unsqueeze(0);
        }
        if (downbeat_tensor.defined() && downbeat_tensor.dim() == 1) {
            downbeat_tensor = downbeat_tensor.unsqueeze(0);
        }

        const auto beat_cpu = beat_tensor.contiguous();
        const auto beat_accessor = beat_cpu.accessor<float, 2>();
        torch::Tensor downbeat_cpu;
        if (downbeat_tensor.defined()) {
            downbeat_cpu = downbeat_tensor.contiguous();
        }

        for (std::size_t b = 0; b < batch_items.size(); ++b) {
            const auto& item = batch_items[b];
            for (std::size_t i = item.write_start; i < item.write_end; ++i) {
                if (filled[i]) {
                    continue;
                }
                const std::size_t local = i - item.start;
                result.beat_activation[i] =
                    beat_accessor[static_cast<long long>(b)][static_cast<long long>(local)];
                filled[i] = 1;
            }

            if (downbeat_cpu.defined()) {
                const auto downbeat_accessor = downbeat_cpu.accessor<float, 2>();
                for (std::size_t i = item.write_start; i < item.write_end; ++i) {
                    if (downbeat_filled[i]) {
                        continue;
                    }
                    const std::size_t local = i - item.start;
                    result.downbeat_activation[i] =
                        downbeat_accessor[static_cast<long long>(b)][static_cast<long long>(local)];
                    downbeat_filled[i] = 1;
                }
            }
        }

        if (batch_items.back().start + window_frames >= frames) {
            break;
        }
    }

    for (std::size_t i = 0; i < frames; ++i) {
        if (result.beat_activation[i] < 0.0f) {
            result.beat_activation[i] = 0.0f;
        }
        if (result.downbeat_activation[i] < 0.0f) {
            result.downbeat_activation[i] = 0.0f;
        }
    }

    return result;
}
#else
CoreMLResult analyze_with_torch_activations(const std::vector<float>&,
                                            double,
                                            const CoreMLConfig& config) {
    if (config.verbose) {
        std::cerr << "Torch backend not enabled in this build.\n";
    }
    return {};
}
#endif

std::size_t estimate_last_active_frame(const std::vector<float>& samples,
                                       double sample_rate,
                                       const CoreMLConfig& config) {
    if (samples.empty() || sample_rate <= 0.0 || config.hop_size == 0) {
        return 0;
    }

    const float rms_threshold = 0.001f;
    const std::size_t window = 1024;
    std::size_t last_active_sample = 0;
    bool found = false;
    for (std::size_t start = 0; start < samples.size(); start += window) {
        const std::size_t end = std::min(samples.size(), start + window);
        double sum_sq = 0.0;
        for (std::size_t i = start; i < end; ++i) {
            const double value = samples[i];
            sum_sq += value * value;
        }
        const double rms = sum_sq > 0.0
            ? std::sqrt(sum_sq / static_cast<double>(end - start))
            : 0.0;
        if (rms >= rms_threshold) {
            last_active_sample = end - 1;
            found = true;
        }
    }
    if (!found) {
        return 0;
    }

    const double ratio = static_cast<double>(config.sample_rate) / sample_rate;
    const double sample_pos = static_cast<double>(last_active_sample) * ratio;
    const std::size_t frame =
        static_cast<std::size_t>(std::llround(sample_pos / static_cast<double>(config.hop_size)));
    return frame;
}

} // namespace

AnalysisResult analyze(const std::vector<float>& samples,
                       double sample_rate,
                       const CoreMLConfig& config) {
    AnalysisResult result;
    if (samples.empty() || sample_rate <= 0.0) {
        return result;
    }

    if (config.backend == CoreMLConfig::Backend::BeatThisExternal) {
        return analyze_with_beatthis(samples, sample_rate, config);
    }
    if (config.backend == CoreMLConfig::Backend::Torch) {
        CoreMLConfig base_config = config;
        base_config.tempo_window_percent = 0.0f;
        base_config.prefer_double_time = false;
        base_config.synthetic_fill = false;

        const std::size_t last_active_frame =
            estimate_last_active_frame(samples, sample_rate, config);

        const std::vector<float> phase_energy = compute_phase_energy(samples, sample_rate, config);

        CoreMLResult raw = analyze_with_torch_activations(samples, sample_rate, base_config);
        CoreMLResult base = postprocess_coreml_activations(raw.beat_activation,
                                                          raw.downbeat_activation,
                                                          &phase_energy,
                                                          base_config,
                                                          sample_rate,
                                                          0.0f,
                                                          last_active_frame);
        const float peaks_bpm =
            estimate_bpm_from_activation(raw.beat_activation, config, sample_rate);
        const float autocorr_bpm =
            estimate_bpm_from_activation_autocorr(raw.beat_activation, config, sample_rate);
        const float comb_bpm =
            estimate_bpm_from_activation_comb(raw.beat_activation, config, sample_rate);
        const float beats_bpm = estimate_bpm_from_beats(base.beat_sample_frames, sample_rate);
        const auto choose_candidate_bpm = [&](float peaks,
                                              float autocorr,
                                              float comb,
                                              float beats) {
            const float tol = 0.02f;
            auto near = [&](float a, float b) {
                if (a <= 0.0f || b <= 0.0f) {
                    return false;
                }
                return (std::abs(a - b) / std::max(a, 1e-6f)) <= tol;
            };
            if (near(peaks, comb)) {
                return 0.5f * (peaks + comb);
            }
            if (near(peaks, autocorr)) {
                return 0.5f * (peaks + autocorr);
            }
            if (near(comb, autocorr)) {
                return 0.5f * (comb + autocorr);
            }
            if (peaks > 0.0f) {
                return peaks;
            }
            if (comb > 0.0f) {
                return comb;
            }
            if (autocorr > 0.0f) {
                return autocorr;
            }
            return beats;
        };

        const float reference_bpm =
            choose_candidate_bpm(peaks_bpm, autocorr_bpm, comb_bpm, beats_bpm);
        CoreMLResult final_result = postprocess_coreml_activations(raw.beat_activation,
                                                                  raw.downbeat_activation,
                                                                  &phase_energy,
                                                                  config,
                                                                  sample_rate,
                                                                  reference_bpm,
                                                                  last_active_frame);

        result.coreml_beat_activation = std::move(final_result.beat_activation);
        result.coreml_downbeat_activation = std::move(final_result.downbeat_activation);
        result.coreml_phase_energy = phase_energy;
        result.coreml_beat_feature_frames = std::move(final_result.beat_feature_frames);
        result.coreml_beat_sample_frames = std::move(final_result.beat_sample_frames);
        result.coreml_beat_projected_feature_frames =
            std::move(final_result.beat_projected_feature_frames);
        result.coreml_beat_projected_sample_frames =
            std::move(final_result.beat_projected_sample_frames);
        result.coreml_beat_strengths = std::move(final_result.beat_strengths);
        result.coreml_downbeat_feature_frames = std::move(final_result.downbeat_feature_frames);
        result.coreml_downbeat_projected_feature_frames =
            std::move(final_result.downbeat_projected_feature_frames);
        const auto& bpm_frames = result.coreml_beat_projected_sample_frames.empty()
            ? result.coreml_beat_sample_frames
            : result.coreml_beat_projected_sample_frames;
        result.estimated_bpm =
            estimate_bpm_from_beats(bpm_frames, sample_rate);
        const auto& marker_feature_frames = result.coreml_beat_projected_feature_frames.empty()
            ? result.coreml_beat_feature_frames
            : result.coreml_beat_projected_feature_frames;
        const auto& marker_sample_frames = result.coreml_beat_projected_sample_frames.empty()
            ? result.coreml_beat_sample_frames
            : result.coreml_beat_projected_sample_frames;
        const auto& marker_downbeats = result.coreml_downbeat_projected_feature_frames.empty()
            ? result.coreml_downbeat_feature_frames
            : result.coreml_downbeat_projected_feature_frames;
        result.coreml_beat_events =
            build_shakespear_markers(marker_feature_frames,
                                     marker_sample_frames,
                                     marker_downbeats,
                                     &result.coreml_beat_activation,
                                     result.estimated_bpm,
                                     sample_rate,
                                     config);

        return result;
    }

    CoreMLConfig base_config = config;
    base_config.tempo_window_percent = 0.0f;
    base_config.prefer_double_time = false;
    base_config.synthetic_fill = false;

    const std::size_t last_active_frame =
        estimate_last_active_frame(samples, sample_rate, config);

    const std::vector<float> phase_energy = compute_phase_energy(samples, sample_rate, config);

    CoreMLResult base = analyze_with_coreml(samples, sample_rate, base_config, 0.0f);
    const float peaks_bpm =
        estimate_bpm_from_activation(base.beat_activation, config, sample_rate);
    const float autocorr_bpm =
        estimate_bpm_from_activation_autocorr(base.beat_activation, config, sample_rate);
    const float comb_bpm =
        estimate_bpm_from_activation_comb(base.beat_activation, config, sample_rate);
    const float beats_bpm = estimate_bpm_from_beats(base.beat_sample_frames, sample_rate);
    const auto choose_candidate_bpm = [&](float peaks,
                                          float autocorr,
                                          float comb,
                                          float beats) {
        const float tol = 0.02f;
        auto near = [&](float a, float b) {
            if (a <= 0.0f || b <= 0.0f) {
                return false;
            }
            return (std::abs(a - b) / std::max(a, 1e-6f)) <= tol;
        };
        if (near(peaks, comb)) {
            return 0.5f * (peaks + comb);
        }
        if (near(peaks, autocorr)) {
            return 0.5f * (peaks + autocorr);
        }
        if (near(comb, autocorr)) {
            return 0.5f * (comb + autocorr);
        }
        if (peaks > 0.0f) {
            return peaks;
        }
        if (comb > 0.0f) {
            return comb;
        }
        if (autocorr > 0.0f) {
            return autocorr;
        }
        return beats;
    };

    const float reference_bpm =
        choose_candidate_bpm(peaks_bpm, autocorr_bpm, comb_bpm, beats_bpm);

    CoreMLResult final_result = postprocess_coreml_activations(base.beat_activation,
                                                              base.downbeat_activation,
                                                              &phase_energy,
                                                              config,
                                                              sample_rate,
                                                              reference_bpm,
                                                              last_active_frame);

    result.coreml_beat_activation = std::move(final_result.beat_activation);
    result.coreml_downbeat_activation = std::move(final_result.downbeat_activation);
    result.coreml_phase_energy = phase_energy;
    result.coreml_beat_feature_frames = std::move(final_result.beat_feature_frames);
    result.coreml_beat_sample_frames = std::move(final_result.beat_sample_frames);
    result.coreml_beat_projected_feature_frames =
        std::move(final_result.beat_projected_feature_frames);
    result.coreml_beat_projected_sample_frames =
        std::move(final_result.beat_projected_sample_frames);
    result.coreml_beat_strengths = std::move(final_result.beat_strengths);
    result.coreml_downbeat_feature_frames = std::move(final_result.downbeat_feature_frames);
    result.coreml_downbeat_projected_feature_frames =
        std::move(final_result.downbeat_projected_feature_frames);
    const auto& bpm_frames = result.coreml_beat_projected_sample_frames.empty()
        ? result.coreml_beat_sample_frames
        : result.coreml_beat_projected_sample_frames;
    result.estimated_bpm = estimate_bpm_from_beats(bpm_frames, sample_rate);
    const auto& marker_feature_frames = result.coreml_beat_projected_feature_frames.empty()
        ? result.coreml_beat_feature_frames
        : result.coreml_beat_projected_feature_frames;
    const auto& marker_sample_frames = result.coreml_beat_projected_sample_frames.empty()
        ? result.coreml_beat_sample_frames
        : result.coreml_beat_projected_sample_frames;
    const auto& marker_downbeats = result.coreml_downbeat_projected_feature_frames.empty()
        ? result.coreml_downbeat_feature_frames
        : result.coreml_downbeat_projected_feature_frames;
    result.coreml_beat_events =
        build_shakespear_markers(marker_feature_frames,
                                 marker_sample_frames,
                                 marker_downbeats,
                                 &result.coreml_beat_activation,
                                 result.estimated_bpm,
                                 sample_rate,
                                 config);

    return result;
}

} // namespace beatit
