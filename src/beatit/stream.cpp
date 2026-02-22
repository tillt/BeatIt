//
//  stream.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/stream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <vector>

#if defined(BEATIT_USE_TORCH)
#include <c10/core/InferenceMode.h>
#include <torch/script.h>
#include <torch/mps.h>
#include "beatit/torch_mel.h"
#endif

namespace beatit {
namespace {

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

} // namespace

#if defined(BEATIT_USE_TORCH)
struct BeatitStream::TorchState {
    torch::jit::script::Module module;
    torch::Device device = torch::kCPU;
};
#endif

BeatitStream::~BeatitStream() = default;

void BeatitStream::reset_state() {
    resampler_.src_index = 0.0;
    resampler_.buffer.clear();
    resampled_buffer_.clear();
    resampled_offset_ = 0;
    coreml_frame_offset_ = 0;
    coreml_beat_activation_.clear();
    coreml_downbeat_activation_.clear();
    coreml_phase_energy_.clear();
    total_input_samples_ = 0;
    total_seen_samples_ = 0;
    prepend_done_ = false;
    last_active_sample_ = 0;
    has_active_sample_ = false;
    phase_energy_state_ = 0.0;
    phase_energy_sum_sq_ = 0.0;
    phase_energy_sample_count_ = 0;
    perf_ = {};
}

bool BeatitStream::request_analysis_window(double* start_seconds,
                                           double* duration_seconds) const {
    const bool sparse_dynamic =
        coreml_config_.sparse_probe_mode &&
        coreml_config_.dbn_window_seconds > 0.0;
    if (sparse_dynamic) {
        // Sparse mode always operates on probe-sized windows.
        if (start_seconds) {
            *start_seconds = 0.0;
        }
        if (duration_seconds) {
            *duration_seconds = std::max(20.0, coreml_config_.dbn_window_seconds);
        }
        return true;
    }

    if (coreml_config_.max_analysis_seconds <= 0.0) {
        return false;
    }

    const double start =
        std::max(0.0,
                 coreml_config_.use_dbn
                     ? coreml_config_.dbn_window_start_seconds
                     : coreml_config_.analysis_start_seconds);
    const double duration = coreml_config_.max_analysis_seconds;
    if (duration <= 0.0) {
        return false;
    }

    if (start_seconds) {
        *start_seconds = start;
    }
    if (duration_seconds) {
        *duration_seconds = duration;
    }
    return true;
}

AnalysisResult BeatitStream::analyze_window(double start_seconds,
                                            double duration_seconds,
                                            double total_duration_seconds,
                                            const SampleProvider& provider) {
    AnalysisResult result;
    if (!provider || duration_seconds <= 0.0) {
        return result;
    }

    CoreMLConfig original_config = coreml_config_;
    auto run_probe = [&](double probe_start,
                         double probe_duration,
                         double forced_reference_bpm = 0.0) -> AnalysisResult {
        reset_state();
        coreml_config_ = original_config;
        if (coreml_config_.sparse_probe_mode) {
            // Sparse mode is intentionally single-switch for callers.
            coreml_config_.use_dbn = true;
        }
        coreml_config_.analysis_start_seconds = 0.0;
        coreml_config_.dbn_window_start_seconds = 0.0;
        coreml_config_.max_analysis_seconds = 0.0;
        if (forced_reference_bpm > 0.0) {
            tempo_reference_bpm_ = forced_reference_bpm;
            tempo_reference_valid_ = true;
            const float hard_min = std::max(1.0f, original_config.min_bpm);
            const float hard_max = std::max(hard_min + 1.0f, original_config.max_bpm);
            float local_min = static_cast<float>(forced_reference_bpm * 0.99);
            float local_max = static_cast<float>(forced_reference_bpm * 1.01);
            local_min = std::max(hard_min, local_min);
            local_max = std::min(hard_max, local_max);
            if (local_max <= local_min) {
                local_min = std::max(hard_min, static_cast<float>(forced_reference_bpm * 0.985));
                local_max = std::min(hard_max, static_cast<float>(forced_reference_bpm * 1.015));
            }
            if (local_max > local_min) {
                coreml_config_.min_bpm = local_min;
                coreml_config_.max_bpm = local_max;
            }
        }

        std::vector<float> window_samples;
        const std::size_t received =
            provider(std::max(0.0, probe_start), probe_duration, &window_samples);
        if (received > 0 && window_samples.size() >= received) {
            push(window_samples.data(), received);
        } else if (!window_samples.empty()) {
            push(window_samples.data(), window_samples.size());
        }

        if (total_duration_seconds > 0.0 && sample_rate_ > 0.0) {
            const double sample_count = std::ceil(total_duration_seconds * sample_rate_);
            total_seen_samples_ = static_cast<std::size_t>(std::max(0.0, sample_count));
        }
        return finalize();
    };

    const bool sparse_dynamic =
        original_config.sparse_probe_mode &&
        original_config.dbn_window_seconds > 0.0 &&
        total_duration_seconds > 0.0;
    if (!sparse_dynamic) {
        result = run_probe(start_seconds, duration_seconds);
        coreml_config_ = original_config;
        return result;
    }

    const double probe_duration = std::max(20.0, original_config.dbn_window_seconds);
    const double total = std::max(0.0, total_duration_seconds);
    const double max_start = std::max(0.0, total - probe_duration);
    const auto clamp_start = [&](double s) {
        return std::min(std::max(0.0, s), max_start);
    };
    constexpr double kSparseEdgeExclusionSeconds = 10.0;
    const double min_allowed_start = clamp_start(kSparseEdgeExclusionSeconds);
    const double max_allowed_start = clamp_start(
        std::max(0.0, total - kSparseEdgeExclusionSeconds - probe_duration));
    const double quality_shift_step = std::clamp(probe_duration * 0.25, 5.0, 20.0);
    const std::size_t max_quality_shifts = 8;
    double left_anchor_start = min_allowed_start;

    struct ProbeResult {
        double start = 0.0;
        AnalysisResult analysis;
        double bpm = 0.0;
        double conf = 0.0;
        double phase_abs_ms = std::numeric_limits<double>::infinity();
    };
    auto estimate_probe_confidence = [&](const AnalysisResult& r) -> double {
        const auto& beats = !r.coreml_beat_projected_sample_frames.empty()
            ? r.coreml_beat_projected_sample_frames
            : r.coreml_beat_sample_frames;
        if (beats.size() < 8 || sample_rate_ <= 0.0) {
            return 0.0;
        }
        std::vector<double> intervals;
        intervals.reserve(beats.size() - 1);
        for (std::size_t i = 1; i < beats.size(); ++i) {
            if (beats[i] > beats[i - 1]) {
                intervals.push_back(static_cast<double>(beats[i] - beats[i - 1]) / sample_rate_);
            }
        }
        if (intervals.size() < 4) {
            return 0.0;
        }
        double sum = 0.0;
        double sum_sq = 0.0;
        for (double v : intervals) {
            sum += v;
            sum_sq += v * v;
        }
        const double n = static_cast<double>(intervals.size());
        const double mean = sum / n;
        if (!(mean > 0.0)) {
            return 0.0;
        }
        const double var = std::max(0.0, (sum_sq / n) - (mean * mean));
        const double cv = std::sqrt(var) / mean;
        return (1.0 / (1.0 + cv)) * std::min(1.0, n / 32.0);
    };
    auto mode_penalty = [](double mode) {
        const double kTol = 1e-6;
        if (std::fabs(mode - 1.0) < kTol) {
            return 1.0;
        }
        if (std::fabs(mode - 0.5) < kTol || std::fabs(mode - 2.0) < kTol) {
            return 1.15;
        }
        if (std::fabs(mode - 1.5) < kTol || std::fabs(mode - (2.0 / 3.0)) < kTol) {
            return 1.9;
        }
        if (std::fabs(mode - 3.0) < kTol) {
            return 2.4;
        }
        return 2.0;
    };
    auto expand_modes = [&](double bpm, double conf) {
        struct ModeCand {
            double bpm = 0.0;
            double conf = 0.0;
            double mode = 1.0;
            double penalty = 1.0;
        };
        std::vector<ModeCand> out;
        if (!(bpm > 0.0) || !(conf > 0.0)) {
            return out;
        }
        static const double kModes[] = {0.5, 1.0, 2.0};
        for (double m : kModes) {
            const double cand = bpm * m;
            if (cand >= original_config.min_bpm && cand <= original_config.max_bpm) {
                out.push_back({cand, conf, m, mode_penalty(m)});
            }
        }
        return out;
    };
    auto relative_diff = [](double a, double b) {
        const double mean = 0.5 * (a + b);
        return mean > 0.0 ? (std::abs(a - b) / mean) : 1.0;
    };

    auto probe_beat_count = [](const AnalysisResult& r) -> std::size_t {
        const auto& beats = !r.coreml_beat_projected_sample_frames.empty()
            ? r.coreml_beat_projected_sample_frames
            : r.coreml_beat_sample_frames;
        return beats.size();
    };
    auto estimate_intro_phase_abs_ms =
        [&](const AnalysisResult& r, double bpm_hint) -> double {
        if (sample_rate_ <= 0.0 || bpm_hint <= 0.0 || !provider) {
            return std::numeric_limits<double>::infinity();
        }
        const auto& beats = !r.coreml_beat_projected_sample_frames.empty()
            ? r.coreml_beat_projected_sample_frames
            : r.coreml_beat_sample_frames;
        if (beats.size() < 12) {
            return std::numeric_limits<double>::infinity();
        }

        const std::size_t probe_beats = std::min<std::size_t>(24, beats.size());
        const double beat_period_s = 60.0 / bpm_hint;
        const double intro_s = std::max(20.0, beat_period_s * static_cast<double>(probe_beats + 4));

        std::vector<float> intro_samples;
        const std::size_t received = provider(0.0, intro_s, &intro_samples);
        if (received == 0 || intro_samples.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        if (intro_samples.size() > received) {
            intro_samples.resize(received);
        }
        const std::size_t radius = static_cast<std::size_t>(
            std::llround(sample_rate_ * beat_period_s * 0.6));
        if (radius == 0) {
            return std::numeric_limits<double>::infinity();
        }

        std::vector<double> abs_offsets_ms;
        abs_offsets_ms.reserve(probe_beats);
        for (std::size_t i = 0; i < probe_beats; ++i) {
            const std::size_t beat_frame = static_cast<std::size_t>(
                std::min<unsigned long long>(beats[i], intro_samples.size() - 1));
            const std::size_t start = beat_frame > radius ? beat_frame - radius : 0;
            const std::size_t end = std::min(intro_samples.size() - 1, beat_frame + radius);
            if (end <= start + 2) {
                continue;
            }

            std::size_t best_peak = beat_frame;
            float best_value = 0.0f;
            for (std::size_t p = start; p <= end; ++p) {
                const float value = std::fabs(intro_samples[p]);
                if (value > best_value) {
                    best_value = value;
                    best_peak = p;
                }
            }
            if (best_value <= 0.0f) {
                continue;
            }

            const double delta_frames = static_cast<double>(
                static_cast<long long>(best_peak) - static_cast<long long>(beat_frame));
            abs_offsets_ms.push_back(std::fabs((delta_frames * 1000.0) / sample_rate_));
        }
        if (abs_offsets_ms.size() < 8) {
            return std::numeric_limits<double>::infinity();
        }
        auto mid = abs_offsets_ms.begin() + static_cast<long>(abs_offsets_ms.size() / 2);
        std::nth_element(abs_offsets_ms.begin(), mid, abs_offsets_ms.end());
        return *mid;
    };
    auto run_probe_result = [&](double start_s) {
        ProbeResult p;
        p.start = clamp_start(start_s);
        p.analysis = run_probe(p.start, probe_duration);
        p.bpm = p.analysis.estimated_bpm;
        p.conf = estimate_probe_confidence(p.analysis);
        p.phase_abs_ms = estimate_intro_phase_abs_ms(p.analysis, p.bpm);
        return p;
    };
    auto probe_quality_score = [&](const ProbeResult& p) {
        const std::size_t beat_count = probe_beat_count(p.analysis);
        if (!(p.bpm > 0.0) || beat_count < 4) {
            return 0.0;
        }
        const double beat_factor =
            std::min(1.0, static_cast<double>(beat_count) / 24.0);
        const double phase_factor = std::isfinite(p.phase_abs_ms)
            ? (1.0 / (1.0 + (p.phase_abs_ms / 120.0)))
            : 0.15;
        return p.conf * beat_factor * phase_factor;
    };
    auto probe_is_usable = [&](const ProbeResult& p) {
        const std::size_t beat_count = probe_beat_count(p.analysis);
        return (p.bpm > 0.0) &&
               (beat_count >= 16) &&
               (p.conf >= 0.55) &&
               (!std::isfinite(p.phase_abs_ms) || p.phase_abs_ms <= 120.0);
    };
    auto seek_quality_probe = [&](double seed_start, bool shift_right) {
        double start_s = std::clamp(seed_start, min_allowed_start, max_allowed_start);
        ProbeResult best = run_probe_result(start_s);
        double best_score = probe_quality_score(best);
        if (probe_is_usable(best)) {
            return best;
        }
        for (std::size_t round = 0; round < max_quality_shifts; ++round) {
            double next_start = shift_right
                ? (start_s + quality_shift_step)
                : (start_s - quality_shift_step);
            next_start = std::clamp(next_start, min_allowed_start, max_allowed_start);
            if (std::abs(next_start - start_s) < 0.5) {
                break;
            }
            start_s = next_start;
            ProbeResult candidate = run_probe_result(start_s);
            const double score = probe_quality_score(candidate);
            if (score > best_score) {
                best = candidate;
                best_score = score;
            }
            if (probe_is_usable(candidate)) {
                return candidate;
            }
        }
        return best;
    };

    std::vector<ProbeResult> probes;
    probes.reserve(3);
    auto push_unique_probe = [&](ProbeResult&& probe) {
        const double incoming_score = probe_quality_score(probe);
        for (auto& existing : probes) {
            if (std::abs(existing.start - probe.start) < 1.0) {
                if (incoming_score > probe_quality_score(existing)) {
                    existing = std::move(probe);
                }
                return;
            }
        }
        probes.push_back(std::move(probe));
    };

    ProbeResult left_probe = seek_quality_probe(min_allowed_start, true);
    left_anchor_start = left_probe.start;
    push_unique_probe(std::move(left_probe));

    if (max_allowed_start > min_allowed_start + 0.5) {
        ProbeResult right_probe = seek_quality_probe(max_allowed_start, false);
        push_unique_probe(std::move(right_probe));
    }

    if (probes.size() < 2 && (max_allowed_start - min_allowed_start) > 1.0) {
        push_unique_probe(run_probe_result(clamp_start(total * 0.5 - probe_duration * 0.5)));
    }

    auto consensus_from_probes = [&](const std::vector<ProbeResult>& values) {
        struct ConsensusCand {
            double bpm = 0.0;
            double conf = 0.0;
            double mode = 1.0;
            double penalty = 1.0;
        };
        std::vector<ConsensusCand> all_modes;
        for (const auto& p : values) {
            const auto modes = expand_modes(p.bpm, p.conf);
            for (const auto& m : modes) {
                all_modes.push_back({m.bpm, m.conf, m.mode, m.penalty});
            }
        }
        if (all_modes.empty()) {
            return 0.0;
        }
        double best_bpm = 0.0;
        double best_score = std::numeric_limits<double>::infinity();
        for (const auto& cand : all_modes) {
            double score = 0.0;
            double support = 0.0;
            for (const auto& p : values) {
                const auto modes = expand_modes(p.bpm, p.conf);
                double best_local = std::numeric_limits<double>::infinity();
                for (const auto& m : modes) {
                    best_local = std::min(best_local, relative_diff(cand.bpm, m.bpm) * m.penalty);
                }
                score += best_local / std::max(1e-6, p.conf);
                if (best_local <= 0.02) {
                    support += p.conf;
                }
            }
            score *= cand.penalty;
            score /= (1.0 + (0.8 * support));
            if (score < best_score) {
                best_score = score;
                best_bpm = cand.bpm;
            }
        }
        return best_bpm;
    };

    struct IntroPhaseMetrics {
        double median_abs_ms = std::numeric_limits<double>::infinity();
        double odd_even_gap_ms = std::numeric_limits<double>::infinity();
        std::size_t count = 0;
    };
    auto measure_intro_phase = [&](const AnalysisResult& r, double bpm_hint) -> IntroPhaseMetrics {
        IntroPhaseMetrics metrics;
        if (sample_rate_ <= 0.0 || bpm_hint <= 0.0 || !provider) {
            return metrics;
        }
        const auto& beats = !r.coreml_beat_projected_sample_frames.empty()
            ? r.coreml_beat_projected_sample_frames
            : r.coreml_beat_sample_frames;
        if (beats.size() < 12) {
            return metrics;
        }

        const std::size_t probe_beats = std::min<std::size_t>(24, beats.size());
        const double beat_period_s = 60.0 / bpm_hint;
        const double intro_s = std::max(20.0, beat_period_s * static_cast<double>(probe_beats + 4));

        std::vector<float> intro_samples;
        const std::size_t received = provider(0.0, intro_s, &intro_samples);
        if (received == 0 || intro_samples.empty()) {
            return metrics;
        }
        if (intro_samples.size() > received) {
            intro_samples.resize(received);
        }

        const std::size_t radius = static_cast<std::size_t>(
            std::llround(sample_rate_ * beat_period_s * 0.6));
        if (radius == 0) {
            return metrics;
        }

        std::vector<double> signed_offsets_ms;
        std::vector<double> abs_offsets_ms;
        signed_offsets_ms.reserve(probe_beats);
        abs_offsets_ms.reserve(probe_beats);
        for (std::size_t i = 0; i < probe_beats; ++i) {
            const std::size_t beat_frame = static_cast<std::size_t>(
                std::min<unsigned long long>(beats[i], intro_samples.size() - 1));
            const std::size_t start = beat_frame > radius ? beat_frame - radius : 0;
            const std::size_t end = std::min(intro_samples.size() - 1, beat_frame + radius);
            if (end <= start + 2) {
                continue;
            }

            float window_max = 0.0f;
            for (std::size_t p = start; p <= end; ++p) {
                window_max = std::max(window_max, std::fabs(intro_samples[p]));
            }
            const float threshold = window_max * 0.6f;

            std::size_t best_peak = beat_frame;
            float best_value = 0.0f;
            for (std::size_t p = start + 1; p < end; ++p) {
                const float left = std::fabs(intro_samples[p - 1]);
                const float value = std::fabs(intro_samples[p]);
                const float right = std::fabs(intro_samples[p + 1]);
                if (value < threshold) {
                    continue;
                }
                if (value >= left && value > right && value > best_value) {
                    best_value = value;
                    best_peak = p;
                }
            }
            if (best_value <= 0.0f) {
                float max_value = 0.0f;
                for (std::size_t p = start; p <= end; ++p) {
                    const float value = std::fabs(intro_samples[p]);
                    if (value > max_value) {
                        max_value = value;
                        best_peak = p;
                    }
                }
            }

            const double delta_frames = static_cast<double>(
                static_cast<long long>(best_peak) - static_cast<long long>(beat_frame));
            const double offset_ms = (delta_frames * 1000.0) / sample_rate_;
            signed_offsets_ms.push_back(offset_ms);
            abs_offsets_ms.push_back(std::fabs(offset_ms));
        }

        if (abs_offsets_ms.size() < 8) {
            return metrics;
        }
        metrics.count = abs_offsets_ms.size();
        auto mid = abs_offsets_ms.begin() + static_cast<long>(abs_offsets_ms.size() / 2);
        std::nth_element(abs_offsets_ms.begin(), mid, abs_offsets_ms.end());
        metrics.median_abs_ms = *mid;

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
        auto median_inplace = [](std::vector<double>& values) -> double {
            if (values.empty()) {
                return 0.0;
            }
            auto m = values.begin() + static_cast<long>(values.size() / 2);
            std::nth_element(values.begin(), m, values.end());
            return *m;
        };
        if (!odd.empty() && !even.empty()) {
            metrics.odd_even_gap_ms = std::fabs(median_inplace(even) - median_inplace(odd));
        } else {
            metrics.odd_even_gap_ms = 0.0;
        }
        return metrics;
    };

    struct WindowPhaseMetrics {
        double median_ms = std::numeric_limits<double>::infinity();
        double median_abs_ms = std::numeric_limits<double>::infinity();
        double odd_even_gap_ms = std::numeric_limits<double>::infinity();
        std::size_t count = 0;
    };
    auto measure_window_phase = [&](const AnalysisResult& r,
                                    double bpm_hint,
                                    double window_start_s) -> WindowPhaseMetrics {
        WindowPhaseMetrics metrics;
        if (sample_rate_ <= 0.0 || bpm_hint <= 0.0 || !provider) {
            return metrics;
        }
        const auto& beats = !r.coreml_beat_projected_sample_frames.empty()
            ? r.coreml_beat_projected_sample_frames
            : r.coreml_beat_sample_frames;
        if (beats.size() < 12) {
            return metrics;
        }

        const unsigned long long window_start_frame = static_cast<unsigned long long>(
            std::llround(std::max(0.0, window_start_s) * sample_rate_));
        const unsigned long long window_end_frame = static_cast<unsigned long long>(
            std::llround(std::max(0.0, window_start_s + probe_duration) * sample_rate_));
        auto begin_it = std::lower_bound(beats.begin(), beats.end(), window_start_frame);
        auto end_it = std::upper_bound(beats.begin(), beats.end(), window_end_frame);
        if (begin_it == end_it || std::distance(begin_it, end_it) < 8) {
            return metrics;
        }

        const std::size_t radius = static_cast<std::size_t>(
            std::llround(sample_rate_ * (60.0 / bpm_hint) * 0.6));
        if (radius == 0) {
            return metrics;
        }

        const std::size_t margin = radius + static_cast<std::size_t>(std::llround(sample_rate_ * 1.5));
        const std::size_t first_frame = static_cast<std::size_t>(*begin_it);
        const std::size_t last_frame = static_cast<std::size_t>(*(end_it - 1));
        const std::size_t segment_start = first_frame > margin ? first_frame - margin : 0;
        const std::size_t segment_end = last_frame + margin;
        const double segment_start_s = static_cast<double>(segment_start) / sample_rate_;
        const double segment_duration_s =
            static_cast<double>(std::max<std::size_t>(1, segment_end - segment_start)) / sample_rate_;

        std::vector<float> samples;
        const std::size_t received = provider(segment_start_s, segment_duration_s, &samples);
        if (received == 0 || samples.empty()) {
            return metrics;
        }
        if (samples.size() > received) {
            samples.resize(received);
        }

        std::vector<double> signed_offsets_ms;
        std::vector<double> abs_offsets_ms;
        signed_offsets_ms.reserve(static_cast<std::size_t>(std::distance(begin_it, end_it)));
        abs_offsets_ms.reserve(static_cast<std::size_t>(std::distance(begin_it, end_it)));

        for (auto it = begin_it; it != end_it; ++it) {
            const std::size_t beat_frame = static_cast<std::size_t>(*it);
            if (beat_frame < segment_start) {
                continue;
            }
            const std::size_t local_center =
                std::min<std::size_t>(samples.size() - 1, beat_frame - segment_start);
            const std::size_t start = local_center > radius ? local_center - radius : 0;
            const std::size_t end = std::min(samples.size() - 1, local_center + radius);
            if (end <= start + 2) {
                continue;
            }

            float window_max = 0.0f;
            for (std::size_t p = start; p <= end; ++p) {
                window_max = std::max(window_max, std::fabs(samples[p]));
            }
            const float threshold = window_max * 0.6f;

            std::size_t best_peak = local_center;
            float best_value = 0.0f;
            for (std::size_t p = start + 1; p < end; ++p) {
                const float left = std::fabs(samples[p - 1]);
                const float value = std::fabs(samples[p]);
                const float right = std::fabs(samples[p + 1]);
                if (value < threshold) {
                    continue;
                }
                if (value >= left && value > right && value > best_value) {
                    best_value = value;
                    best_peak = p;
                }
            }
            if (best_value <= 0.0f) {
                float max_value = 0.0f;
                for (std::size_t p = start; p <= end; ++p) {
                    const float value = std::fabs(samples[p]);
                    if (value > max_value) {
                        max_value = value;
                        best_peak = p;
                    }
                }
            }

            const double delta_frames = static_cast<double>(
                static_cast<long long>(best_peak) - static_cast<long long>(local_center));
            const double offset_ms = (delta_frames * 1000.0) / sample_rate_;
            signed_offsets_ms.push_back(offset_ms);
            abs_offsets_ms.push_back(std::fabs(offset_ms));
        }

        if (abs_offsets_ms.size() < 8) {
            return metrics;
        }
        metrics.count = abs_offsets_ms.size();
        auto signed_mid = signed_offsets_ms.begin() + static_cast<long>(signed_offsets_ms.size() / 2);
        std::nth_element(signed_offsets_ms.begin(), signed_mid, signed_offsets_ms.end());
        metrics.median_ms = *signed_mid;
        auto abs_mid = abs_offsets_ms.begin() + static_cast<long>(abs_offsets_ms.size() / 2);
        std::nth_element(abs_offsets_ms.begin(), abs_mid, abs_offsets_ms.end());
        metrics.median_abs_ms = *abs_mid;

        auto median_inplace = [](std::vector<double>& values) -> double {
            if (values.empty()) {
                return 0.0;
            }
            auto m = values.begin() + static_cast<long>(values.size() / 2);
            std::nth_element(values.begin(), m, values.end());
            return *m;
        };

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
            metrics.odd_even_gap_ms = std::fabs(median_inplace(even) - median_inplace(odd));
        } else {
            metrics.odd_even_gap_ms = 0.0;
        }
        return metrics;
    };

    auto probe_start_extents = [&]() -> std::pair<double, double> {
        if (probes.empty()) {
            return {min_allowed_start, max_allowed_start};
        }
        double min_start = probes.front().start;
        double max_start = probes.front().start;
        for (const auto& p : probes) {
            min_start = std::min(min_start, p.start);
            max_start = std::max(max_start, p.start);
        }
        return {min_start, max_start};
    };

    double consensus_bpm = consensus_from_probes(probes);
    if (probes.size() >= 2 && consensus_bpm > 0.0) {
        double max_err = 0.0;
        for (const auto& p : probes) {
            const auto modes = expand_modes(p.bpm, p.conf);
            double best_local = std::numeric_limits<double>::infinity();
            for (const auto& m : modes) {
                best_local = std::min(best_local, relative_diff(consensus_bpm, m.bpm) * m.penalty);
            }
            max_err = std::max(max_err, best_local);
        }
        if (max_err > 0.025 && probes.size() == 2) {
            push_unique_probe(run_probe_result(clamp_start(total * 0.5 - probe_duration * 0.5)));
        }
    }

    bool have_consensus = false;
    std::vector<double> probe_mode_errors;
    std::vector<IntroPhaseMetrics> probe_intro_metrics;
    std::vector<WindowPhaseMetrics> probe_middle_metrics;
    double middle_probe_start = clamp_start(total * 0.5 - probe_duration * 0.5);
    double between_probe_start = clamp_start(0.5 * (min_allowed_start + middle_probe_start));

    auto recompute_probe_scores = [&]() {
        consensus_bpm = consensus_from_probes(probes);
        have_consensus = consensus_bpm > 0.0;

        probe_mode_errors.assign(probes.size(), 1.0);
        for (std::size_t i = 0; i < probes.size(); ++i) {
            if (!have_consensus) {
                probe_mode_errors[i] = 0.0;
                continue;
            }
            const auto modes = expand_modes(probes[i].bpm, probes[i].conf);
            double mode_error = 1.0;
            for (const auto& m : modes) {
                mode_error = std::min(mode_error, relative_diff(consensus_bpm, m.bpm) * m.penalty);
            }
            probe_mode_errors[i] = mode_error;
        }

        const auto probe_extents = probe_start_extents();
        middle_probe_start = clamp_start(0.5 * (probe_extents.first + probe_extents.second));
        between_probe_start = clamp_start(0.5 * (probe_extents.first + middle_probe_start));

        probe_intro_metrics.assign(probes.size(), IntroPhaseMetrics{});
        probe_middle_metrics.assign(probes.size(), WindowPhaseMetrics{});
        for (std::size_t i = 0; i < probes.size(); ++i) {
            const double bpm_hint = have_consensus ? consensus_bpm : probes[i].bpm;
            probe_intro_metrics[i] = measure_intro_phase(probes[i].analysis, bpm_hint);
            probe_middle_metrics[i] = measure_window_phase(
                probes[i].analysis, bpm_hint, middle_probe_start);
        }
    };

    recompute_probe_scores();

    const double anchor_start = left_anchor_start;

    struct DecisionOutcome {
        std::size_t selected_index = 0;
        double selected_score = std::numeric_limits<double>::infinity();
        double score_margin = 0.0;
        bool low_confidence = true;
        std::string mode;
    };
    auto decide_unified = [&]() {
        DecisionOutcome out;
        out.mode = "unified";
        double second_best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < probes.size(); ++i) {
            const auto& intro = probe_intro_metrics[i];
            const double tempo_term = probe_mode_errors[i];
            const double phase_term =
                std::isfinite(intro.median_abs_ms) ? (intro.median_abs_ms / 120.0) : 2.0;
            const double stability_term =
                std::isfinite(intro.odd_even_gap_ms) ? (intro.odd_even_gap_ms / 220.0) : 1.0;
            const double confidence_term = 1.0 - std::min(1.0, std::max(0.0, probes[i].conf));
            const double score =
                (2.4 * tempo_term) +
                (1.2 * phase_term) +
                (0.9 * stability_term) +
                (0.5 * confidence_term);
            if (score < out.selected_score) {
                second_best = out.selected_score;
                out.selected_score = score;
                out.selected_index = i;
            } else if (score < second_best) {
                second_best = score;
            }
        }
        out.score_margin = std::isfinite(second_best)
            ? std::max(0.0, second_best - out.selected_score)
            : 0.0;
        const auto& intro = probe_intro_metrics[out.selected_index];
        const bool severe_intro_miss =
            std::isfinite(intro.median_abs_ms) && intro.median_abs_ms > 220.0;
        const bool intro_miss_with_weak_conf =
            std::isfinite(intro.median_abs_ms) &&
            intro.median_abs_ms > 170.0 &&
            probes[out.selected_index].conf < 0.80;
        out.low_confidence =
            (probes[out.selected_index].conf < 0.55) ||
            (probe_mode_errors[out.selected_index] > 0.03) ||
            severe_intro_miss ||
            intro_miss_with_weak_conf ||
            (std::isfinite(intro.odd_even_gap_ms) && intro.odd_even_gap_ms > 220.0);
        return out;
    };
    auto window_phase_unstable = [&](const WindowPhaseMetrics& metrics, double bpm_hint) {
        if (metrics.count < 10 ||
            !std::isfinite(metrics.median_ms) ||
            !std::isfinite(metrics.median_abs_ms)) {
            return false;
        }
        const double beat_ms = bpm_hint > 0.0 ? (60000.0 / bpm_hint) : 500.0;
        const double signed_limit = std::max(30.0, beat_ms * 0.12);
        const double abs_limit = std::max(45.0, beat_ms * 0.18);
        return (std::fabs(metrics.median_ms) > signed_limit) &&
               (metrics.median_abs_ms > abs_limit);
    };

    DecisionOutcome decision = decide_unified();
    std::size_t selected_index = decision.selected_index;
    double selected_score = decision.selected_score;
    double score_margin = decision.score_margin;
    bool low_confidence = decision.low_confidence;
    IntroPhaseMetrics selected_intro_metrics = probe_intro_metrics[selected_index];
    WindowPhaseMetrics selected_middle_metrics = probe_middle_metrics[selected_index];
    bool interior_probe_added = false;

    if (probes.size() == 2) {
        const double selected_bpm_hint = have_consensus ? consensus_bpm : probes[selected_index].bpm;
        if (window_phase_unstable(selected_middle_metrics, selected_bpm_hint)) {
            push_unique_probe(run_probe_result(between_probe_start));
            recompute_probe_scores();
            decision = decide_unified();
            selected_index = decision.selected_index;
            selected_score = decision.selected_score;
            score_margin = decision.score_margin;
            low_confidence = decision.low_confidence;
            selected_intro_metrics = probe_intro_metrics[selected_index];
            selected_middle_metrics = probe_middle_metrics[selected_index];
            interior_probe_added = true;
        }
    }

    // Sparse mode returns directly from the best probe to avoid an extra anchor pass.
    result = probes[selected_index].analysis;
    const double selected_mode_error = probe_mode_errors[selected_index];
    if (low_confidence) {
        const double repair_bpm = have_consensus
            ? consensus_bpm
            : (probes[selected_index].bpm > 0.0 ? probes[selected_index].bpm : 0.0);
        double repair_start = anchor_start;
        if (have_consensus && !probes.empty()) {
            std::size_t best_mode_index = 0;
            double best_mode_error = probe_mode_errors[0];
            for (std::size_t i = 1; i < probes.size(); ++i) {
                if (probe_mode_errors[i] < best_mode_error) {
                    best_mode_error = probe_mode_errors[i];
                    best_mode_index = i;
                }
            }
            repair_start = probes[best_mode_index].start;
        }
        result = run_probe(repair_start, probe_duration, repair_bpm);
    }
    auto median_diff = [](const std::vector<unsigned long long>& frames) -> double {
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
    };
    auto median_value = [](std::vector<double>& values) -> double {
        if (values.empty()) {
            return 0.0;
        }
        auto mid = values.begin() + static_cast<long>(values.size() / 2);
        std::nth_element(values.begin(), mid, values.end());
        return *mid;
    };
    struct EdgeOffsetMetrics {
        double median_ms = std::numeric_limits<double>::infinity();
        double mad_ms = std::numeric_limits<double>::infinity();
        std::size_t count = 0;
    };
    auto measure_edge_offsets = [&](const std::vector<unsigned long long>& beats,
                                    double bpm_hint,
                                    bool from_end) -> EdgeOffsetMetrics {
        EdgeOffsetMetrics metrics;
        if (sample_rate_ <= 0.0 || bpm_hint <= 0.0 || !provider || beats.size() < 16) {
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

        const std::size_t radius = static_cast<std::size_t>(
            std::llround(sample_rate_ * (60.0 / bpm_hint) * 0.6));
        if (radius == 0) {
            return metrics;
        }
        const std::size_t margin = radius + static_cast<std::size_t>(std::llround(sample_rate_ * 1.5));
        const std::size_t first_frame = static_cast<std::size_t>(edge_beats.front());
        const std::size_t last_frame = static_cast<std::size_t>(edge_beats.back());
        const std::size_t segment_start = first_frame > margin ? first_frame - margin : 0;
        const std::size_t segment_end = last_frame + margin;
        const double segment_start_s = static_cast<double>(segment_start) / sample_rate_;
        const double segment_duration_s =
            static_cast<double>(std::max<std::size_t>(1, segment_end - segment_start)) / sample_rate_;

        std::vector<float> samples;
        const std::size_t received = provider(segment_start_s, segment_duration_s, &samples);
        if (received == 0 || samples.empty()) {
            return metrics;
        }
        if (samples.size() > received) {
            samples.resize(received);
        }

        std::vector<double> offsets_ms;
        offsets_ms.reserve(edge_beats.size());
        for (unsigned long long beat_frame_ull : edge_beats) {
            const std::size_t beat_frame = static_cast<std::size_t>(beat_frame_ull);
            if (beat_frame < segment_start) {
                continue;
            }
            const std::size_t local_center =
                std::min<std::size_t>(samples.size() - 1, beat_frame - segment_start);
            const std::size_t start = local_center > radius ? local_center - radius : 0;
            const std::size_t end = std::min(samples.size() - 1, local_center + radius);
            if (end <= start + 2) {
                continue;
            }

            float window_max = 0.0f;
            for (std::size_t i = start; i <= end; ++i) {
                window_max = std::max(window_max, std::fabs(samples[i]));
            }
            const float threshold = window_max * 0.6f;

            std::size_t best_peak = local_center;
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
            if (best_value <= 0.0f) {
                float max_value = 0.0f;
                for (std::size_t i = start; i <= end; ++i) {
                    const float value = std::fabs(samples[i]);
                    if (value > max_value) {
                        max_value = value;
                        best_peak = i;
                    }
                }
            }

            const double delta_frames = static_cast<double>(
                static_cast<long long>(best_peak) - static_cast<long long>(local_center));
            offsets_ms.push_back((delta_frames * 1000.0) / sample_rate_);
        }
        if (offsets_ms.size() < 8) {
            return metrics;
        }
        metrics.count = offsets_ms.size();
        std::vector<double> med_buf = offsets_ms;
        metrics.median_ms = median_value(med_buf);
        std::vector<double> abs_dev;
        abs_dev.reserve(offsets_ms.size());
        for (double v : offsets_ms) {
            abs_dev.push_back(std::fabs(v - metrics.median_ms));
        }
        metrics.mad_ms = median_value(abs_dev);
        return metrics;
    };
    auto apply_bounded_grid_refit = [&](AnalysisResult* r) {
        if (!r) {
            return;
        }
        auto& projected = r->coreml_beat_projected_sample_frames;
        const auto& observed = r->coreml_beat_sample_frames;
        if (projected.size() < 32 || observed.size() < 32) {
            return;
        }
        const double base_step = median_diff(projected);
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
        const double head_med = median_value(head);
        const double tail_med = median_value(tail);
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
    };
    const bool needs_bounded_refit =
        low_confidence ||
        (std::isfinite(selected_intro_metrics.median_abs_ms) &&
         selected_intro_metrics.median_abs_ms > 60.0);
    if (needs_bounded_refit) {
        apply_bounded_grid_refit(&result);
    }
    auto apply_sparse_anchor_state_refit = [&](AnalysisResult* r) {
        if (!r || sample_rate_ <= 0.0 || probes.size() < 2) {
            return;
        }

        std::vector<unsigned long long>* projected = nullptr;
        if (!r->coreml_beat_projected_sample_frames.empty()) {
            projected = &r->coreml_beat_projected_sample_frames;
        } else if (!r->coreml_beat_sample_frames.empty()) {
            projected = &r->coreml_beat_sample_frames;
        }
        if (!projected || projected->size() < 64) {
            return;
        }

        struct AnchorObservation {
            double local_step = 0.0;
            double weight = 0.0;
            double start = 0.0;
        };

        auto nearest_index = [](const std::vector<unsigned long long>& beats,
                                unsigned long long frame) -> std::size_t {
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
                return median_diff(beats);
            }
            return median_value(diffs);
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
                static_cast<unsigned long long>(std::llround(center_s * sample_rate_));
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

        const double base_step = median_diff(*projected);
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

        if (original_config.verbose) {
            std::cerr << "Sparse anchor state refit:"
                      << " anchors=" << anchors.size()
                      << " spread_ratio=" << spread_ratio
                      << " step_target=" << step_target
                      << " ratio=" << raw_ratio
                      << " ratio_applied=" << ratio
                      << " base_step=" << base_step
                      << "\n";
        }
    };
    apply_sparse_anchor_state_refit(&result);
    auto apply_waveform_edge_refit = [&](AnalysisResult* r) {
        if (!r || sample_rate_ <= 0.0 || !provider) {
            return;
        }
        const bool second_pass_enabled = []() {
            const char* v = std::getenv("BEATIT_EDGE_REFIT_SECOND_PASS");
            if (!v || v[0] == '\0') {
                return true;
            }
            return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' ||
                     v[0] == 'n' || v[0] == 'N');
        }();
        std::vector<unsigned long long>* projected = nullptr;
        if (!r->coreml_beat_projected_sample_frames.empty()) {
            projected = &r->coreml_beat_projected_sample_frames;
        } else if (!r->coreml_beat_sample_frames.empty()) {
            projected = &r->coreml_beat_sample_frames;
        }
        if (!projected || projected->size() < 64) {
            return;
        }

        const double bpm_hint = r->estimated_bpm > 0.0f
            ? static_cast<double>(r->estimated_bpm)
            : estimate_bpm_from_beats_local(*projected, sample_rate_);
        if (!(bpm_hint > 0.0)) {
            return;
        }

        std::size_t first_probe_index = 0;
        std::size_t last_probe_index = 0;
        for (std::size_t i = 1; i < probes.size(); ++i) {
            if (probes[i].start < probes[first_probe_index].start) {
                first_probe_index = i;
            }
            if (probes[i].start > probes[last_probe_index].start) {
                last_probe_index = i;
            }
        }
        const double first_probe_start = probes[first_probe_index].start;
        const double last_probe_start = probes[last_probe_index].start;
        if (std::abs(last_probe_start - first_probe_start) < 1.0) {
            return;
        }

        auto select_window_beats = [&](const std::vector<unsigned long long>& beats,
                                       double window_start_s) {
            std::vector<unsigned long long> selected;
            if (sample_rate_ <= 0.0 || beats.empty()) {
                return selected;
            }
            const unsigned long long window_start_frame = static_cast<unsigned long long>(
                std::llround(std::max(0.0, window_start_s) * sample_rate_));
            const unsigned long long window_end_frame = static_cast<unsigned long long>(
                std::llround(std::max(0.0, window_start_s + probe_duration) * sample_rate_));
            auto begin_it = std::lower_bound(beats.begin(), beats.end(), window_start_frame);
            auto end_it = std::upper_bound(beats.begin(), beats.end(), window_end_frame);
            if (begin_it < end_it) {
                selected.insert(selected.end(), begin_it, end_it);
            }
            return selected;
        };
        auto measure_window_pair = [&](const std::vector<unsigned long long>& beats,
                                       double first_window_start_s,
                                       double last_window_start_s) {
            const std::vector<unsigned long long> first_window_beats =
                select_window_beats(beats, first_window_start_s);
            const std::vector<unsigned long long> last_window_beats =
                select_window_beats(beats, last_window_start_s);
            return std::pair<EdgeOffsetMetrics, EdgeOffsetMetrics>{
                measure_edge_offsets(first_window_beats, bpm_hint, false),
                measure_edge_offsets(last_window_beats, bpm_hint, true)};
        };
        auto window_usable = [](const EdgeOffsetMetrics& m) {
            return m.count >= 10 &&
                   std::isfinite(m.median_ms) &&
                   std::isfinite(m.mad_ms) &&
                   m.mad_ms <= 120.0;
        };
        const double shift_step = std::clamp(probe_duration * 0.25, 5.0, 20.0);
        double first_window_start = first_probe_start;
        double last_window_start = last_probe_start;
        EdgeOffsetMetrics intro;
        EdgeOffsetMetrics outro;
        std::size_t quality_shift_rounds = 0;
        for (; quality_shift_rounds < 6; ++quality_shift_rounds) {
            auto pair = measure_window_pair(*projected, first_window_start, last_window_start);
            intro = pair.first;
            outro = pair.second;
            const bool intro_ok = window_usable(intro);
            const bool outro_ok = window_usable(outro);
            if (intro_ok && outro_ok) {
                break;
            }
            bool moved = false;
            if (!intro_ok) {
                const double next = clamp_start(first_window_start + shift_step);
                if (next > first_window_start + 0.5) {
                    first_window_start = next;
                    moved = true;
                }
            }
            if (!outro_ok) {
                const double next = clamp_start(last_window_start - shift_step);
                if (next + 0.5 < last_window_start) {
                    last_window_start = next;
                    moved = true;
                }
            }
            if (!moved || (last_window_start - first_window_start) < std::max(1.0, shift_step)) {
                break;
            }
        }
        if (intro.count < 8 || outro.count < 8 ||
            !std::isfinite(intro.median_ms) || !std::isfinite(outro.median_ms)) {
            return;
        }

        auto apply_scale = [&](std::vector<unsigned long long>* beats,
                               double ratio,
                               double min_ratio,
                               double max_ratio,
                               double min_delta) -> double {
            if (!beats || beats->size() < 2) {
                return 1.0;
            }
            const double clamped_ratio = std::clamp(ratio, min_ratio, max_ratio);
            if (std::abs(clamped_ratio - 1.0) < min_delta) {
                return 1.0;
            }
            const long long anchor = static_cast<long long>(beats->front());
            for (std::size_t i = 0; i < beats->size(); ++i) {
                const long long current = static_cast<long long>((*beats)[i]);
                const double rel = static_cast<double>(current - anchor);
                const long long adjusted =
                    anchor + static_cast<long long>(std::llround(rel * clamped_ratio));
                (*beats)[i] =
                    static_cast<unsigned long long>(std::max<long long>(0, adjusted));
            }
            return clamped_ratio;
        };

        const auto compute_ratio = [&](const std::vector<unsigned long long>& beats,
                                       const EdgeOffsetMetrics& a,
                                       const EdgeOffsetMetrics& b) -> double {
            const double base_step = median_diff(beats);
            const double beats_span = static_cast<double>(beats.size() - 1);
            if (!(base_step > 0.0) || !(beats_span > 0.0)) {
                return 1.0;
            }
            const double err_delta_frames = ((b.median_ms - a.median_ms) * sample_rate_) / 1000.0;
            const double per_beat_adjust = err_delta_frames / beats_span;
            return 1.0 + (per_beat_adjust / base_step);
        };

        const double ratio = compute_ratio(*projected, intro, outro);
        const double applied_ratio =
            apply_scale(projected, ratio, 0.9995, 1.0005, 1e-5);
        if (applied_ratio == 1.0) {
            return;
        }

        EdgeOffsetMetrics post_intro = intro;
        EdgeOffsetMetrics post_outro = outro;
        if (second_pass_enabled) {
            auto measured = measure_window_pair(*projected, first_window_start, last_window_start);
            post_intro = measured.first;
            post_outro = measured.second;
            if (post_intro.count >= 8 && post_outro.count >= 8 &&
                std::isfinite(post_intro.median_ms) && std::isfinite(post_outro.median_ms)) {
                const double post_delta = std::abs(post_outro.median_ms - post_intro.median_ms);
                std::vector<unsigned long long> candidate = *projected;
                const double pass2_ratio = compute_ratio(candidate, post_intro, post_outro);
                const double pass2_applied =
                    apply_scale(&candidate, pass2_ratio, 0.9997, 1.0003, 1e-6);
                if (pass2_applied != 1.0) {
                    auto candidate_measured =
                        measure_window_pair(candidate, first_window_start, last_window_start);
                    const EdgeOffsetMetrics cand_intro = candidate_measured.first;
                    const EdgeOffsetMetrics cand_outro = candidate_measured.second;
                    if (cand_intro.count >= 8 && cand_outro.count >= 8 &&
                        std::isfinite(cand_intro.median_ms) && std::isfinite(cand_outro.median_ms)) {
                        const double cand_delta = std::abs(cand_outro.median_ms - cand_intro.median_ms);
                        const bool improves_delta = cand_delta <= post_delta;
                        const bool keeps_intro =
                            std::abs(cand_intro.median_ms) <= (std::abs(post_intro.median_ms) + 3.0);
                        if (improves_delta && keeps_intro) {
                            *projected = std::move(candidate);
                            post_intro = cand_intro;
                            post_outro = cand_outro;
                        }
                    }
                }
            }
        }

        auto try_uniform_shift_on_windows =
            [&](const EdgeOffsetMetrics& base_intro,
                const EdgeOffsetMetrics& base_outro,
                double max_beat_fraction) {
            if (base_intro.count < 8 || base_outro.count < 8 ||
                !std::isfinite(base_intro.median_ms) || !std::isfinite(base_outro.median_ms)) {
                return false;
            }
            if ((base_intro.median_ms * base_outro.median_ms) <= 0.0) {
                return false;
            }
            const double mean_ms = 0.5 * (base_intro.median_ms + base_outro.median_ms);
            const double beat_ms = 60000.0 / std::max(1e-6, bpm_hint);
            const double max_shift_ms = std::max(25.0, beat_ms * max_beat_fraction);
            const double clamped_shift_ms = std::clamp(mean_ms, -max_shift_ms, max_shift_ms);
            const long long shift_frames = static_cast<long long>(
                std::llround((clamped_shift_ms * sample_rate_) / 1000.0));
            if (shift_frames == 0) {
                return false;
            }

            std::vector<unsigned long long> candidate = *projected;
            for (std::size_t i = 0; i < candidate.size(); ++i) {
                const long long shifted = static_cast<long long>(candidate[i]) + shift_frames;
                candidate[i] = static_cast<unsigned long long>(std::max<long long>(0, shifted));
            }
            const auto measured =
                measure_window_pair(candidate, first_window_start, last_window_start);
            const EdgeOffsetMetrics cand_intro = measured.first;
            const EdgeOffsetMetrics cand_outro = measured.second;
            if (cand_intro.count < 8 || cand_outro.count < 8 ||
                !std::isfinite(cand_intro.median_ms) || !std::isfinite(cand_outro.median_ms)) {
                return false;
            }

            const double base_worst =
                std::max(std::abs(base_intro.median_ms), std::abs(base_outro.median_ms));
            const double cand_worst =
                std::max(std::abs(cand_intro.median_ms), std::abs(cand_outro.median_ms));
            if (cand_worst + 5.0 < base_worst) {
                *projected = std::move(candidate);
                return true;
            }
            return false;
        };
        if (try_uniform_shift_on_windows(post_intro, post_outro, 0.30)) {
            auto measured = measure_window_pair(*projected, first_window_start, last_window_start);
            post_intro = measured.first;
            post_outro = measured.second;
        }

        // Final safety pass: minimize whole-file edge drift while preserving intro alignment.
        EdgeOffsetMetrics global_intro = measure_edge_offsets(*projected, bpm_hint, false);
        EdgeOffsetMetrics global_outro = measure_edge_offsets(*projected, bpm_hint, true);
        double global_guard_ratio = 1.0;
        if (global_intro.count >= 8 && global_outro.count >= 8 &&
            std::isfinite(global_intro.median_ms) && std::isfinite(global_outro.median_ms)) {
            const double global_delta = std::abs(global_outro.median_ms - global_intro.median_ms);
            if (global_delta > 30.0) {
                std::vector<unsigned long long> candidate = *projected;
                const double guard_ratio =
                    compute_ratio(candidate, global_intro, global_outro);
                const double guard_applied =
                    apply_scale(&candidate, guard_ratio, 0.99985, 1.00015, 1e-6);
                if (guard_applied != 1.0) {
                    const EdgeOffsetMetrics cand_intro =
                        measure_edge_offsets(candidate, bpm_hint, false);
                    const EdgeOffsetMetrics cand_outro =
                        measure_edge_offsets(candidate, bpm_hint, true);
                    if (cand_intro.count >= 8 && cand_outro.count >= 8 &&
                        std::isfinite(cand_intro.median_ms) && std::isfinite(cand_outro.median_ms)) {
                        const double cand_delta =
                            std::abs(cand_outro.median_ms - cand_intro.median_ms);
                        const bool improves_delta = cand_delta <= (global_delta - 1.0);
                        const bool keeps_intro =
                            std::abs(cand_intro.median_ms) <= (std::abs(global_intro.median_ms) + 4.0);
                        if (improves_delta && keeps_intro) {
                            *projected = std::move(candidate);
                            global_intro = cand_intro;
                            global_outro = cand_outro;
                            global_guard_ratio = guard_applied;
                        }
                    }
                }
            }
        }
        if (global_intro.count >= 8 && global_outro.count >= 8 &&
            std::isfinite(global_intro.median_ms) && std::isfinite(global_outro.median_ms)) {
            const double global_delta = std::abs(global_outro.median_ms - global_intro.median_ms);
            if (global_delta > 60.0 && global_delta <= 120.0) {
                std::vector<unsigned long long> candidate = *projected;
                const double guard_ratio =
                    compute_ratio(candidate, global_intro, global_outro);
                const double guard_applied =
                    apply_scale(&candidate, guard_ratio, 0.9996, 1.0004, 1e-6);
                if (guard_applied != 1.0) {
                    const EdgeOffsetMetrics cand_intro =
                        measure_edge_offsets(candidate, bpm_hint, false);
                    const EdgeOffsetMetrics cand_outro =
                        measure_edge_offsets(candidate, bpm_hint, true);
                    if (cand_intro.count >= 8 && cand_outro.count >= 8 &&
                        std::isfinite(cand_intro.median_ms) && std::isfinite(cand_outro.median_ms)) {
                        const double cand_delta =
                            std::abs(cand_outro.median_ms - cand_intro.median_ms);
                        const double base_worst =
                            std::max(std::abs(global_intro.median_ms), std::abs(global_outro.median_ms));
                        const double cand_worst =
                            std::max(std::abs(cand_intro.median_ms), std::abs(cand_outro.median_ms));
                        const bool improves_delta = cand_delta + 2.0 < global_delta;
                        const bool keeps_worst_reasonable = cand_worst <= (base_worst + 10.0);
                        if (improves_delta && keeps_worst_reasonable) {
                            *projected = std::move(candidate);
                            global_intro = cand_intro;
                            global_outro = cand_outro;
                            global_guard_ratio = guard_applied;
                        }
                    }
                }
            }
        }
        if (global_intro.count >= 8 && global_outro.count >= 8 &&
            std::isfinite(global_intro.median_ms) && std::isfinite(global_outro.median_ms) &&
            (global_intro.median_ms * global_outro.median_ms) > 0.0) {
            const double mean_ms = 0.5 * (global_intro.median_ms + global_outro.median_ms);
            const double beat_ms = 60000.0 / std::max(1e-6, bpm_hint);
            const double max_shift_ms = std::max(40.0, beat_ms * 0.35);
            const double clamped_shift_ms = std::clamp(mean_ms, -max_shift_ms, max_shift_ms);
            const long long shift_frames = static_cast<long long>(
                std::llround((clamped_shift_ms * sample_rate_) / 1000.0));
            if (shift_frames != 0) {
                std::vector<unsigned long long> candidate = *projected;
                for (std::size_t i = 0; i < candidate.size(); ++i) {
                    const long long shifted =
                        static_cast<long long>(candidate[i]) + shift_frames;
                    candidate[i] =
                        static_cast<unsigned long long>(std::max<long long>(0, shifted));
                }
                const EdgeOffsetMetrics cand_intro =
                    measure_edge_offsets(candidate, bpm_hint, false);
                const EdgeOffsetMetrics cand_outro =
                    measure_edge_offsets(candidate, bpm_hint, true);
                if (cand_intro.count >= 8 && cand_outro.count >= 8 &&
                    std::isfinite(cand_intro.median_ms) && std::isfinite(cand_outro.median_ms)) {
                    const double base_worst =
                        std::max(std::abs(global_intro.median_ms), std::abs(global_outro.median_ms));
                    const double cand_worst =
                        std::max(std::abs(cand_intro.median_ms), std::abs(cand_outro.median_ms));
                    if (cand_worst + 5.0 < base_worst) {
                        *projected = std::move(candidate);
                        global_intro = cand_intro;
                        global_outro = cand_outro;
                    }
                }
            }
        }

        if (original_config.verbose) {
            const double err_delta_frames =
                ((outro.median_ms - intro.median_ms) * sample_rate_) / 1000.0;
            std::cerr << "Sparse edge refit:"
                      << " second_pass=" << (second_pass_enabled ? 1 : 0)
                      << " first_probe_start_s=" << first_probe_start
                      << " last_probe_start_s=" << last_probe_start
                      << " first_window_start_s=" << first_window_start
                      << " last_window_start_s=" << last_window_start
                      << " quality_shift_rounds=" << quality_shift_rounds
                      << " intro_ms=" << intro.median_ms
                      << " outro_ms=" << outro.median_ms
                      << " post_intro_ms=" << post_intro.median_ms
                      << " post_outro_ms=" << post_outro.median_ms
                      << " global_intro_ms=" << global_intro.median_ms
                      << " global_outro_ms=" << global_outro.median_ms
                      << " global_ratio_applied=" << global_guard_ratio
                      << " delta_frames=" << err_delta_frames
                      << " ratio=" << ratio
                      << " ratio_applied=" << applied_ratio
                      << " beats=" << projected->size()
                      << "\n";
        }
    };
    apply_waveform_edge_refit(&result);

    {
        // Keep reported BPM consistent with the returned beat grid.
        const auto& bpm_frames = !result.coreml_beat_projected_sample_frames.empty()
            ? result.coreml_beat_projected_sample_frames
            : result.coreml_beat_sample_frames;
        const float grid_bpm = normalize_bpm_to_range_local(
            estimate_bpm_from_beats_local(bpm_frames, sample_rate_),
            std::max(1.0f, original_config.min_bpm),
            std::max(std::max(1.0f, original_config.min_bpm) + 1.0f, original_config.max_bpm));
        if (grid_bpm > 0.0f) {
            result.estimated_bpm = grid_bpm;
        } else if (have_consensus && consensus_bpm > 0.0) {
            result.estimated_bpm = static_cast<float>(consensus_bpm);
        }
    }
    {
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
                                     sample_rate_,
                                     original_config);
    }
    if (original_config.verbose) {
        std::cerr << "Sparse probes:";
        for (std::size_t i = 0; i < probes.size(); ++i) {
            const auto& p = probes[i];
            std::cerr << " start=" << p.start
                      << " bpm=" << p.bpm
                      << " conf=" << p.conf
                      << " mode_err=" << probe_mode_errors[i];
        }
        std::cerr << " consensus=" << consensus_bpm
                  << " anchor_start=" << anchor_start
                  << " decision=" << decision.mode
                  << " selected_start=" << probes[selected_index].start
                  << " selected_score=" << selected_score
                  << " score_margin=" << score_margin
                  << " selected_mode_err=" << selected_mode_error
                  << " selected_conf=" << probes[selected_index].conf
                  << " selected_intro_abs_ms=" << selected_intro_metrics.median_abs_ms
                  << " selected_odd_even_gap_ms=" << selected_intro_metrics.odd_even_gap_ms
                  << " selected_middle_ms=" << selected_middle_metrics.median_ms
                  << " selected_middle_abs_ms=" << selected_middle_metrics.median_abs_ms
                  << " selected_middle_odd_even_gap_ms=" << selected_middle_metrics.odd_even_gap_ms
                  << " middle_probe_start_s=" << middle_probe_start
                  << " between_probe_start_s=" << between_probe_start
                  << " interior_probe_added=" << (interior_probe_added ? 1 : 0)
                  << " repair=" << (low_confidence ? 1 : 0)
                  << "\n";
    }
    coreml_config_ = original_config;
    return result;
}

void BeatitStream::LinearResampler::push(const float* input,
                                         std::size_t count,
                                         std::vector<float>* output) {
    if (!output || count == 0) {
        return;
    }

    buffer.insert(buffer.end(), input, input + count);
    if (ratio <= 0.0) {
        return;
    }

    const double step = 1.0 / ratio;
    while (src_index + 1.0 < static_cast<double>(buffer.size())) {
        const std::size_t index = static_cast<std::size_t>(src_index);
        const double frac = src_index - static_cast<double>(index);
        const float a = buffer[index];
        const float b = buffer[index + 1];
        output->push_back(static_cast<float>((1.0 - frac) * a + frac * b));
        src_index += step;
    }

    const std::size_t drop = static_cast<std::size_t>(src_index);
    if (drop > 0) {
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<long>(drop));
        src_index -= static_cast<double>(drop);
    }
}

#if defined(BEATIT_USE_TORCH)
bool BeatitStream::infer_torch_window(const std::vector<float>& window,
                                      std::vector<float>* beat,
                                      std::vector<float>* downbeat) {
    if (!beat || !downbeat) {
        return false;
    }
    if (coreml_config_.torch_model_path.empty()) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: missing model path.\n";
        }
        return false;
    }

    if (!torch_state_) {
        torch_state_ = std::make_unique<TorchState>();
        torch_state_->device = torch::kCPU;
        if (coreml_config_.torch_device == "mps") {
            void* metal_graph = dlopen("/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/MetalPerformanceShadersGraph",
                                       RTLD_LAZY);
            void* metal_mps = dlopen("/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders",
                                     RTLD_LAZY);
            void* metal = dlopen("/System/Library/Frameworks/Metal.framework/Metal",
                                 RTLD_LAZY);
            if (!metal_graph || !metal_mps || !metal) {
                if (coreml_config_.verbose) {
                    std::cerr << "Torch backend: Metal frameworks missing, MPS unavailable.\n";
                }
                torch_state_->device = torch::kCPU;
            } else {
                dlclose(metal_graph);
                dlclose(metal_mps);
                dlclose(metal);
                torch_state_->device = torch::kMPS;
            }
        }
        try {
            torch_state_->module = torch::jit::load(coreml_config_.torch_model_path, torch::kCPU);
            torch_state_->module.to(torch::kFloat32);
            if (torch_state_->device.type() != torch::kCPU) {
                try {
                    torch_state_->module.to(torch_state_->device);
                } catch (const c10::Error& err) {
                    if (coreml_config_.verbose) {
                        std::string message = err.what();
                        const std::size_t newline = message.find('\n');
                        if (newline != std::string::npos) {
                            message = message.substr(0, newline);
                        }
                        std::cerr << "Torch backend: device move failed, falling back to cpu: "
                                  << message << "\n";
                    }
                    torch_state_->device = torch::kCPU;
                }
            }
            if (coreml_config_.verbose) {
                std::cerr << "Torch backend: resolved device="
                          << torch_state_->device.str() << "\n";
                std::cerr << "Torch backend: mel backend="
                          << (coreml_config_.mel_backend == CoreMLConfig::MelBackend::Torch ? "torch" : "cpu")
                          << "\n";
            }
        } catch (const c10::Error& err) {
            if (coreml_config_.verbose) {
                std::cerr << "Torch backend: failed to load model: " << err.what() << "\n";
            }
            return false;
        }
    }

    std::size_t frames = 0;
    std::vector<float> features;
    if (coreml_config_.mel_backend == CoreMLConfig::MelBackend::Torch) {
        std::string mel_error;
        const auto mel_start = std::chrono::steady_clock::now();
        features = compute_mel_features_torch(window,
                                              coreml_config_.sample_rate,
                                              coreml_config_,
                                              torch_state_->device,
                                              &frames,
                                              &mel_error);
        const auto mel_end = std::chrono::steady_clock::now();
        perf_.mel_ms +=
            std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
    } else {
        const auto mel_start = std::chrono::steady_clock::now();
        features = compute_mel_features(window, coreml_config_.sample_rate, coreml_config_, &frames);
        const auto mel_end = std::chrono::steady_clock::now();
        perf_.mel_ms +=
            std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
    }
    if (features.empty() || frames == 0) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: mel feature extraction failed.\n";
        }
        return false;
    }

    const std::size_t expected_frames = coreml_config_.fixed_frames;
    if (expected_frames > 0 && frames < expected_frames) {
        features.resize(expected_frames * coreml_config_.mel_bins, 0.0f);
        frames = expected_frames;
    }

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch_state_->device);
    torch::Tensor input =
        torch::from_blob(features.data(),
                         {1, static_cast<long long>(frames),
                          static_cast<long long>(coreml_config_.mel_bins)},
                         torch::kFloat32)
            .to(options)
            .clone();

    torch::IValue output;
    std::vector<torch::IValue> inputs;
    inputs.reserve(1);
    inputs.emplace_back(input);
    try {
        c10::InferenceMode inference_guard(true);
        const auto forward_start = std::chrono::steady_clock::now();
        output = torch_state_->module.forward(inputs);
        const auto forward_end = std::chrono::steady_clock::now();
        perf_.torch_forward_ms +=
            std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
    } catch (const c10::Error& err) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward failed: " << err.what() << "\n";
        }
        return false;
    } catch (const std::exception& err) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward exception: " << err.what() << "\n";
        }
        return false;
    } catch (...) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward unknown exception\n";
        }
        return false;
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
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: unexpected output signature.\n";
        }
        return false;
    }

    beat_tensor = torch::sigmoid(beat_tensor).to(torch::kCPU).flatten();
    if (downbeat_tensor.defined()) {
        downbeat_tensor = torch::sigmoid(downbeat_tensor).to(torch::kCPU).flatten();
    }

    const std::size_t total = static_cast<std::size_t>(beat_tensor.numel());
    beat->assign(total, 0.0f);
    downbeat->assign(total, 0.0f);
    const auto beat_accessor = beat_tensor.accessor<float, 1>();
    for (std::size_t i = 0; i < total; ++i) {
        (*beat)[i] = beat_accessor[static_cast<long long>(i)];
    }
    if (downbeat_tensor.defined() && downbeat_tensor.numel() == beat_tensor.numel()) {
        const auto downbeat_accessor = downbeat_tensor.accessor<float, 1>();
        for (std::size_t i = 0; i < total; ++i) {
            (*downbeat)[i] = downbeat_accessor[static_cast<long long>(i)];
        }
    }

    return true;
}

bool BeatitStream::infer_torch_windows(const std::vector<std::vector<float>>& windows,
                                       std::vector<std::vector<float>>* beats,
                                       std::vector<std::vector<float>>* downbeats) {
    if (!beats || !downbeats) {
        return false;
    }
    if (windows.empty()) {
        return true;
    }

    if (!torch_state_) {
        std::vector<float> dummy_beat;
        std::vector<float> dummy_downbeat;
        if (!infer_torch_window(windows.front(), &dummy_beat, &dummy_downbeat)) {
            return false;
        }
    }

    if (!torch_state_) {
        return false;
    }

    const std::size_t expected_frames = coreml_config_.fixed_frames;
    const std::size_t batch = windows.size();
    const std::size_t mel_bins = coreml_config_.mel_bins;

    std::vector<float> batch_buffer(batch * expected_frames * mel_bins, 0.0f);
    for (std::size_t b = 0; b < batch; ++b) {
        std::size_t frames = 0;
        std::vector<float> features;
        if (coreml_config_.mel_backend == CoreMLConfig::MelBackend::Torch) {
            std::string mel_error;
            const auto mel_start = std::chrono::steady_clock::now();
            features = compute_mel_features_torch(windows[b],
                                                  coreml_config_.sample_rate,
                                                  coreml_config_,
                                                  torch_state_->device,
                                                  &frames,
                                                  &mel_error);
            const auto mel_end = std::chrono::steady_clock::now();
            perf_.mel_ms +=
                std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
        } else {
            const auto mel_start = std::chrono::steady_clock::now();
            features = compute_mel_features(windows[b], coreml_config_.sample_rate, coreml_config_, &frames);
            const auto mel_end = std::chrono::steady_clock::now();
            perf_.mel_ms +=
                std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
        }
        if (features.empty() || frames == 0) {
            if (coreml_config_.verbose) {
                std::cerr << "Torch backend: mel feature extraction failed.\n";
            }
            return false;
        }
        if (expected_frames > 0 && frames < expected_frames) {
            features.resize(expected_frames * mel_bins, 0.0f);
            frames = expected_frames;
        }
        for (std::size_t f = 0; f < frames; ++f) {
            const std::size_t src = f * mel_bins;
            const std::size_t dst = (b * expected_frames + f) * mel_bins;
            std::copy(features.begin() + src,
                      features.begin() + src + mel_bins,
                      batch_buffer.begin() + dst);
        }
    }

    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch_state_->device);
    torch::Tensor input =
        torch::from_blob(batch_buffer.data(),
                         {static_cast<long long>(batch),
                          static_cast<long long>(expected_frames),
                          static_cast<long long>(mel_bins)},
                         torch::kFloat32)
            .to(options)
            .clone();

    torch::IValue output;
    std::vector<torch::IValue> inputs;
    inputs.reserve(1);
    inputs.emplace_back(input);
    try {
        c10::InferenceMode inference_guard(true);
        const auto forward_start = std::chrono::steady_clock::now();
        output = torch_state_->module.forward(inputs);
        const auto forward_end = std::chrono::steady_clock::now();
        perf_.torch_forward_ms +=
            std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
    } catch (const c10::Error& err) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward failed: " << err.what() << "\n";
        }
        return false;
    } catch (const std::exception& err) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward exception: " << err.what() << "\n";
        }
        return false;
    } catch (...) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward unknown exception\n";
        }
        return false;
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
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: unexpected output signature.\n";
        }
        return false;
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

    beats->assign(batch, {});
    downbeats->assign(batch, {});

    const auto beat_cpu = beat_tensor.contiguous();
    const auto beat_accessor = beat_cpu.accessor<float, 2>();
    for (std::size_t b = 0; b < batch; ++b) {
        (*beats)[b].assign(expected_frames, 0.0f);
        for (std::size_t i = 0; i < expected_frames; ++i) {
            (*beats)[b][i] = beat_accessor[static_cast<long long>(b)][static_cast<long long>(i)];
        }
    }

    if (downbeat_tensor.defined() && downbeat_tensor.numel() > 0) {
        const auto downbeat_cpu = downbeat_tensor.contiguous();
        const auto downbeat_accessor = downbeat_cpu.accessor<float, 2>();
        for (std::size_t b = 0; b < batch; ++b) {
            (*downbeats)[b].assign(expected_frames, 0.0f);
            for (std::size_t i = 0; i < expected_frames; ++i) {
                (*downbeats)[b][i] =
                    downbeat_accessor[static_cast<long long>(b)][static_cast<long long>(i)];
            }
        }
    }

    return true;
}
#endif

BeatitStream::BeatitStream(double sample_rate,
                           const CoreMLConfig& coreml_config,
                           bool enable_coreml)
    : sample_rate_(sample_rate),
      coreml_config_(coreml_config),
      coreml_enabled_(enable_coreml) {
    resampler_.ratio = coreml_config_.sample_rate / sample_rate_;
    if (coreml_config_.prepend_silence_seconds > 0.0 && sample_rate_ > 0.0) {
        prepend_samples_ = static_cast<std::size_t>(
            std::llround(coreml_config_.prepend_silence_seconds * sample_rate_));
    }
    if (coreml_config_.sample_rate > 0) {
        const double cutoff_hz = 150.0;
        const double dt = 1.0 / static_cast<double>(coreml_config_.sample_rate);
        const double rc = 1.0 / (2.0 * 3.141592653589793 * cutoff_hz);
        phase_energy_alpha_ = dt / (rc + dt);
    }
}

void BeatitStream::process_coreml_windows() {
    if (!coreml_enabled_ || coreml_config_.fixed_frames == 0) {
        return;
    }

    const std::size_t window_samples =
        coreml_config_.frame_size + (coreml_config_.fixed_frames - 1) * coreml_config_.hop_size;
    const std::size_t hop_samples = coreml_config_.window_hop_frames * coreml_config_.hop_size;

    while (resampled_buffer_.size() - resampled_offset_ >= window_samples) {
        std::vector<float> window(window_samples, 0.0f);
        const float* start_ptr = resampled_buffer_.data() + resampled_offset_;
        std::copy(start_ptr, start_ptr + window_samples, window.begin());

        CoreMLConfig local_config = coreml_config_;
        local_config.tempo_window_percent = 0.0f;
        local_config.prefer_double_time = false;

        const auto infer_start = std::chrono::steady_clock::now();
        CoreMLResult window_result = analyze_with_coreml(window,
                                                         local_config.sample_rate,
                                                         local_config,
                                                         0.0f);
        const auto infer_end = std::chrono::steady_clock::now();
        perf_.window_infer_ms +=
            std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        perf_.window_count += 1;
        if (window_result.beat_activation.size() > local_config.fixed_frames) {
            window_result.beat_activation.resize(local_config.fixed_frames);
        }
        if (window_result.downbeat_activation.size() > local_config.fixed_frames) {
            window_result.downbeat_activation.resize(local_config.fixed_frames);
        }

        const std::size_t needed = coreml_frame_offset_ + local_config.fixed_frames;
        if (coreml_beat_activation_.size() < needed) {
            coreml_beat_activation_.resize(needed, 0.0f);
            coreml_downbeat_activation_.resize(needed, 0.0f);
        }

        for (std::size_t i = 0; i < local_config.fixed_frames; ++i) {
            const std::size_t idx = coreml_frame_offset_ + i;
            if (i < window_result.beat_activation.size()) {
                coreml_beat_activation_[idx] = std::max(coreml_beat_activation_[idx],
                                                        window_result.beat_activation[i]);
            }
            if (i < window_result.downbeat_activation.size()) {
                coreml_downbeat_activation_[idx] = std::max(coreml_downbeat_activation_[idx],
                                                            window_result.downbeat_activation[i]);
            }
        }

        resampled_offset_ += hop_samples;
        coreml_frame_offset_ += coreml_config_.window_hop_frames;

        if (resampled_offset_ > window_samples) {
            resampled_buffer_.erase(resampled_buffer_.begin(),
                                    resampled_buffer_.begin() + static_cast<long>(resampled_offset_));
            resampled_offset_ = 0;
        }
    }
}

void BeatitStream::process_torch_windows() {
    if (!coreml_enabled_ || coreml_config_.fixed_frames == 0) {
        return;
    }
#if !defined(BEATIT_USE_TORCH)
    if (coreml_config_.verbose) {
        std::cerr << "Torch backend not enabled in this build.\n";
    }
    return;
#else
    if (coreml_config_.backend != CoreMLConfig::Backend::Torch) {
        return;
    }

    const std::size_t window_samples =
        coreml_config_.frame_size + (coreml_config_.fixed_frames - 1) * coreml_config_.hop_size;
    const std::size_t hop_samples = coreml_config_.window_hop_frames * coreml_config_.hop_size;
    const std::size_t border = std::min(coreml_config_.window_border_frames,
                                        coreml_config_.fixed_frames / 2);
    const std::size_t batch_size = std::max<std::size_t>(1, coreml_config_.torch_batch_size);

    while (resampled_buffer_.size() - resampled_offset_ >= window_samples) {
        const std::size_t available =
            (resampled_buffer_.size() - resampled_offset_ - window_samples) / hop_samples + 1;
        const std::size_t batch_count = std::min(batch_size, available);

        std::vector<std::vector<float>> windows;
        windows.reserve(batch_count);
        for (std::size_t w = 0; w < batch_count; ++w) {
            std::vector<float> window(window_samples, 0.0f);
            const float* start_ptr =
                resampled_buffer_.data() + resampled_offset_ + w * hop_samples;
            std::copy(start_ptr, start_ptr + window_samples, window.begin());
            windows.push_back(std::move(window));
        }

        std::vector<std::vector<float>> beat_activations;
        std::vector<std::vector<float>> downbeat_activations;

        const auto infer_start = std::chrono::steady_clock::now();
        bool ok = infer_torch_windows(windows, &beat_activations, &downbeat_activations);
        const auto infer_end = std::chrono::steady_clock::now();
        perf_.window_infer_ms +=
            std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        perf_.window_count += batch_count;
        if (!ok) {
            return;
        }

        for (std::size_t w = 0; w < batch_count; ++w) {
            auto& beat_activation = beat_activations[w];
            auto& downbeat_activation = downbeat_activations[w];

            if (beat_activation.size() > coreml_config_.fixed_frames) {
                beat_activation.resize(coreml_config_.fixed_frames);
            }
            if (downbeat_activation.size() > coreml_config_.fixed_frames) {
                downbeat_activation.resize(coreml_config_.fixed_frames);
            }

            const std::size_t window_offset =
                coreml_frame_offset_ + w * coreml_config_.window_hop_frames;
            const std::size_t needed = window_offset + coreml_config_.fixed_frames;
            if (coreml_beat_activation_.size() < needed) {
                coreml_beat_activation_.resize(needed, 0.0f);
                coreml_downbeat_activation_.resize(needed, 0.0f);
            }

            for (std::size_t i = 0; i < coreml_config_.fixed_frames; ++i) {
                if (i < border || i >= coreml_config_.fixed_frames - border) {
                    continue;
                }
                const std::size_t idx = window_offset + i;
                if (i < beat_activation.size()) {
                    coreml_beat_activation_[idx] =
                        std::max(coreml_beat_activation_[idx], beat_activation[i]);
                }
                if (i < downbeat_activation.size()) {
                    coreml_downbeat_activation_[idx] =
                        std::max(coreml_downbeat_activation_[idx], downbeat_activation[i]);
                }
            }
        }

        resampled_offset_ += hop_samples * batch_count;
        coreml_frame_offset_ += coreml_config_.window_hop_frames * batch_count;

        if (resampled_offset_ > window_samples) {
            resampled_buffer_.erase(resampled_buffer_.begin(),
                                    resampled_buffer_.begin() + static_cast<long>(resampled_offset_));
            resampled_offset_ = 0;
        }
    }
#endif
}

void BeatitStream::push(const float* samples, std::size_t count) {
    if (!samples || count == 0 || !coreml_enabled_) {
        return;
    }

    if (coreml_config_.analysis_start_seconds > 0.0 && sample_rate_ > 0.0) {
        const auto start_limit = static_cast<std::size_t>(
            std::llround(coreml_config_.analysis_start_seconds * sample_rate_));
        if (total_seen_samples_ < start_limit) {
            const std::size_t remaining = start_limit - total_seen_samples_;
            if (count <= remaining) {
                total_seen_samples_ += count;
                return;
            }
            samples += remaining;
            count -= remaining;
            total_seen_samples_ += remaining;
        }
    }
    total_seen_samples_ += count;

    if (!prepend_done_ && prepend_samples_ > 0) {
        const std::size_t before_size = resampled_buffer_.size();
        std::vector<float> silence(prepend_samples_, 0.0f);
        resampler_.push(silence.data(), silence.size(), &resampled_buffer_);
        const std::size_t after_size = resampled_buffer_.size();
        if (after_size > before_size && coreml_config_.hop_size > 0) {
            for (std::size_t i = before_size; i < after_size; ++i) {
                const double input = resampled_buffer_[i];
                phase_energy_state_ += phase_energy_alpha_ * (input - phase_energy_state_);
                phase_energy_sum_sq_ += phase_energy_state_ * phase_energy_state_;
                phase_energy_sample_count_++;
                if (phase_energy_sample_count_ >= coreml_config_.hop_size) {
                    const double rms = std::sqrt(
                        phase_energy_sum_sq_ / static_cast<double>(phase_energy_sample_count_));
                    coreml_phase_energy_.push_back(static_cast<float>(rms));
                    phase_energy_sum_sq_ = 0.0;
                    phase_energy_sample_count_ = 0;
                }
            }
        }
        prepend_done_ = true;
    }

    if (coreml_config_.max_analysis_seconds > 0.0 && sample_rate_ > 0.0) {
        const auto limit = static_cast<std::size_t>(
            std::llround(coreml_config_.max_analysis_seconds * sample_rate_));
        if (total_input_samples_ >= limit) {
            return;
        }
        const std::size_t remaining = limit - total_input_samples_;
        if (count > remaining) {
            count = remaining;
        }
        if (count == 0) {
            return;
        }
    }

    const float rms_threshold = 0.001f;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double value = samples[i];
        sum_sq += value * value;
    }
    const double rms = count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0;
    if (rms >= rms_threshold) {
        last_active_sample_ = total_input_samples_ + count - 1;
        has_active_sample_ = true;
    }
    total_input_samples_ += count;

    const std::size_t before_size = resampled_buffer_.size();
    const auto resample_start = std::chrono::steady_clock::now();
    resampler_.push(samples, count, &resampled_buffer_);
    const auto resample_end = std::chrono::steady_clock::now();
    perf_.resample_ms +=
        std::chrono::duration<double, std::milli>(resample_end - resample_start).count();
    const std::size_t after_size = resampled_buffer_.size();
    if (after_size > before_size && coreml_config_.hop_size > 0) {
        for (std::size_t i = before_size; i < after_size; ++i) {
            const double input = resampled_buffer_[i];
            phase_energy_state_ += phase_energy_alpha_ * (input - phase_energy_state_);
            phase_energy_sum_sq_ += phase_energy_state_ * phase_energy_state_;
            phase_energy_sample_count_++;
            if (phase_energy_sample_count_ >= coreml_config_.hop_size) {
                const double rms =
                    std::sqrt(phase_energy_sum_sq_ / static_cast<double>(phase_energy_sample_count_));
                coreml_phase_energy_.push_back(static_cast<float>(rms));
                phase_energy_sum_sq_ = 0.0;
                phase_energy_sample_count_ = 0;
            }
        }
    }
    const auto process_start = std::chrono::steady_clock::now();
    if (coreml_config_.backend == CoreMLConfig::Backend::Torch) {
        process_torch_windows();
    } else {
        process_coreml_windows();
    }
    const auto process_end = std::chrono::steady_clock::now();
    perf_.process_ms +=
        std::chrono::duration<double, std::milli>(process_end - process_start).count();
}

AnalysisResult BeatitStream::finalize() {
    AnalysisResult result;

    const auto finalize_start = std::chrono::steady_clock::now();
    if (!coreml_enabled_) {
        return result;
    }

    if (coreml_config_.fixed_frames > 0 && coreml_config_.pad_final_window) {
        const std::size_t window_samples =
            coreml_config_.frame_size + (coreml_config_.fixed_frames - 1) * coreml_config_.hop_size;
        const std::size_t available =
            resampled_buffer_.size() > resampled_offset_
                ? resampled_buffer_.size() - resampled_offset_
                : 0;
        if (available > 0 && window_samples > 0) {
            std::vector<float> window(window_samples, 0.0f);
            const float* start_ptr = resampled_buffer_.data() + resampled_offset_;
            std::copy(start_ptr, start_ptr + std::min(available, window_samples), window.begin());

            CoreMLConfig local_config = coreml_config_;
            local_config.tempo_window_percent = 0.0f;
            local_config.prefer_double_time = false;

            const auto infer_start = std::chrono::steady_clock::now();
            std::vector<float> beat_activation;
            std::vector<float> downbeat_activation;
            bool ok = false;
            if (coreml_config_.backend == CoreMLConfig::Backend::Torch) {
#if defined(BEATIT_USE_TORCH)
                ok = infer_torch_window(window, &beat_activation, &downbeat_activation);
#else
                ok = false;
#endif
            } else {
                CoreMLResult window_result = analyze_with_coreml(window,
                                                                 local_config.sample_rate,
                                                                 local_config,
                                                                 0.0f);
                beat_activation = std::move(window_result.beat_activation);
                downbeat_activation = std::move(window_result.downbeat_activation);
                ok = true;
            }
            const auto infer_end = std::chrono::steady_clock::now();
            perf_.finalize_infer_ms +=
                std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
            if (!ok) {
                return result;
            }
            if (beat_activation.size() > local_config.fixed_frames) {
                beat_activation.resize(local_config.fixed_frames);
            }
            if (downbeat_activation.size() > local_config.fixed_frames) {
                downbeat_activation.resize(local_config.fixed_frames);
            }

            const std::size_t needed = coreml_frame_offset_ + local_config.fixed_frames;
            if (coreml_beat_activation_.size() < needed) {
                coreml_beat_activation_.resize(needed, 0.0f);
                coreml_downbeat_activation_.resize(needed, 0.0f);
            }

            for (std::size_t i = 0; i < local_config.fixed_frames; ++i) {
                const std::size_t idx = coreml_frame_offset_ + i;
                if (i < beat_activation.size()) {
                    coreml_beat_activation_[idx] =
                        std::max(coreml_beat_activation_[idx], beat_activation[i]);
                }
                if (i < downbeat_activation.size()) {
                    coreml_downbeat_activation_[idx] =
                        std::max(coreml_downbeat_activation_[idx], downbeat_activation[i]);
                }
            }
        }
    }

    if (coreml_beat_activation_.empty()) {
        return result;
    }

    CoreMLConfig base_config = coreml_config_;
    base_config.tempo_window_percent = 0.0f;
    base_config.prefer_double_time = false;
    base_config.synthetic_fill = false;
    base_config.dbn_window_seconds = 0.0;

    std::size_t last_active_frame = 0;
    std::size_t full_frame_count = 0;
    if (coreml_config_.hop_size > 0 && sample_rate_ > 0.0) {
        const double ratio = coreml_config_.sample_rate / sample_rate_;
        const double total_pos = static_cast<double>(total_seen_samples_) * ratio;
        full_frame_count =
            static_cast<std::size_t>(std::llround(total_pos / coreml_config_.hop_size));
        if (coreml_config_.disable_silence_trimming) {
            last_active_frame = full_frame_count;
        } else if (has_active_sample_) {
            const double sample_pos = static_cast<double>(last_active_sample_) * ratio;
            last_active_frame =
                static_cast<std::size_t>(std::llround(sample_pos / coreml_config_.hop_size));
        }
    }

    const auto postprocess_start = std::chrono::steady_clock::now();
    CoreMLResult base = postprocess_coreml_activations(coreml_beat_activation_,
                                                      coreml_downbeat_activation_,
                                                      &coreml_phase_energy_,
                                                      base_config,
                                                      sample_rate_,
                                                      0.0f,
                                                      last_active_frame,
                                                      full_frame_count);
    const float bpm_min = std::max(1.0f, coreml_config_.min_bpm);
    const float bpm_max = std::max(bpm_min + 1.0f, coreml_config_.max_bpm);
    const float peaks_bpm_raw =
        estimate_bpm_from_activation_peaks_local(coreml_beat_activation_, coreml_config_, sample_rate_);
    const float autocorr_bpm_raw =
        estimate_bpm_from_activation_autocorr_local(coreml_beat_activation_, coreml_config_, sample_rate_);
    const float comb_bpm_raw =
        estimate_bpm_from_activation_comb(coreml_beat_activation_, coreml_config_, sample_rate_);
    const float beats_bpm_raw = estimate_bpm_from_beats_local(base.beat_sample_frames, sample_rate_);
    const float peaks_bpm = normalize_bpm_to_range_local(peaks_bpm_raw, bpm_min, bpm_max);
    const float autocorr_bpm = normalize_bpm_to_range_local(autocorr_bpm_raw, bpm_min, bpm_max);
    const float comb_bpm = normalize_bpm_to_range_local(comb_bpm_raw, bpm_min, bpm_max);
    const float beats_bpm = normalize_bpm_to_range_local(beats_bpm_raw, bpm_min, bpm_max);
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

    const float candidate_bpm = choose_candidate_bpm(peaks_bpm, autocorr_bpm, comb_bpm, beats_bpm);
    const double prior_bpm = tempo_reference_valid_ ? tempo_reference_bpm_ : 0.0;
    float reference_bpm = candidate_bpm;
    std::string tempo_state = "init";
    float consensus_ratio = 0.0f;
    float prior_ratio = 0.0f;

    if (candidate_bpm > 0.0f) {
        std::vector<float> anchors;
        anchors.reserve(4);
        if (peaks_bpm > 0.0f) {
            anchors.push_back(peaks_bpm);
        }
        if (autocorr_bpm > 0.0f) {
            anchors.push_back(autocorr_bpm);
        }
        if (comb_bpm > 0.0f) {
            anchors.push_back(comb_bpm);
        }
        if (beats_bpm > 0.0f) {
            anchors.push_back(beats_bpm);
        }
        const float consensus_tol = 0.02f;
        std::size_t consensus = 0;
        for (float value : anchors) {
            if (std::abs(value - candidate_bpm) / std::max(candidate_bpm, 1e-6f) <= consensus_tol) {
                ++consensus;
            }
        }
        consensus_ratio = static_cast<float>(consensus) / std::max<std::size_t>(1, anchors.size());

        if (tempo_reference_valid_ && prior_bpm > 0.0) {
            prior_ratio =
                static_cast<float>(std::abs(candidate_bpm - prior_bpm) / prior_bpm);
            const float hold_tol = 0.02f;
            const float switch_tol = std::max(hold_tol, coreml_config_.dbn_interval_tolerance);
            if (prior_ratio <= hold_tol) {
                reference_bpm =
                    static_cast<float>(0.7 * prior_bpm + 0.3 * candidate_bpm);
                tempo_state = "blend";
            } else if (prior_ratio <= switch_tol || consensus >= 2) {
                reference_bpm = candidate_bpm;
                tempo_state = "switch";
            } else {
                reference_bpm = static_cast<float>(prior_bpm);
                tempo_state = "hold";
            }
        }
    }

    if (reference_bpm > 0.0f) {
        tempo_reference_bpm_ = reference_bpm;
        tempo_reference_valid_ = true;
    }
    if (coreml_config_.verbose) {
        std::cerr << "Tempo anchor: peaks=" << peaks_bpm
                  << " autocorr=" << autocorr_bpm
                  << " comb=" << comb_bpm
                  << " beats=" << beats_bpm
                  << " chosen=" << reference_bpm
                  << " prior=" << prior_bpm
                  << " state=" << tempo_state
                  << " ratio=" << prior_ratio
                  << " consensus=" << consensus_ratio
                  << "\n";
    }

    CoreMLResult final_result = postprocess_coreml_activations(coreml_beat_activation_,
                                                              coreml_downbeat_activation_,
                                                              &coreml_phase_energy_,
                                                              coreml_config_,
                                                              sample_rate_,
                                                              reference_bpm,
                                                              last_active_frame,
                                                              full_frame_count);
    const auto postprocess_end = std::chrono::steady_clock::now();
    perf_.postprocess_ms +=
        std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();

    result.coreml_beat_activation = std::move(final_result.beat_activation);
    result.coreml_downbeat_activation = std::move(final_result.downbeat_activation);
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
    result.coreml_phase_energy = std::move(coreml_phase_energy_);
    const auto& bpm_frames = result.coreml_beat_projected_sample_frames.empty()
        ? result.coreml_beat_sample_frames
        : result.coreml_beat_projected_sample_frames;
    // Keep reported BPM consistent with the returned beat grid.
    float estimated_bpm =
        normalize_bpm_to_range_local(estimate_bpm_from_beats_local(bpm_frames, sample_rate_),
                                     bpm_min,
                                     bpm_max);
    if (!(estimated_bpm > 0.0f)) {
        const float anchored_bpm = normalize_bpm_to_range_local(reference_bpm, bpm_min, bpm_max);
        if (anchored_bpm > 0.0f) {
            estimated_bpm = anchored_bpm;
        }
    }
    result.estimated_bpm = estimated_bpm;

    if (prepend_samples_ > 0) {
        for (auto& frame : result.coreml_beat_sample_frames) {
            frame = frame > prepend_samples_ ? frame - prepend_samples_ : 0;
        }
        for (auto& frame : result.coreml_beat_projected_sample_frames) {
            frame = frame > prepend_samples_ ? frame - prepend_samples_ : 0;
        }
    }

    auto dedupe_monotonic = [](std::vector<unsigned long long>& samples,
                               std::vector<unsigned long long>* feature_frames,
                               std::vector<float>* strengths) {
        if (samples.empty()) {
            return;
        }
        std::size_t write = 1;
        unsigned long long last = samples[0];
        for (std::size_t i = 1; i < samples.size(); ++i) {
            const unsigned long long current = samples[i];
            if (current <= last) {
                continue;
            }
            samples[write] = current;
            if (feature_frames && i < feature_frames->size()) {
                (*feature_frames)[write] = (*feature_frames)[i];
            }
            if (strengths && i < strengths->size()) {
                (*strengths)[write] = (*strengths)[i];
            }
            last = current;
            ++write;
        }
        samples.resize(write);
        if (feature_frames && feature_frames->size() >= write) {
            feature_frames->resize(write);
        }
        if (strengths && strengths->size() >= write) {
            strengths->resize(write);
        }
    };

    dedupe_monotonic(result.coreml_beat_sample_frames,
                     &result.coreml_beat_feature_frames,
                     &result.coreml_beat_strengths);
    dedupe_monotonic(result.coreml_beat_projected_sample_frames,
                     &result.coreml_beat_projected_feature_frames,
                     nullptr);

    if (!result.coreml_beat_projected_feature_frames.empty() &&
        result.coreml_beat_projected_feature_frames.size() >= 2) {
        std::vector<unsigned long long> projected_downbeats;
        projected_downbeats.reserve(result.coreml_downbeat_projected_feature_frames.size());
        std::size_t down_idx = 0;
        for (unsigned long long frame : result.coreml_downbeat_projected_feature_frames) {
            while (down_idx < result.coreml_beat_projected_feature_frames.size() &&
                   result.coreml_beat_projected_feature_frames[down_idx] < frame) {
                ++down_idx;
            }
            if (down_idx < result.coreml_beat_projected_feature_frames.size() &&
                result.coreml_beat_projected_feature_frames[down_idx] == frame) {
                projected_downbeats.push_back(frame);
            }
        }
        result.coreml_downbeat_projected_feature_frames = std::move(projected_downbeats);
    }
    const auto marker_start = std::chrono::steady_clock::now();
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
                                 sample_rate_,
                                 coreml_config_);
    const auto marker_end = std::chrono::steady_clock::now();
    perf_.marker_ms +=
        std::chrono::duration<double, std::milli>(marker_end - marker_start).count();

    const auto finalize_end = std::chrono::steady_clock::now();
    perf_.finalize_ms =
        std::chrono::duration<double, std::milli>(finalize_end - finalize_start).count();

    if (coreml_config_.profile) {
        std::cerr << "Timing(stream): resample=" << perf_.resample_ms
                  << "ms process=" << perf_.process_ms
                  << "ms mel=" << perf_.mel_ms
                  << "ms torch=" << perf_.torch_forward_ms
                  << "ms window_infer=" << perf_.window_infer_ms
                  << "ms windows=" << perf_.window_count
                  << " finalize_infer=" << perf_.finalize_infer_ms
                  << "ms postprocess=" << perf_.postprocess_ms
                  << "ms markers=" << perf_.marker_ms
                  << "ms total_finalize=" << perf_.finalize_ms
                  << "ms\n";
    }

    return result;
}

} // namespace beatit
