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
    if (coreml_config_.max_analysis_seconds <= 0.0) {
        const bool sparse_dynamic =
            coreml_config_.sparse_probe_mode &&
            coreml_config_.use_logit_consensus &&
            coreml_config_.dbn_window_intro_mid_outro &&
            coreml_config_.dbn_window_seconds > 0.0;
        if (!sparse_dynamic) {
            return false;
        }
        if (start_seconds) {
            *start_seconds = 0.0;
        }
        if (duration_seconds) {
            *duration_seconds = coreml_config_.dbn_window_seconds;
        }
        return true;
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
        coreml_config_.analysis_start_seconds = 0.0;
        coreml_config_.dbn_window_start_seconds = 0.0;
        coreml_config_.max_analysis_seconds = 0.0;
        if (forced_reference_bpm > 0.0) {
            coreml_config_.dbn_window_consensus = true;
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
        original_config.use_logit_consensus &&
        original_config.dbn_window_intro_mid_outro &&
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

    std::vector<double> probe_starts;
    probe_starts.push_back(clamp_start(std::max(0.0, original_config.dbn_tempo_anchor_intro_seconds)));
    probe_starts.push_back(clamp_start(total - 60.0 - probe_duration));
    if (probe_starts.size() >= 2 && std::abs(probe_starts[1] - probe_starts[0]) < 1.0) {
        probe_starts[1] = clamp_start(total * 0.5 - probe_duration * 0.5);
    }

    struct ProbeResult {
        double start = 0.0;
        AnalysisResult analysis;
        double bpm = 0.0;
        double conf = 0.0;
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
        static const double kModes[] = {0.5, 1.0, 2.0, 3.0, (2.0 / 3.0), 1.5};
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

    std::vector<ProbeResult> probes;
    probes.reserve(3);
    for (double s : probe_starts) {
        ProbeResult p;
        p.start = s;
        p.analysis = run_probe(s, probe_duration);
        p.bpm = p.analysis.estimated_bpm;
        p.conf = estimate_probe_confidence(p.analysis);
        probes.push_back(std::move(p));
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
            for (const auto& p : values) {
                const auto modes = expand_modes(p.bpm, p.conf);
                double best_local = std::numeric_limits<double>::infinity();
                for (const auto& m : modes) {
                    best_local = std::min(best_local, relative_diff(cand.bpm, m.bpm) * m.penalty);
                }
                score += best_local / std::max(1e-6, p.conf);
            }
            score *= cand.penalty;
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
            ProbeResult p;
            p.start = clamp_start(total * 0.5 - probe_duration * 0.5);
            p.analysis = run_probe(p.start, probe_duration);
            p.bpm = p.analysis.estimated_bpm;
            p.conf = estimate_probe_confidence(p.analysis);
            probes.push_back(std::move(p));
            consensus_bpm = consensus_from_probes(probes);
        }
    }

    std::vector<double> probe_mode_errors(probes.size(), 1.0);
    bool have_consensus = consensus_bpm > 0.0;
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

    const double anchor_start = clamp_start(
        std::max(0.0, original_config.dbn_window_start_seconds));
    double best_mode_error = std::numeric_limits<double>::infinity();
    for (double e : probe_mode_errors) {
        best_mode_error = std::min(best_mode_error, e);
    }
    const double mode_slack = 0.005;
    std::size_t selected_index = 0;
    double selected_score = std::numeric_limits<double>::infinity();
    IntroPhaseMetrics selected_intro_metrics;
    for (std::size_t i = 0; i < probes.size(); ++i) {
        if (probe_mode_errors[i] > best_mode_error + mode_slack) {
            continue;
        }
        const double start_penalty = std::abs(probes[i].start - anchor_start) /
            std::max(1.0, probe_duration);
        const IntroPhaseMetrics intro_metrics = measure_intro_phase(
            probes[i].analysis,
            have_consensus ? consensus_bpm : probes[i].bpm);
        const double phase_penalty =
            std::isfinite(intro_metrics.median_abs_ms) ? (intro_metrics.median_abs_ms / 40.0) : 10.0;
        const double confidence_penalty = 1.0 / std::max(1e-6, probes[i].conf);
        const double score = confidence_penalty + (start_penalty * 0.25) + phase_penalty;
        if (score < selected_score) {
            selected_score = score;
            selected_index = i;
            selected_intro_metrics = intro_metrics;
        }
    }

    // Sparse mode returns directly from the best probe to avoid an extra anchor pass.
    result = probes[selected_index].analysis;
    const double selected_mode_error = probe_mode_errors[selected_index];
    const bool low_confidence =
        (selected_mode_error > 0.015) ||
        (probes[selected_index].conf < 0.70) ||
        (std::isfinite(selected_intro_metrics.median_abs_ms) &&
         selected_intro_metrics.median_abs_ms > 90.0) ||
        (std::isfinite(selected_intro_metrics.odd_even_gap_ms) &&
         selected_intro_metrics.odd_even_gap_ms > 180.0);
    if (low_confidence) {
        const double repair_bpm = have_consensus
            ? consensus_bpm
            : (probes[selected_index].bpm > 0.0 ? probes[selected_index].bpm : 0.0);
        result = run_probe(anchor_start, probe_duration, repair_bpm);
    }
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
                  << " selected_start=" << probes[selected_index].start
                  << " selected_mode_err=" << selected_mode_error
                  << " selected_conf=" << probes[selected_index].conf
                  << " selected_intro_abs_ms=" << selected_intro_metrics.median_abs_ms
                  << " selected_odd_even_gap_ms=" << selected_intro_metrics.odd_even_gap_ms
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
    base_config.dbn_window_intro_mid_outro = false;
    base_config.dbn_window_consensus = false;
    base_config.dbn_window_stitch = false;

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
