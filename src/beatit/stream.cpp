//
//  stream.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/stream.h"
#include "beatit/stream_activation_accumulator.h"
#include "beatit/stream_inference_backend.h"
#include "beatit/stream_sparse.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

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

    result = detail::analyze_sparse_probe_window(
        original_config,
        sample_rate_,
        total_duration_seconds,
        provider,
        run_probe,
        estimate_bpm_from_beats_local,
        normalize_bpm_to_range_local);

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

BeatitStream::BeatitStream(double sample_rate,
                           const CoreMLConfig& coreml_config,
                           bool enable_coreml)
    : sample_rate_(sample_rate),
      coreml_config_(coreml_config),
      coreml_enabled_(enable_coreml),
      inference_backend_(detail::make_stream_inference_backend(coreml_config)) {
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
    if (!coreml_enabled_ || coreml_config_.fixed_frames == 0 || !inference_backend_) {
        return;
    }

    const std::size_t window_samples =
        coreml_config_.frame_size + (coreml_config_.fixed_frames - 1) * coreml_config_.hop_size;
    const std::size_t hop_samples = coreml_config_.window_hop_frames * coreml_config_.hop_size;

    while (resampled_buffer_.size() - resampled_offset_ >= window_samples) {
        std::vector<float> window(window_samples, 0.0f);
        const float* start_ptr = resampled_buffer_.data() + resampled_offset_;
        std::copy(start_ptr, start_ptr + window_samples, window.begin());

        const auto infer_start = std::chrono::steady_clock::now();
        std::vector<float> beat_activation;
        std::vector<float> downbeat_activation;
        detail::StreamInferenceTiming timing;
        const bool ok = inference_backend_->infer_window(window,
                                                         coreml_config_,
                                                         &beat_activation,
                                                         &downbeat_activation,
                                                         &timing);
        const auto infer_end = std::chrono::steady_clock::now();
        perf_.mel_ms += timing.mel_ms;
        perf_.torch_forward_ms += timing.torch_forward_ms;
        perf_.window_infer_ms +=
            std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        perf_.window_count += 1;
        if (!ok) {
            return;
        }
        detail::trim_activation_to_frames(&beat_activation, coreml_config_.fixed_frames);
        detail::trim_activation_to_frames(&downbeat_activation, coreml_config_.fixed_frames);

        detail::merge_window_activations(&coreml_beat_activation_,
                                         &coreml_downbeat_activation_,
                                         coreml_frame_offset_,
                                         coreml_config_.fixed_frames,
                                         beat_activation,
                                         downbeat_activation,
                                         0);

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
    if (!coreml_enabled_ || coreml_config_.fixed_frames == 0 || !inference_backend_) {
        return;
    }
    if (coreml_config_.backend != CoreMLConfig::Backend::Torch) {
        return;
    }

    const std::size_t window_samples =
        coreml_config_.frame_size + (coreml_config_.fixed_frames - 1) * coreml_config_.hop_size;
    const std::size_t hop_samples = coreml_config_.window_hop_frames * coreml_config_.hop_size;
    const std::size_t border = std::min(inference_backend_->border_frames(coreml_config_),
                                        coreml_config_.fixed_frames / 2);
    const std::size_t batch_size = std::max<std::size_t>(
        1, inference_backend_->max_batch_size(coreml_config_));

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
        detail::StreamInferenceTiming timing;
        bool ok = inference_backend_->infer_windows(windows,
                                                    coreml_config_,
                                                    &beat_activations,
                                                    &downbeat_activations,
                                                    &timing);
        const auto infer_end = std::chrono::steady_clock::now();
        perf_.mel_ms += timing.mel_ms;
        perf_.torch_forward_ms += timing.torch_forward_ms;
        perf_.window_infer_ms +=
            std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        perf_.window_count += batch_count;
        if (!ok) {
            return;
        }

        for (std::size_t w = 0; w < batch_count; ++w) {
            auto& beat_activation = beat_activations[w];
            auto& downbeat_activation = downbeat_activations[w];

            detail::trim_activation_to_frames(&beat_activation, coreml_config_.fixed_frames);
            detail::trim_activation_to_frames(&downbeat_activation, coreml_config_.fixed_frames);

            const std::size_t window_offset =
                coreml_frame_offset_ + w * coreml_config_.window_hop_frames;
            detail::merge_window_activations(&coreml_beat_activation_,
                                             &coreml_downbeat_activation_,
                                             window_offset,
                                             coreml_config_.fixed_frames,
                                             beat_activation,
                                             downbeat_activation,
                                             border);
        }

        resampled_offset_ += hop_samples * batch_count;
        coreml_frame_offset_ += coreml_config_.window_hop_frames * batch_count;

        if (resampled_offset_ > window_samples) {
            resampled_buffer_.erase(resampled_buffer_.begin(),
                                    resampled_buffer_.begin() + static_cast<long>(resampled_offset_));
            resampled_offset_ = 0;
        }
    }
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
            detail::StreamInferenceTiming timing;
            const bool ok = inference_backend_ &&
                inference_backend_->infer_window(window,
                                                 local_config,
                                                 &beat_activation,
                                                 &downbeat_activation,
                                                 &timing);
            const auto infer_end = std::chrono::steady_clock::now();
            perf_.mel_ms += timing.mel_ms;
            perf_.torch_forward_ms += timing.torch_forward_ms;
            perf_.finalize_infer_ms +=
                std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
            if (!ok) {
                return result;
            }
            detail::trim_activation_to_frames(&beat_activation, local_config.fixed_frames);
            detail::trim_activation_to_frames(&downbeat_activation, local_config.fixed_frames);
            detail::merge_window_activations(&coreml_beat_activation_,
                                             &coreml_downbeat_activation_,
                                             coreml_frame_offset_,
                                             local_config.fixed_frames,
                                             beat_activation,
                                             downbeat_activation,
                                             0);
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
