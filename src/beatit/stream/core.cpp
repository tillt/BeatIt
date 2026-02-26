//
//  core.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/stream.h"
#include "beatit/inference/window_merge.h"
#include "beatit/inference/backend.h"
#include "beatit/logging.hpp"
#include "beatit/sparse/probe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace beatit {

BeatitStream::~BeatitStream() = default;

void BeatitStream::reset_state(bool reset_tempo_anchor) {
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
    if (reset_tempo_anchor) {
        tempo_reference_bpm_ = 0.0;
        tempo_reference_valid_ = false;
    }
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

    BeatitConfig original_config = coreml_config_;
    auto run_probe = [&](double probe_start,
                         double probe_duration,
                         double forced_reference_bpm = 0.0) -> AnalysisResult {
        reset_state(true);
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
        estimate_bpm_from_beats,
        normalize_bpm_to_range);

    coreml_config_ = original_config;
    return result;
}

void BeatitStream::LinearResampler::push(const float* input,
                                         std::size_t count,
                                         std::vector<float>& output) {
    if (count == 0) {
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
        output.push_back(static_cast<float>((1.0 - frac) * a + frac * b));
        src_index += step;
    }

    const std::size_t drop = static_cast<std::size_t>(src_index);
    if (drop > 0) {
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<long>(drop));
        src_index -= static_cast<double>(drop);
    }
}

BeatitStream::BeatitStream(double sample_rate,
                           const BeatitConfig& coreml_config,
                           bool enable_coreml)
    : sample_rate_(sample_rate),
      coreml_config_(coreml_config),
      coreml_enabled_(enable_coreml),
      inference_backend_(detail::make_inference_backend(coreml_config)) {
    set_log_verbosity_from_config(coreml_config_);

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
        detail::InferenceTiming timing;
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
            BEATIT_LOG_ERROR("Stream inference failed in process_coreml_windows."
                             << " backend=" << static_cast<int>(coreml_config_.backend)
                             << " frame_offset=" << coreml_frame_offset_);
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
    if (coreml_config_.backend != BeatitConfig::Backend::Torch) {
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
        detail::InferenceTiming timing;
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
            BEATIT_LOG_ERROR("Stream inference failed in process_torch_windows."
                             << " batch_count=" << batch_count
                             << " frame_offset=" << coreml_frame_offset_);
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

void BeatitStream::accumulate_phase_energy(std::size_t begin_sample, std::size_t end_sample) {
    if (coreml_config_.hop_size == 0 || end_sample <= begin_sample) {
        return;
    }
    for (std::size_t i = begin_sample; i < end_sample; ++i) {
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
        resampler_.push(silence.data(), silence.size(), resampled_buffer_);
        const std::size_t after_size = resampled_buffer_.size();
        accumulate_phase_energy(before_size, after_size);
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
    resampler_.push(samples, count, resampled_buffer_);
    const auto resample_end = std::chrono::steady_clock::now();
    perf_.resample_ms +=
        std::chrono::duration<double, std::milli>(resample_end - resample_start).count();
    const std::size_t after_size = resampled_buffer_.size();
    accumulate_phase_energy(before_size, after_size);
    const auto process_start = std::chrono::steady_clock::now();
    if (coreml_config_.backend == BeatitConfig::Backend::Torch) {
        process_torch_windows();
    } else {
        process_coreml_windows();
    }
    const auto process_end = std::chrono::steady_clock::now();
    perf_.process_ms +=
        std::chrono::duration<double, std::milli>(process_end - process_start).count();
}

} // namespace beatit
