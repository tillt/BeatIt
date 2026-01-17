//
//  stream.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/stream.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace beatit {
namespace {

float estimate_bpm_from_beats(const std::vector<unsigned long long>& beat_samples,
                              double sample_rate) {
    if (beat_samples.size() < 2 || sample_rate <= 0.0) {
        return 0.0f;
    }

    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 1; i < beat_samples.size(); ++i) {
        const unsigned long long prev = beat_samples[i - 1];
        const unsigned long long next = beat_samples[i];
        if (next > prev) {
            const double interval = static_cast<double>(next - prev) / sample_rate;
            if (interval > 0.0) {
                sum += 60.0 / interval;
                ++count;
            }
        }
    }

    if (count == 0) {
        return 0.0f;
    }

    return static_cast<float>(sum / static_cast<double>(count));
}

} // namespace

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
      coreml_enabled_(enable_coreml) {
    resampler_.ratio = coreml_config_.sample_rate / sample_rate_;
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

        CoreMLResult window_result = analyze_with_coreml(window,
                                                         local_config.sample_rate,
                                                         local_config,
                                                         0.0f);
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

void BeatitStream::push(const float* samples, std::size_t count) {
    if (!samples || count == 0 || !coreml_enabled_) {
        return;
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
    resampler_.push(samples, count, &resampled_buffer_);
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
    process_coreml_windows();
}

AnalysisResult BeatitStream::finalize() {
    AnalysisResult result;

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

            CoreMLResult window_result = analyze_with_coreml(window,
                                                             local_config.sample_rate,
                                                             local_config,
                                                             0.0f);
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
        }
    }

    if (coreml_beat_activation_.empty()) {
        return result;
    }

    CoreMLConfig base_config = coreml_config_;
    base_config.tempo_window_percent = 0.0f;
    base_config.prefer_double_time = false;
    base_config.synthetic_fill = false;

    std::size_t last_active_frame = 0;
    if (has_active_sample_ && coreml_config_.hop_size > 0 && sample_rate_ > 0.0) {
        const double ratio = coreml_config_.sample_rate / sample_rate_;
        const double sample_pos = static_cast<double>(last_active_sample_) * ratio;
        last_active_frame =
            static_cast<std::size_t>(std::llround(sample_pos / coreml_config_.hop_size));
    }

    CoreMLResult base = postprocess_coreml_activations(coreml_beat_activation_,
                                                      coreml_downbeat_activation_,
                                                      base_config,
                                                      sample_rate_,
                                                      0.0f,
                                                      last_active_frame);
    const float reference_bpm = estimate_bpm_from_beats(base.beat_sample_frames, sample_rate_);

    CoreMLResult final_result = postprocess_coreml_activations(coreml_beat_activation_,
                                                              coreml_downbeat_activation_,
                                                              coreml_config_,
                                                              sample_rate_,
                                                              reference_bpm,
                                                              last_active_frame);

    result.coreml_beat_activation = std::move(final_result.beat_activation);
    result.coreml_downbeat_activation = std::move(final_result.downbeat_activation);
    result.coreml_beat_feature_frames = std::move(final_result.beat_feature_frames);
    result.coreml_beat_sample_frames = std::move(final_result.beat_sample_frames);
    result.coreml_beat_strengths = std::move(final_result.beat_strengths);
    result.coreml_downbeat_feature_frames = std::move(final_result.downbeat_feature_frames);
    result.coreml_phase_energy = std::move(coreml_phase_energy_);
    result.estimated_bpm = estimate_bpm_from_beats(result.coreml_beat_sample_frames, sample_rate_);

    return result;
}

} // namespace beatit
