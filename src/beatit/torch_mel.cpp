//
//  torch_mel.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-23.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/torch_mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <torch/torch.h>

namespace beatit {
namespace {

float hz_to_mel(float hz, CoreMLConfig::MelScale scale) {
    if (scale == CoreMLConfig::MelScale::Slaney) {
        return 1127.01048f * std::log1p(hz / 700.0f);
    }
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel, CoreMLConfig::MelScale scale) {
    if (scale == CoreMLConfig::MelScale::Slaney) {
        return 700.0f * (std::exp(mel / 1127.01048f) - 1.0f);
    }
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> build_mel_filterbank(std::size_t mel_bins,
                                        std::size_t fft_bins,
                                        std::size_t sample_rate,
                                        float f_min,
                                        float f_max,
                                        CoreMLConfig::MelScale scale) {
    std::vector<float> filters(mel_bins * fft_bins, 0.0f);
    if (mel_bins == 0 || fft_bins == 0 || sample_rate == 0) {
        return filters;
    }

    const float nyquist = static_cast<float>(sample_rate) / 2.0f;
    const float clamped_min = std::max(0.0f, f_min);
    const float clamped_max = (f_max > 0.0f) ? std::min(f_max, nyquist) : nyquist;
    if (clamped_max <= clamped_min) {
        return filters;
    }

    const float mel_min = hz_to_mel(clamped_min, scale);
    const float mel_max = hz_to_mel(clamped_max, scale);
    std::vector<float> mel_points(mel_bins + 2);
    for (std::size_t i = 0; i < mel_points.size(); ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(mel_bins + 1);
        mel_points[i] = mel_min + t * (mel_max - mel_min);
    }

    std::vector<float> hz_points(mel_points.size());
    for (std::size_t i = 0; i < mel_points.size(); ++i) {
        hz_points[i] = mel_to_hz(mel_points[i], scale);
    }

    std::vector<float> bin_points(hz_points.size());
    for (std::size_t i = 0; i < hz_points.size(); ++i) {
        bin_points[i] = hz_points[i] * static_cast<float>(fft_bins * 2) / static_cast<float>(sample_rate);
    }

    for (std::size_t m = 0; m < mel_bins; ++m) {
        const float left = bin_points[m];
        const float center = bin_points[m + 1];
        const float right = bin_points[m + 2];
        for (std::size_t k = 0; k < fft_bins; ++k) {
            const float bin = static_cast<float>(k);
            float weight = 0.0f;
            if (bin >= left && bin <= center) {
                weight = (center <= left) ? 0.0f : (bin - left) / (center - left);
            } else if (bin >= center && bin <= right) {
                weight = (right <= center) ? 0.0f : (right - bin) / (right - center);
            }
            filters[m * fft_bins + k] = weight;
        }
    }

    return filters;
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

} // namespace

std::vector<float> compute_mel_features_torch(const std::vector<float>& samples,
                                              double sample_rate,
                                              const CoreMLConfig& config,
                                              const torch::Device& device,
                                              std::size_t* out_frames,
                                              std::string* error) {
    if (out_frames) {
        *out_frames = 0;
    }
    if (samples.empty() || sample_rate <= 0.0) {
        if (error) {
            *error = "Samples empty or sample rate invalid.";
        }
        return {};
    }
    if (config.sample_rate == 0 || config.frame_size == 0 || config.hop_size == 0 || config.mel_bins == 0) {
        if (error) {
            *error = "Config missing mel parameters.";
        }
        return {};
    }

    std::vector<float> resampled = resample_linear(samples, sample_rate, config.sample_rate);
    if (resampled.size() < config.frame_size) {
        if (error) {
            *error = "Resampled audio shorter than frame size.";
        }
        return {};
    }

    const std::size_t frames =
        1 + (resampled.size() - config.frame_size) / config.hop_size;
    if (out_frames) {
        *out_frames = frames;
    }

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor audio =
        torch::from_blob(resampled.data(),
                         {static_cast<long long>(resampled.size())},
                         torch::kFloat32)
            .to(options)
            .clone();

    const int64_t frame_size = static_cast<int64_t>(config.frame_size);
    const int64_t hop_size = static_cast<int64_t>(config.hop_size);
    torch::Tensor window = torch::hann_window(frame_size, options);

    torch::Tensor framed = audio.unfold(0, frame_size, hop_size);
    framed = framed * window;

    torch::Tensor spectrum = torch::fft::rfft(framed, frame_size);
    torch::Tensor magnitude = torch::abs(spectrum);
    if (std::abs(config.power - 1.0f) > 1e-6f) {
        magnitude = magnitude.pow(2.0f);
    }
    if (config.spectrogram_norm == CoreMLConfig::SpectrogramNorm::FrameLength) {
        magnitude = magnitude * (1.0f / static_cast<float>(config.frame_size));
    }

    const std::size_t fft_bins = config.frame_size / 2;
    if (magnitude.size(1) > static_cast<long long>(fft_bins)) {
        magnitude = magnitude.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(0, static_cast<long long>(fft_bins))});
    }

    std::vector<float> mel_filters =
        build_mel_filterbank(config.mel_bins,
                             fft_bins,
                             config.sample_rate,
                             config.f_min,
                             config.f_max,
                             config.mel_scale);
    torch::Tensor mel_filter =
        torch::from_blob(mel_filters.data(),
                         {static_cast<long long>(config.mel_bins),
                          static_cast<long long>(fft_bins)},
                         torch::kFloat32)
            .to(options)
            .clone();

    torch::Tensor mel = torch::matmul(magnitude, mel_filter.transpose(0, 1));
    if (config.use_log_mel) {
        mel = torch::log1p(mel * config.log_multiplier);
    }

    torch::Tensor mel_cpu = mel.to(torch::kCPU).contiguous();
    std::vector<float> output(mel_cpu.numel());
    std::memcpy(output.data(), mel_cpu.data_ptr<float>(),
                output.size() * sizeof(float));

    return output;
}

} // namespace beatit
