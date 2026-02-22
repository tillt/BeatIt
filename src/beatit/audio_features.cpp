//
//  audio_features.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/coreml.h"

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace beatit {
namespace {

constexpr float kPi = 3.14159265358979323846f;

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

std::vector<float> make_hann_window(std::size_t size) {
    std::vector<float> window(size);
    if (size == 0) {
        return window;
    }

    const float denom = static_cast<float>(size - 1);
    for (std::size_t i = 0; i < size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * (static_cast<float>(i) / denom)));
    }
    return window;
}

float hz_to_mel(float hz, CoreMLConfig::MelScale scale) {
    if (scale == CoreMLConfig::MelScale::Slaney) {
        return 1127.01048f * std::log(1.0f + hz / 700.0f);
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
                                        double sample_rate,
                                        float f_min,
                                        float f_max,
                                        CoreMLConfig::MelScale scale) {
    std::vector<float> filters(mel_bins * fft_bins, 0.0f);
    if (mel_bins == 0 || fft_bins == 0 || sample_rate <= 0.0) {
        return filters;
    }

    const float nyquist = static_cast<float>(sample_rate / 2.0);
    const float clamped_min = std::max(0.0f, f_min);
    const float clamped_max = (f_max <= 0.0f || f_max > nyquist) ? nyquist : f_max;
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

    std::vector<std::size_t> bin_points(hz_points.size());
    for (std::size_t i = 0; i < hz_points.size(); ++i) {
        bin_points[i] = static_cast<std::size_t>(std::floor((fft_bins * 2) * hz_points[i] / sample_rate));
        if (bin_points[i] >= fft_bins) {
            bin_points[i] = fft_bins - 1;
        }
    }

    for (std::size_t m = 0; m < mel_bins; ++m) {
        const std::size_t left = bin_points[m];
        const std::size_t center = bin_points[m + 1];
        const std::size_t right = bin_points[m + 2];

        for (std::size_t k = left; k < center && k < fft_bins; ++k) {
            const float weight = static_cast<float>(k - left) / static_cast<float>(center - left + 1e-6f);
            filters[m * fft_bins + k] = weight;
        }
        for (std::size_t k = center; k < right && k < fft_bins; ++k) {
            const float weight = static_cast<float>(right - k) / static_cast<float>(right - center + 1e-6f);
            filters[m * fft_bins + k] = weight;
        }
    }

    return filters;
}

std::vector<float> compute_mel(const std::vector<float>& samples,
                               std::size_t sample_rate,
                               std::size_t frame_size,
                               std::size_t hop_size,
                               std::size_t mel_bins,
                               bool use_log,
                               float log_multiplier,
                               float f_min,
                               float f_max,
                               CoreMLConfig::MelScale mel_scale,
                               CoreMLConfig::SpectrogramNorm spectrogram_norm,
                               float power) {
    if (samples.empty() || frame_size == 0 || hop_size == 0 || mel_bins == 0) {
        return {};
    }
    if (samples.size() < frame_size) {
        return {};
    }

    const std::size_t fft_bins = frame_size / 2;
    const std::size_t frame_count = 1 + (samples.size() - frame_size) / hop_size;

    std::vector<float> window = make_hann_window(frame_size);
    std::vector<float> buffer(frame_size, 0.0f);
    std::vector<float> windowed(frame_size, 0.0f);
    std::vector<float> spectrum(fft_bins, 0.0f);

    std::vector<float> split_real(fft_bins, 0.0f);
    std::vector<float> split_imag(fft_bins, 0.0f);
    DSPSplitComplex split{};
    split.realp = split_real.data();
    split.imagp = split_imag.data();

    const int fft_log2 = static_cast<int>(std::log2(frame_size));
    FFTSetup setup = vDSP_create_fftsetup(fft_log2, kFFTRadix2);

    std::vector<float> mel_filters =
        build_mel_filterbank(mel_bins, fft_bins, sample_rate, f_min, f_max, mel_scale);
    std::vector<float> features(frame_count * mel_bins, 0.0f);

    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const std::size_t offset = frame * hop_size;
        std::copy(samples.begin() + offset, samples.begin() + offset + frame_size, buffer.begin());
        vDSP_vmul(buffer.data(), 1, window.data(), 1, windowed.data(), 1, frame_size);
        vDSP_ctoz(reinterpret_cast<DSPComplex*>(windowed.data()), 2, &split, 1, fft_bins);
        vDSP_fft_zrip(setup, &split, 1, fft_log2, FFT_FORWARD);
        if (std::abs(power - 1.0f) < 1e-6f) {
            vDSP_zvabs(&split, 1, spectrum.data(), 1, fft_bins);
        } else {
            vDSP_zvmags(&split, 1, spectrum.data(), 1, fft_bins);
        }
        if (spectrogram_norm == CoreMLConfig::SpectrogramNorm::FrameLength) {
            const float scale = 1.0f / static_cast<float>(frame_size);
            vDSP_vsmul(spectrum.data(), 1, &scale, spectrum.data(), 1, fft_bins);
        }

        for (std::size_t m = 0; m < mel_bins; ++m) {
            float sum = 0.0f;
            const float* filter = mel_filters.data() + m * fft_bins;
            for (std::size_t k = 0; k < fft_bins; ++k) {
                sum += spectrum[k] * filter[k];
            }
            const float value =
                use_log ? std::log1p(log_multiplier * sum) : sum;
            features[frame * mel_bins + m] = value;
        }
    }

    vDSP_destroy_fftsetup(setup);
    return features;
}

} // namespace

std::vector<float> compute_mel_features(const std::vector<float>& samples,
                                        double sample_rate,
                                        const CoreMLConfig& config,
                                        std::size_t* out_frames) {
    if (out_frames) {
        *out_frames = 0;
    }
    if (samples.empty() || sample_rate <= 0.0) {
        return {};
    }
    if (config.sample_rate == 0 || config.frame_size == 0 || config.hop_size == 0 || config.mel_bins == 0) {
        return {};
    }

    std::vector<float> resampled = resample_linear(samples, sample_rate, config.sample_rate);
    if (resampled.size() < config.frame_size) {
        return {};
    }

    const std::size_t frames = 1 + (resampled.size() - config.frame_size) / config.hop_size;
    if (out_frames) {
        *out_frames = frames;
    }

    return compute_mel(resampled,
                       config.sample_rate,
                       config.frame_size,
                       config.hop_size,
                       config.mel_bins,
                       config.use_log_mel,
                       config.log_multiplier,
                       config.f_min,
                       config.f_max,
                       config.mel_scale,
                       config.spectrogram_norm,
                       config.power);
}

} // namespace beatit
