//
//  features.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/config.h"
#include "dsp.h"

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace beatit {
namespace {

constexpr float kPi = 3.14159265358979323846f;

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

std::vector<float> compute_mel(const std::vector<float>& samples,
                               std::size_t sample_rate,
                               std::size_t frame_size,
                               std::size_t hop_size,
                               std::size_t mel_bins,
                               bool use_log,
                               float log_multiplier,
                               float f_min,
                               float f_max,
                               BeatitConfig::MelScale mel_scale,
                               BeatitConfig::SpectrogramNorm spectrogram_norm,
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
        detail::build_mel_filterbank(mel_bins, fft_bins, sample_rate, f_min, f_max, mel_scale);
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
        if (spectrogram_norm == BeatitConfig::SpectrogramNorm::FrameLength) {
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
                                        const BeatitConfig& config,
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

    std::vector<float> resampled =
        detail::resample_linear_mono(samples, sample_rate, config.sample_rate);
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
