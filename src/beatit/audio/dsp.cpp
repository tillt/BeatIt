//
//  dsp.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "dsp.h"

#include <algorithm>
#include <cmath>

namespace beatit::detail {

std::vector<float> resample_linear_mono(const std::vector<float> &input,
                                        double input_rate,
                                        std::size_t target_rate) {
  if (input_rate <= 0.0 || target_rate == 0 || input.empty()) {
    return {};
  }
  if (static_cast<std::size_t>(std::lround(input_rate)) == target_rate) {
    return input;
  }

  const double ratio = static_cast<double>(target_rate) / input_rate;
  const std::size_t output_size =
      static_cast<std::size_t>(std::lround(input.size() * ratio));
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
                                        double sample_rate, float f_min,
                                        float f_max,
                                        CoreMLConfig::MelScale scale) {
  std::vector<float> filters(mel_bins * fft_bins, 0.0f);
  if (mel_bins == 0 || fft_bins == 0 || sample_rate <= 0.0) {
    return filters;
  }

  const float nyquist = static_cast<float>(sample_rate / 2.0);
  const float clamped_min = std::max(0.0f, f_min);
  const float clamped_max =
      (f_max <= 0.0f || f_max > nyquist) ? nyquist : f_max;

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
    bin_points[i] = static_cast<std::size_t>(
        std::floor((fft_bins * 2) * hz_points[i] / sample_rate));
    if (bin_points[i] >= fft_bins) {
      bin_points[i] = fft_bins - 1;
    }
  }

  for (std::size_t m = 0; m < mel_bins; ++m) {
    const std::size_t left = bin_points[m];
    const std::size_t center = bin_points[m + 1];
    const std::size_t right = bin_points[m + 2];

    for (std::size_t k = left; k < center && k < fft_bins; ++k) {
      const float weight = static_cast<float>(k - left) /
                           static_cast<float>(center - left + 1e-6f);
      filters[m * fft_bins + k] = weight;
    }
    for (std::size_t k = center; k < right && k < fft_bins; ++k) {
      const float weight = static_cast<float>(right - k) /
                           static_cast<float>(right - center + 1e-6f);
      filters[m * fft_bins + k] = weight;
    }
  }

  return filters;
}

} // namespace beatit::detail
