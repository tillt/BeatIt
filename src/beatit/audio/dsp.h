//
//  dsp.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/coreml.h"

#include <cstddef>
#include <vector>

namespace beatit::detail {

std::vector<float> resample_linear_mono(const std::vector<float> &input,
                                        double input_rate,
                                        std::size_t target_rate);

float hz_to_mel(float hz, CoreMLConfig::MelScale scale);

float mel_to_hz(float mel, CoreMLConfig::MelScale scale);

std::vector<float> build_mel_filterbank(std::size_t mel_bins,
                                        std::size_t fft_bins,
                                        double sample_rate, float f_min,
                                        float f_max,
                                        CoreMLConfig::MelScale scale);

} // namespace beatit::detail
