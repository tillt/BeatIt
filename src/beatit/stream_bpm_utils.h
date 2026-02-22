//
//  stream_bpm_utils.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#pragma once

#include "beatit/coreml.h"

#include <vector>

namespace beatit {
namespace detail {

float estimate_bpm_from_beats_local(const std::vector<unsigned long long>& beat_samples,
                                    double sample_rate);

float estimate_bpm_from_activation_peaks_local(const std::vector<float>& activation,
                                               const CoreMLConfig& config,
                                               double sample_rate);

float estimate_bpm_from_activation_autocorr_local(const std::vector<float>& activation,
                                                  const CoreMLConfig& config,
                                                  double sample_rate);

float normalize_bpm_to_range_local(float bpm, float min_bpm, float max_bpm);

} // namespace detail
} // namespace beatit
