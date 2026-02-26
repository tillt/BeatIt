//
//  mel_torch.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-23.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "beatit/coreml.h"

namespace beatit {

std::vector<float> compute_mel_features_torch(const std::vector<float>& samples,
                                              double sample_rate,
                                              const BeatitConfig& config,
                                              const torch::Device& device,
                                              std::size_t* out_frames,
                                              std::string* error);

} // namespace beatit
