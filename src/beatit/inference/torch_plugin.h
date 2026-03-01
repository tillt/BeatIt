//
//  torch_plugin.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/analysis/torch_backend.h"
#include "beatit/inference/backend.h"

#include <string>
#include <memory>
#include <vector>

namespace beatit {
namespace detail {

/**
 * Load the Torch plugin and run raw activation inference.
 */
bool torch_plugin_analyze_activations(const std::vector<float>& samples,
                                      double sample_rate,
                                      const BeatitConfig& config,
                                      CoreMLResult* out_result);

/**
 * Load the Torch plugin and create the inference backend instance.
 */
std::unique_ptr<InferenceBackend> make_torch_inference_backend_plugin();

/**
 * Return the last Torch plugin load failure description.
 */
std::string torch_plugin_error_message();

} // namespace detail
} // namespace beatit
