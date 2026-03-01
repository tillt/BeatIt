//
//  coreml_plugin.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/config.h"
#include "beatit/inference/backend.h"

#include <memory>
#include <string>
#include <vector>

namespace beatit {
namespace detail {

/**
 * @brief Load the CoreML plugin and run raw activation inference.
 */
bool coreml_plugin_analyze_activations(const std::vector<float>& samples,
                                       double sample_rate,
                                       const BeatitConfig& config,
                                       float reference_bpm,
                                       CoreMLResult* out_result);

/**
 * @brief Load the CoreML plugin and create the inference backend instance.
 */
std::unique_ptr<InferenceBackend> make_coreml_inference_backend_plugin();

/**
 * @brief Return the last CoreML plugin load failure description.
 */
std::string coreml_plugin_error_message();

/**
 * @brief Probe whether the CoreML plugin can be loaded.
 */
bool coreml_plugin_available();

} // namespace detail
} // namespace beatit
