//
//  coreml_plugin_api.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/config.h"
#include "beatit/inference/backend.h"

namespace beatit {
namespace detail {

inline constexpr const char* kCoreMLPluginAnalyzeSymbol =
    "beatit_coreml_plugin_analyze_activations";
inline constexpr const char* kCoreMLPluginCreateBackendSymbol =
    "beatit_coreml_plugin_create_inference_backend";
inline constexpr const char* kCoreMLPluginDestroyBackendSymbol =
    "beatit_coreml_plugin_destroy_inference_backend";

using CoreMLPluginAnalyzeActivationsFn =
    bool (*)(const std::vector<float>& samples,
             double sample_rate,
             const BeatitConfig& config,
             float reference_bpm,
             CoreMLResult* out_result);

using CoreMLPluginCreateBackendFn = InferenceBackend* (*)();
using CoreMLPluginDestroyBackendFn = void (*)(InferenceBackend* backend);

} // namespace detail
} // namespace beatit
