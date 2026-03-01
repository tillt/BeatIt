//
//  torch_plugin_api.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/analysis/torch_backend.h"
#include "beatit/inference/backend.h"

namespace beatit {
namespace detail {

inline constexpr const char* kTorchPluginAnalyzeSymbol =
    "beatit_torch_plugin_analyze_activations";
inline constexpr const char* kTorchPluginCreateBackendSymbol =
    "beatit_torch_plugin_create_inference_backend";
inline constexpr const char* kTorchPluginDestroyBackendSymbol =
    "beatit_torch_plugin_destroy_inference_backend";

using TorchPluginAnalyzeActivationsFn =
    bool (*)(const std::vector<float>& samples,
             double sample_rate,
             const BeatitConfig& config,
             CoreMLResult* out_result);

using TorchPluginCreateBackendFn = InferenceBackend* (*)();
using TorchPluginDestroyBackendFn = void (*)(InferenceBackend* backend);

} // namespace detail
} // namespace beatit
