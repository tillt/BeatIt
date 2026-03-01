//
//  backend_coreml.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/inference/backend.h"
#include "beatit/inference/coreml_plugin_api.h"
#include "beatit/logging.hpp"

#include <vector>

namespace beatit {

CoreMLResult coreml_plugin_analyze_with_coreml_impl(const std::vector<float>& samples,
                                                    double sample_rate,
                                                    const BeatitConfig& config,
                                                    float reference_bpm);

namespace detail {
namespace {

class CoreMLInferenceBackend final : public InferenceBackend {
public:
    std::size_t max_batch_size(const BeatitConfig&) const override {
        return 1;
    }

    std::size_t border_frames(const BeatitConfig&) const override {
        return 0;
    }

    bool infer_window(const std::vector<float>& window,
                      const BeatitConfig& config,
                      std::vector<float>* beat,
                      std::vector<float>* downbeat,
                      InferenceTiming*) override {
        if (!beat || !downbeat) {
            return false;
        }

        BeatitConfig local_config = config;
        local_config.tempo_window_percent = 0.0f;
        local_config.prefer_double_time = false;
        CoreMLResult result = coreml_plugin_analyze_with_coreml_impl(window,
                                                                     local_config.sample_rate,
                                                                     local_config,
                                                                     0.0f);
        if (result.beat_activation.empty()) {
            BEATIT_LOG_ERROR("CoreML backend: inference returned empty beat activation.");
            return false;
        }

        *beat = std::move(result.beat_activation);
        *downbeat = std::move(result.downbeat_activation);
        return true;
    }
};

} // namespace
} // namespace detail
} // namespace beatit

extern "C" bool beatit_coreml_plugin_analyze_activations(const std::vector<float>& samples,
                                                         double sample_rate,
                                                         const beatit::BeatitConfig& config,
                                                         float reference_bpm,
                                                         beatit::CoreMLResult* out_result) {
    if (!out_result) {
        return false;
    }

    *out_result = beatit::coreml_plugin_analyze_with_coreml_impl(samples,
                                                                 sample_rate,
                                                                 config,
                                                                 reference_bpm);
    return !out_result->beat_activation.empty();
}

extern "C" beatit::detail::InferenceBackend* beatit_coreml_plugin_create_inference_backend() {
    return new beatit::detail::CoreMLInferenceBackend();
}

extern "C" void beatit_coreml_plugin_destroy_inference_backend(
    beatit::detail::InferenceBackend* backend) {
    delete backend;
}
