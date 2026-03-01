//
//  backend_coreml.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/inference/backend.h"
#include "beatit/inference/backend_torch.h"
#include "beatit/logging.hpp"

#include <algorithm>

namespace beatit {
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
        CoreMLResult result = analyze_with_coreml(window,
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

std::unique_ptr<InferenceBackend> make_inference_backend(
    const BeatitConfig& config) {
    if (config.backend == BeatitConfig::Backend::Torch) {
        return make_torch_inference_backend();
    }
    return std::make_unique<CoreMLInferenceBackend>();
}

} // namespace detail
} // namespace beatit
