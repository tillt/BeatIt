//
//  inference_backend_coreml.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#include "beatit/inference_backend.h"
#include "beatit/inference_backend_torch.h"

#include <algorithm>

namespace beatit {
namespace detail {

bool InferenceBackend::infer_windows(const std::vector<std::vector<float>>& windows,
                                     const CoreMLConfig& config,
                                     std::vector<std::vector<float>>* beats,
                                     std::vector<std::vector<float>>* downbeats,
                                     InferenceTiming* timing) {
    if (!beats || !downbeats) {
        return false;
    }
    beats->clear();
    downbeats->clear();
    beats->reserve(windows.size());
    downbeats->reserve(windows.size());

    for (const auto& window : windows) {
        std::vector<float> beat;
        std::vector<float> downbeat;
        if (!infer_window(window, config, &beat, &downbeat, timing)) {
            return false;
        }
        beats->push_back(std::move(beat));
        downbeats->push_back(std::move(downbeat));
    }
    return true;
}

namespace {

class CoreMLInferenceBackend final : public InferenceBackend {
public:
    std::size_t max_batch_size(const CoreMLConfig&) const override {
        return 1;
    }

    std::size_t border_frames(const CoreMLConfig&) const override {
        return 0;
    }

    bool infer_window(const std::vector<float>& window,
                      const CoreMLConfig& config,
                      std::vector<float>* beat,
                      std::vector<float>* downbeat,
                      InferenceTiming*) override {
        if (!beat || !downbeat) {
            return false;
        }

        CoreMLConfig local_config = config;
        local_config.tempo_window_percent = 0.0f;
        local_config.prefer_double_time = false;
        CoreMLResult result = analyze_with_coreml(window,
                                                  local_config.sample_rate,
                                                  local_config,
                                                  0.0f);
        *beat = std::move(result.beat_activation);
        *downbeat = std::move(result.downbeat_activation);
        return true;
    }
};

} // namespace

std::unique_ptr<InferenceBackend> make_inference_backend(
    const CoreMLConfig& config) {
    if (config.backend == CoreMLConfig::Backend::Torch) {
        return make_torch_inference_backend();
    }
    return std::make_unique<CoreMLInferenceBackend>();
}

} // namespace detail
} // namespace beatit
