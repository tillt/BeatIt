//
//  inference_backend_torch_stub.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/inference_backend_torch.h"
#include "beatit/logging.hpp"

#include <algorithm>

namespace beatit {
namespace detail {
namespace {

class UnsupportedTorchInferenceBackend final : public InferenceBackend {
public:
    std::size_t max_batch_size(const CoreMLConfig& config) const override {
        return std::max<std::size_t>(1, config.torch_batch_size);
    }

    std::size_t border_frames(const CoreMLConfig& config) const override {
        return config.window_border_frames;
    }

    bool infer_window(const std::vector<float>&,
                      const CoreMLConfig&,
                      std::vector<float>*,
                      std::vector<float>*,
                      InferenceTiming*) override {
        BEATIT_LOG_ERROR("Torch backend not enabled in this build.");
        return false;
    }
};

} // namespace

std::unique_ptr<InferenceBackend> make_torch_inference_backend() {
    return std::make_unique<UnsupportedTorchInferenceBackend>();
}

} // namespace detail
} // namespace beatit
