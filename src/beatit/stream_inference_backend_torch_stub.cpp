//
//  stream_inference_backend_torch_stub.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#include "beatit/stream_inference_backend_torch.h"

#include <algorithm>
#include <iostream>

namespace beatit {
namespace detail {
namespace {

class UnsupportedTorchStreamInferenceBackend final : public StreamInferenceBackend {
public:
    std::size_t max_batch_size(const CoreMLConfig& config) const override {
        return std::max<std::size_t>(1, config.torch_batch_size);
    }

    std::size_t border_frames(const CoreMLConfig& config) const override {
        return config.window_border_frames;
    }

    bool infer_window(const std::vector<float>&,
                      const CoreMLConfig& config,
                      std::vector<float>*,
                      std::vector<float>*,
                      StreamInferenceTiming*) override {
        if (config.verbose) {
            std::cerr << "Torch backend not enabled in this build.\n";
        }
        return false;
    }
};

} // namespace

std::unique_ptr<StreamInferenceBackend> make_torch_stream_inference_backend() {
    return std::make_unique<UnsupportedTorchStreamInferenceBackend>();
}

} // namespace detail
} // namespace beatit
