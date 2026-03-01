//
//  backend_coreml_loader.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/inference/backend.h"
#include "beatit/inference/backend_torch.h"
#include "beatit/inference/coreml_plugin.h"
#include "beatit/logging.hpp"

#include <string>

namespace beatit {
namespace detail {
namespace {

class UnsupportedCoreMLInferenceBackend final : public InferenceBackend {
public:
    UnsupportedCoreMLInferenceBackend()
        : error_message_(coreml_plugin_error_message()) {}

    std::size_t max_batch_size(const BeatitConfig&) const override {
        return 1;
    }

    std::size_t border_frames(const BeatitConfig&) const override {
        return 0;
    }

    bool infer_window(const std::vector<float>&,
                      const BeatitConfig&,
                      std::vector<float>*,
                      std::vector<float>*,
                      InferenceTiming*) override {
        if (!reported_) {
            BEATIT_LOG_ERROR(error_message_);
            reported_ = true;
        }
        return false;
    }

private:
    std::string error_message_;
    mutable bool reported_ = false;
};

} // namespace

std::unique_ptr<InferenceBackend> make_inference_backend(const BeatitConfig& config) {
    if (config.backend == BeatitConfig::Backend::Torch) {
        return make_torch_inference_backend();
    }

    std::unique_ptr<InferenceBackend> backend = make_coreml_inference_backend_plugin();
    if (backend) {
        return backend;
    }

    return std::make_unique<UnsupportedCoreMLInferenceBackend>();
}

} // namespace detail
} // namespace beatit
