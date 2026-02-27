//
//  backend.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/config.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace beatit {
namespace detail {

struct InferenceTiming {
    double mel_ms = 0.0;
    double torch_forward_ms = 0.0;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual std::size_t max_batch_size(const BeatitConfig& config) const = 0;
    virtual std::size_t border_frames(const BeatitConfig& config) const = 0;

    virtual bool infer_window(const std::vector<float>& window,
                              const BeatitConfig& config,
                              std::vector<float>* beat,
                              std::vector<float>* downbeat,
                              InferenceTiming* timing) = 0;

    virtual bool infer_windows(const std::vector<std::vector<float>>& windows,
                               const BeatitConfig& config,
                               std::vector<std::vector<float>>* beats,
                               std::vector<std::vector<float>>* downbeats,
                               InferenceTiming* timing);
};

std::unique_ptr<InferenceBackend> make_inference_backend(
    const BeatitConfig& config);

} // namespace detail
} // namespace beatit
