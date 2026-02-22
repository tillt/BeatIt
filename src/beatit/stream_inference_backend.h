//
//  stream_inference_backend.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#pragma once

#include "beatit/coreml.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace beatit {
namespace detail {

struct StreamInferenceTiming {
    double mel_ms = 0.0;
    double torch_forward_ms = 0.0;
};

class StreamInferenceBackend {
public:
    virtual ~StreamInferenceBackend() = default;

    virtual std::size_t max_batch_size(const CoreMLConfig& config) const = 0;
    virtual std::size_t border_frames(const CoreMLConfig& config) const = 0;

    virtual bool infer_window(const std::vector<float>& window,
                              const CoreMLConfig& config,
                              std::vector<float>* beat,
                              std::vector<float>* downbeat,
                              StreamInferenceTiming* timing) = 0;

    virtual bool infer_windows(const std::vector<std::vector<float>>& windows,
                               const CoreMLConfig& config,
                               std::vector<std::vector<float>>* beats,
                               std::vector<std::vector<float>>* downbeats,
                               StreamInferenceTiming* timing);
};

std::unique_ptr<StreamInferenceBackend> make_stream_inference_backend(
    const CoreMLConfig& config);

} // namespace detail
} // namespace beatit
