//
//  stream_inference_backend_torch.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#pragma once

#include "beatit/stream_inference_backend.h"

#include <memory>

namespace beatit {
namespace detail {

std::unique_ptr<StreamInferenceBackend> make_torch_stream_inference_backend();

} // namespace detail
} // namespace beatit
