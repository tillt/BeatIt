//
//  torch_stub.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/analysis/torch_backend.h"
#include "beatit/logging.hpp"

namespace beatit {

CoreMLResult analyze_with_torch_activations(const std::vector<float>&,
                                            double,
                                            const CoreMLConfig&) {
    BEATIT_LOG_ERROR("Torch backend not enabled in this build.");
    return {};
}

} // namespace beatit
