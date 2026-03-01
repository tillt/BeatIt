//
//  torch_loader.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/analysis/torch_backend.h"

#include "beatit/inference/torch_plugin.h"
#include "beatit/logging.hpp"

namespace beatit {

CoreMLResult analyze_with_torch_activations(const std::vector<float>& samples,
                                            double sample_rate,
                                            const BeatitConfig& config) {
    CoreMLResult result;
    if (!detail::torch_plugin_analyze_activations(samples, sample_rate, config, &result)) {
        BEATIT_LOG_ERROR("Torch backend plugin failed to produce activations.");
        return {};
    }
    return result;
}

} // namespace beatit
