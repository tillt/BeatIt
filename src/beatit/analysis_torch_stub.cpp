//
//  analysis_torch_stub.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#include "beatit/analysis_torch_backend.h"

#include <iostream>

namespace beatit {

CoreMLResult analyze_with_torch_activations(const std::vector<float>&,
                                            double,
                                            const CoreMLConfig& config) {
    if (config.verbose) {
        std::cerr << "Torch backend not enabled in this build.\n";
    }
    return {};
}

} // namespace beatit
