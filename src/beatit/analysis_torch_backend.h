//
//  analysis_torch_backend.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/coreml.h"

#include <vector>

namespace beatit {

CoreMLResult analyze_with_torch_activations(const std::vector<float>& samples,
                                            double sample_rate,
                                            const CoreMLConfig& config);

} // namespace beatit
