//
//  coreml_loader.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/config.h"
#include "beatit/inference/coreml_plugin.h"

namespace beatit {

CoreMLResult analyze_with_coreml(const std::vector<float>& samples,
                                 double sample_rate,
                                 const BeatitConfig& config,
                                 float reference_bpm) {
    CoreMLResult result;
    detail::coreml_plugin_analyze_activations(samples,
                                              sample_rate,
                                              config,
                                              reference_bpm,
                                              &result);
    return result;
}

} // namespace beatit
