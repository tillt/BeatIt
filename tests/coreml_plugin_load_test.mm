//
//  coreml_plugin_load_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/config.h"
#include "beatit/inference/backend.h"
#include "beatit/inference/coreml_plugin.h"

#include <iostream>
#include <memory>

int main() {
    if (!beatit::detail::coreml_plugin_available()) {
        std::cerr << "CoreML plugin load test failed: plugin is not available.\n";
        return 1;
    }

    beatit::BeatitConfig config;
    config.backend = beatit::BeatitConfig::Backend::CoreML;
    std::unique_ptr<beatit::detail::InferenceBackend> backend =
        beatit::detail::make_inference_backend(config);
    if (!backend) {
        std::cerr << "CoreML plugin load test failed: backend factory returned null.\n";
        return 1;
    }

    if (backend->max_batch_size(config) == 0) {
        std::cerr << "CoreML plugin load test failed: invalid batch size.\n";
        return 1;
    }

    return 0;
}
