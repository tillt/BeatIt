//
//  coreml_metadata_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <Foundation/Foundation.h>

#include "beatit/config.h"
#include "coreml_test_config.h"

#include <iostream>
#include <string>

namespace {

bool test_loads_rich_package_metadata() {
    beatit::BeatitConfig config;
    beatit::tests::apply_beatthis_coreml_test_config(config);
    config.model_path = beatit::tests::resolve_beatthis_coreml_model_path();

    if (config.model_path.empty()) {
        std::cerr << "CoreML metadata test failed: could not locate BeatThis model package.\n";
        return false;
    }

    const beatit::CoreMLMetadata metadata = beatit::load_coreml_metadata(config);
    if (metadata.author != "CPJKU (Converted for iOS)") {
        std::cerr << "CoreML metadata test failed: unexpected author '" << metadata.author
                  << "'.\n";
        return false;
    }

    if (metadata.short_description != "Beat This! (small0) - Beat Tracking") {
        std::cerr << "CoreML metadata test failed: unexpected description '"
                  << metadata.short_description << "'.\n";
        return false;
    }

    if (metadata.license != "MIT") {
        std::cerr << "CoreML metadata test failed: unexpected license '" << metadata.license
                  << "'.\n";
        return false;
    }

    if (metadata.version != "small0") {
        std::cerr << "CoreML metadata test failed: unexpected version '" << metadata.version
                  << "'.\n";
        return false;
    }

    return true;
}

} // namespace

int main() {
    @autoreleasepool {
        if (!test_loads_rich_package_metadata()) {
            return 1;
        }
    }
    return 0;
}
