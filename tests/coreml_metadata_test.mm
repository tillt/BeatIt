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
#include "inference/coreml/model_utils.h"

#include <iostream>
#include <string>

namespace {

enum class CoreMLMetadataTestResult {
    Passed,
    Failed,
    Skipped,
};

bool is_unsupported_model_error(NSError* error) {
    if (!error || !error.localizedDescription) {
        return false;
    }

    const std::string message = error.localizedDescription.UTF8String;
    return message.find("Unable to parse ML Program") != std::string::npos ||
           message.find("Unknown opset") != std::string::npos;
}

CoreMLMetadataTestResult test_loads_rich_package_metadata() {
    beatit::BeatitConfig config;
    beatit::tests::apply_beatthis_coreml_test_config(config);
    config.model_path = beatit::tests::resolve_beatthis_coreml_model_path();

    if (config.model_path.empty()) {
        std::cerr << "CoreML metadata test failed: could not locate BeatThis model package.\n";
        return CoreMLMetadataTestResult::Failed;
    }

    @autoreleasepool {
        NSError* error = nil;
        NSURL* source_url =
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:config.model_path.c_str()]];
        (void)beatit::detail::compile_model_if_needed(source_url, &error);
        if (is_unsupported_model_error(error)) {
            std::cerr << "CoreML metadata test skipped: model format unsupported on this runner";
            if (error) {
                std::cerr << " (" << error.localizedDescription.UTF8String << ")";
            }
            std::cerr << ".\n";
            return CoreMLMetadataTestResult::Skipped;
        }
    }

    const beatit::CoreMLMetadata metadata = beatit::load_coreml_metadata(config);
    if (metadata.author != "CPJKU (Converted for iOS)") {
        std::cerr << "CoreML metadata test failed: unexpected author '" << metadata.author
                  << "'.\n";
        return CoreMLMetadataTestResult::Failed;
    }

    if (metadata.short_description != "Beat This! (small0) - Beat Tracking") {
        std::cerr << "CoreML metadata test failed: unexpected description '"
                  << metadata.short_description << "'.\n";
        return CoreMLMetadataTestResult::Failed;
    }

    if (metadata.license != "MIT") {
        std::cerr << "CoreML metadata test failed: unexpected license '" << metadata.license
                  << "'.\n";
        return CoreMLMetadataTestResult::Failed;
    }

    if (metadata.version != "small0") {
        std::cerr << "CoreML metadata test failed: unexpected version '" << metadata.version
                  << "'.\n";
        return CoreMLMetadataTestResult::Failed;
    }

    return CoreMLMetadataTestResult::Passed;
}

} // namespace

int main() {
    @autoreleasepool {
        switch (test_loads_rich_package_metadata()) {
        case CoreMLMetadataTestResult::Passed:
            return 0;
        case CoreMLMetadataTestResult::Skipped:
            return 77;
        case CoreMLMetadataTestResult::Failed:
            return 1;
        }
    }
}
