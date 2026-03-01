//
//  coreml_compile_cache_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "beatit/coreml_preset.h"
#include "coreml_test_config.h"
#include "inference/coreml/model_utils.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

bool test_compile_cache_reuses_compiled_model() {
    const std::string model_path = beatit::tests::resolve_beatthis_coreml_model_path();
    if (model_path.empty()) {
        std::cerr << "CoreML compile cache test failed: could not locate BeatThis model package.\n";
        return false;
    }

    const std::filesystem::path path(model_path);
    if (path.extension() != ".mlpackage") {
        std::cerr << "CoreML compile cache test failed: expected mlpackage, got "
                  << path.extension().string() << ".\n";
        return false;
    }

    @autoreleasepool {
        NSURL* source_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:model_path.c_str()]];
        NSError* error = nil;
        NSURL* first_compiled = beatit::detail::compile_model_if_needed(source_url, &error);
        if (!first_compiled || error) {
            std::cerr << "CoreML compile cache test failed: first compile failed";
            if (error) {
                std::cerr << " (" << error.localizedDescription.UTF8String << ")";
            }
            std::cerr << ".\n";
            return false;
        }

        const std::string first_path = first_compiled.path.UTF8String ? first_compiled.path.UTF8String : "";
        if (first_path.empty() || std::filesystem::path(first_path).extension() != ".mlmodelc") {
            std::cerr << "CoreML compile cache test failed: first compile did not produce mlmodelc.\n";
            return false;
        }
        if (!std::filesystem::exists(first_path)) {
            std::cerr << "CoreML compile cache test failed: compiled cache path does not exist: "
                      << first_path << ".\n";
            return false;
        }

        error = nil;
        NSURL* second_compiled = beatit::detail::compile_model_if_needed(source_url, &error);
        if (!second_compiled || error) {
            std::cerr << "CoreML compile cache test failed: second compile failed";
            if (error) {
                std::cerr << " (" << error.localizedDescription.UTF8String << ")";
            }
            std::cerr << ".\n";
            return false;
        }

        const std::string second_path =
            second_compiled.path.UTF8String ? second_compiled.path.UTF8String : "";
        if (first_path != second_path) {
            std::cerr << "CoreML compile cache test failed: system did not reuse compiled path.\n";
            return false;
        }
    }

    return true;
}

} // namespace

int main() {
    if (!test_compile_cache_reuses_compiled_model()) {
        return 1;
    }
    return 0;
}
