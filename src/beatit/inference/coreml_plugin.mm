//
//  coreml_plugin.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <Foundation/Foundation.h>

#include "beatit/inference/coreml_plugin.h"

#include "beatit/inference/coreml_plugin_api.h"
#include "beatit/logging.hpp"

#include <dlfcn.h>
#include <mach-o/dyld.h>

#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

@interface BeatitCoreMLPluginAnchor : NSObject
@end

@implementation BeatitCoreMLPluginAnchor
@end

namespace beatit {
namespace detail {
namespace {

struct CoreMLPluginFunctions {
    void* handle = nullptr;
    CoreMLPluginAnalyzeActivationsFn analyze = nullptr;
    CoreMLPluginCreateBackendFn create_backend = nullptr;
    CoreMLPluginDestroyBackendFn destroy_backend = nullptr;
};

std::string& coreml_plugin_last_error() {
    static std::string error;
    return error;
}

std::string executable_dir() {
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    if (size == 0) {
        return {};
    }

    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return {};
    }
    buffer.resize(std::strlen(buffer.c_str()));
    return std::filesystem::path(buffer).parent_path().string();
}

std::vector<std::string> coreml_plugin_candidates() {
    constexpr const char* kPluginName = "libbeatit_backend_coreml.dylib";

    std::vector<std::string> candidates;
    const std::string exe_dir = executable_dir();
    if (!exe_dir.empty()) {
        candidates.push_back((std::filesystem::path(exe_dir) / kPluginName).string());
        candidates.push_back(
            (std::filesystem::path(exe_dir) / "plugins" / kPluginName).string());
    }

    candidates.push_back((std::filesystem::current_path() / kPluginName).string());
    candidates.push_back((std::filesystem::current_path() / "plugins" / kPluginName).string());
    candidates.push_back("/opt/homebrew/lib/beatit/libbeatit_backend_coreml.dylib");
    candidates.push_back("/usr/local/lib/beatit/libbeatit_backend_coreml.dylib");

    NSString* main_bundle_path = [[NSBundle mainBundle] pathForResource:@"libbeatit_backend_coreml"
                                                                 ofType:@"dylib"
                                                            inDirectory:@"plugins"];
    if (main_bundle_path) {
        candidates.emplace_back(main_bundle_path.UTF8String);
    }

    NSBundle* framework_bundle = [NSBundle bundleForClass:[BeatitCoreMLPluginAnchor class]];
    if (framework_bundle && framework_bundle != [NSBundle mainBundle]) {
        NSString* framework_plugin =
            [framework_bundle pathForResource:@"libbeatit_backend_coreml"
                                       ofType:@"dylib"
                                  inDirectory:@"plugins"];
        if (framework_plugin) {
            candidates.emplace_back(framework_plugin.UTF8String);
        }
    }

    return candidates;
}

CoreMLPluginFunctions* load_coreml_plugin() {
    static CoreMLPluginFunctions functions;
    static bool initialized = false;
    static std::mutex mutex;

    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return functions.handle ? &functions : nullptr;
    }
    initialized = true;

    std::vector<std::string> searched_paths;
    for (const std::string& candidate : coreml_plugin_candidates()) {
        searched_paths.push_back(candidate);
        if (!std::filesystem::exists(candidate)) {
            continue;
        }

        void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            BEATIT_LOG_WARN("CoreML plugin: failed to load '" << candidate << "': " << dlerror());
            continue;
        }

        auto analyze = reinterpret_cast<CoreMLPluginAnalyzeActivationsFn>(
            dlsym(handle, kCoreMLPluginAnalyzeSymbol));
        auto create_backend = reinterpret_cast<CoreMLPluginCreateBackendFn>(
            dlsym(handle, kCoreMLPluginCreateBackendSymbol));
        auto destroy_backend = reinterpret_cast<CoreMLPluginDestroyBackendFn>(
            dlsym(handle, kCoreMLPluginDestroyBackendSymbol));

        if (!analyze || !create_backend || !destroy_backend) {
            BEATIT_LOG_WARN("CoreML plugin: missing required symbols in '" << candidate << "'.");
            dlclose(handle);
            continue;
        }

        functions.handle = handle;
        functions.analyze = analyze;
        functions.create_backend = create_backend;
        functions.destroy_backend = destroy_backend;
        BEATIT_LOG_INFO("CoreML plugin: loaded '" << candidate << "'.");
        return &functions;
    }

    std::ostringstream message;
    message << "CoreML backend plugin is not available. Searched:";
    for (const std::string& path : searched_paths) {
        message << "\n  - " << path;
    }
    coreml_plugin_last_error() = message.str();
    return nullptr;
}

} // namespace

bool coreml_plugin_analyze_activations(const std::vector<float>& samples,
                                       double sample_rate,
                                       const BeatitConfig& config,
                                       float reference_bpm,
                                       CoreMLResult* out_result) {
    if (!out_result) {
        return false;
    }

    CoreMLPluginFunctions* plugin = load_coreml_plugin();
    if (!plugin || !plugin->analyze) {
        BEATIT_LOG_ERROR(coreml_plugin_last_error());
        return false;
    }

    return plugin->analyze(samples, sample_rate, config, reference_bpm, out_result);
}

std::unique_ptr<InferenceBackend> make_coreml_inference_backend_plugin() {
    CoreMLPluginFunctions* plugin = load_coreml_plugin();
    if (!plugin || !plugin->create_backend) {
        BEATIT_LOG_ERROR(coreml_plugin_last_error());
        return nullptr;
    }

    return std::unique_ptr<InferenceBackend>(plugin->create_backend());
}

std::string coreml_plugin_error_message() {
    load_coreml_plugin();
    return coreml_plugin_last_error();
}

bool coreml_plugin_available() {
    return load_coreml_plugin() != nullptr;
}

} // namespace detail
} // namespace beatit
