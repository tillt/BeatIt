//
//  torch_plugin.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <Foundation/Foundation.h>

#include "beatit/inference/torch_plugin.h"

#include "beatit/inference/torch_plugin_api.h"
#include "beatit/logging.hpp"

#include <dlfcn.h>
#include <mach-o/dyld.h>

#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

@interface BeatitTorchPluginAnchor : NSObject
@end

@implementation BeatitTorchPluginAnchor
@end

namespace beatit {
namespace detail {
namespace {

struct TorchPluginFunctions {
    void* handle = nullptr;
    TorchPluginAnalyzeActivationsFn analyze = nullptr;
    TorchPluginCreateBackendFn create_backend = nullptr;
    TorchPluginDestroyBackendFn destroy_backend = nullptr;
};

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

std::vector<std::string> torch_plugin_candidates() {
    constexpr const char* kPluginName = "libbeatit_backend_torch.dylib";

    std::vector<std::string> candidates;
    const std::string exe_dir = executable_dir();
    if (!exe_dir.empty()) {
        candidates.push_back((std::filesystem::path(exe_dir) / kPluginName).string());
        candidates.push_back((std::filesystem::path(exe_dir) / "plugins" / kPluginName).string());
    }

    candidates.push_back((std::filesystem::current_path() / kPluginName).string());
    candidates.push_back((std::filesystem::current_path() / "plugins" / kPluginName).string());
    candidates.push_back("/opt/homebrew/lib/beatit/libbeatit_backend_torch.dylib");
    candidates.push_back("/usr/local/lib/beatit/libbeatit_backend_torch.dylib");

    NSString* main_bundle_path = [[NSBundle mainBundle] pathForResource:@"libbeatit_backend_torch"
                                                                 ofType:@"dylib"
                                                            inDirectory:@"plugins"];
    if (main_bundle_path) {
        candidates.emplace_back(main_bundle_path.UTF8String);
    }

    NSBundle* framework_bundle = [NSBundle bundleForClass:[BeatitTorchPluginAnchor class]];
    if (framework_bundle && framework_bundle != [NSBundle mainBundle]) {
        NSString* framework_plugin =
            [framework_bundle pathForResource:@"libbeatit_backend_torch"
                                       ofType:@"dylib"
                                  inDirectory:@"plugins"];
        if (framework_plugin) {
            candidates.emplace_back(framework_plugin.UTF8String);
        }
    }

    return candidates;
}

TorchPluginFunctions* load_torch_plugin() {
    static TorchPluginFunctions functions;
    static bool initialized = false;
    static std::mutex mutex;

    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return functions.handle ? &functions : nullptr;
    }
    initialized = true;

    for (const std::string& candidate : torch_plugin_candidates()) {
        if (!std::filesystem::exists(candidate)) {
            continue;
        }

        void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            BEATIT_LOG_WARN("Torch plugin: failed to load '" << candidate << "': " << dlerror());
            continue;
        }

        auto analyze =
            reinterpret_cast<TorchPluginAnalyzeActivationsFn>(dlsym(handle, kTorchPluginAnalyzeSymbol));
        auto create_backend =
            reinterpret_cast<TorchPluginCreateBackendFn>(dlsym(handle, kTorchPluginCreateBackendSymbol));
        auto destroy_backend =
            reinterpret_cast<TorchPluginDestroyBackendFn>(dlsym(handle, kTorchPluginDestroyBackendSymbol));

        if (!analyze || !create_backend || !destroy_backend) {
            BEATIT_LOG_WARN("Torch plugin: missing required symbols in '" << candidate << "'.");
            dlclose(handle);
            continue;
        }

        functions.handle = handle;
        functions.analyze = analyze;
        functions.create_backend = create_backend;
        functions.destroy_backend = destroy_backend;
        BEATIT_LOG_INFO("Torch plugin: loaded '" << candidate << "'.");
        return &functions;
    }

    return nullptr;
}

} // namespace

bool torch_plugin_analyze_activations(const std::vector<float>& samples,
                                      double sample_rate,
                                      const BeatitConfig& config,
                                      CoreMLResult* out_result) {
    if (!out_result) {
        return false;
    }

    TorchPluginFunctions* plugin = load_torch_plugin();
    if (!plugin || !plugin->analyze) {
        BEATIT_LOG_ERROR("Torch backend plugin is not available.");
        return false;
    }

    return plugin->analyze(samples, sample_rate, config, out_result);
}

std::unique_ptr<InferenceBackend> make_torch_inference_backend_plugin() {
    TorchPluginFunctions* plugin = load_torch_plugin();
    if (!plugin || !plugin->create_backend) {
        return nullptr;
    }
    return std::unique_ptr<InferenceBackend>(plugin->create_backend());
}

} // namespace detail
} // namespace beatit
