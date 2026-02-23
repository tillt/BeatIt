//
//  coreml_test_config.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#ifndef BEATIT_TESTS_COREML_TEST_CONFIG_H
#define BEATIT_TESTS_COREML_TEST_CONFIG_H

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "beatit/coreml.h"
#include "beatit/coreml_preset.h"

namespace beatit::tests {

inline void apply_beatthis_coreml_test_config(CoreMLConfig& config) {
    if (auto preset = make_coreml_preset("beatthis")) {
        preset->apply(config);
    }

    // Keep explicit values to make tests robust if preset defaults evolve.
    config.backend = CoreMLConfig::Backend::CoreML;
    config.sample_rate = 22050;
    config.frame_size = 1024;
    config.hop_size = 441;
    config.mel_bins = 128;
    config.use_log_mel = true;
    config.log_multiplier = 1000.0f;
    config.f_min = 30.0f;
    config.f_max = 11000.0f;
    config.power = 1.0f;
    config.input_layout = CoreMLConfig::InputLayout::FramesByMels;
    config.fixed_frames = 1500;
    config.window_hop_frames = 1488;
    config.window_border_frames = 6;
    config.input_name = "mel_spectrogram";
    config.beat_output_name = "beat_logits";
    config.downbeat_output_name = "downbeat_logits";
    config.use_dbn = false;
}

inline std::string resolve_beatthis_coreml_model_path() {
    std::filesystem::path test_root = std::filesystem::current_path();
#ifdef BEATIT_TEST_DATA_DIR
    test_root = BEATIT_TEST_DATA_DIR;
#endif

    if (const char* env = std::getenv("BEATIT_COREML_MODEL_PATH")) {
        if (env[0] != '\0' && std::filesystem::exists(env)) {
            return env;
        }
    }

    const std::vector<std::string> candidates = {
        "coreml_out_latest/BeatThis_small0.mlpackage",
        "coreml_out_latest/BeatThis_small0.mlmodelc",
        "coreml_out/BeatThis_small0.mlpackage",
        "coreml_out/BeatThis_small0.mlmodelc",
        "coreml_out_latest/BeatThis_final0.mlpackage",
        "coreml_out_latest/BeatThis_final0.mlmodelc",
        "coreml_out/BeatThis_final0.mlpackage",
        "coreml_out/BeatThis_final0.mlmodelc"
    };
    for (const auto& path : candidates) {
        const std::filesystem::path candidate = test_root / path;
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }

#ifdef BEATIT_TEST_MODEL_PATH
    if (std::filesystem::exists(BEATIT_TEST_MODEL_PATH)) {
        return BEATIT_TEST_MODEL_PATH;
    }
#endif
    return {};
}

}  // namespace beatit::tests

#endif  // BEATIT_TESTS_COREML_TEST_CONFIG_H
