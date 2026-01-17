//
//  coreml.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace beatit {

struct CoreMLConfig {
    std::string model_path = "models/beatit.mlmodelc";
    std::string input_name = "input";
    std::string beat_output_name = "beat";
    std::string downbeat_output_name = "downbeat";
    std::size_t sample_rate = 44100;
    std::size_t frame_size = 2048;
    std::size_t hop_size = 441;
    std::size_t mel_bins = 81;
    bool use_log_mel = false;
    enum class InputLayout {
        FramesByMels,
        ChannelsFramesMels,
    };
    InputLayout input_layout = InputLayout::ChannelsFramesMels;
    enum class ComputeUnits {
        All,
        CPUAndGPU,
        CPUOnly,
        CPUAndNeuralEngine,
    };
    ComputeUnits compute_units = ComputeUnits::All;
    std::size_t fixed_frames = 3000;
    std::size_t window_hop_frames = 1500;
    float min_bpm = 55.0f;
    float max_bpm = 215.0f;
    float tempo_window_percent = 20.0f;
    bool prefer_double_time = true;
    float activation_threshold = 0.5f;
    bool synthetic_fill = false;
    bool pad_final_window = true;
    float gap_tolerance = 0.05f;
    float offbeat_tolerance = 0.10f;
    std::size_t tempo_window_beats = 8;
    bool verbose = false;
};

struct CoreMLResult {
    std::vector<float> beat_activation;
    std::vector<float> downbeat_activation;
    // Feature-frame indices (CoreML output timeline) for diagnostics/inspection.
    std::vector<unsigned long long> beat_feature_frames;
    // Sample-frame indices (audio timeline) intended for end-user consumption.
    std::vector<unsigned long long> beat_sample_frames;
    std::vector<float> beat_strengths;
    // Feature-frame indices for downbeats (diagnostic timeline).
    std::vector<unsigned long long> downbeat_feature_frames;
};

struct CoreMLMetadata {
    std::string author;
    std::string short_description;
    std::string license;
    std::string version;
    std::vector<std::pair<std::string, std::string>> user_defined;
};

CoreMLResult analyze_with_coreml(const std::vector<float>& samples,
                                 double sample_rate,
                                 const CoreMLConfig& config,
                                 float reference_bpm);

CoreMLResult postprocess_coreml_activations(const std::vector<float>& beat_activation,
                                            const std::vector<float>& downbeat_activation,
                                            const CoreMLConfig& config,
                                            double sample_rate,
                                            float reference_bpm,
                                            std::size_t last_active_frame = 0);

CoreMLMetadata load_coreml_metadata(const CoreMLConfig& config);

} // namespace beatit
