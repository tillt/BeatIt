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
    enum class Backend {
        CoreML,
        BeatThisExternal,
        Torch,
    };
    Backend backend = Backend::CoreML;
    std::string model_path = "models/beatit.mlmodelc";
    std::string input_name = "input";
    std::string beat_output_name = "beat";
    std::string downbeat_output_name = "downbeat";
    std::size_t sample_rate = 44100;
    std::size_t frame_size = 2048;
    std::size_t hop_size = 441;
    std::size_t mel_bins = 81;
    bool use_log_mel = false;
    float log_multiplier = 1.0f;
    float f_min = 0.0f;
    float f_max = 0.0f;
    float power = 2.0f;
    enum class MelScale {
        Htk,
        Slaney,
    };
    MelScale mel_scale = MelScale::Htk;
    enum class SpectrogramNorm {
        None,
        FrameLength,
    };
    SpectrogramNorm spectrogram_norm = SpectrogramNorm::None;
    enum class MelBackend {
        Cpu,
        Torch,
    };
    MelBackend mel_backend = MelBackend::Cpu;
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
    std::size_t window_border_frames = 0;
    float gap_tolerance = 0.05f;
    float offbeat_tolerance = 0.10f;
    std::size_t tempo_window_beats = 8;
    bool use_dbn = false;
    float dbn_bpm_step = 1.0f;
    float dbn_interval_tolerance = 0.05f;
    float dbn_activation_floor = 0.05f;
    std::size_t dbn_beats_per_bar = 4;
    bool dbn_use_downbeat = true;
    float dbn_downbeat_weight = 1.0f;
    float dbn_tempo_change_penalty = 0.0f;
    float dbn_tempo_prior_weight = 0.0f;
    std::size_t dbn_max_candidates = 1024;
    float dbn_transition_reward = 0.5f;
    bool dbn_use_all_candidates = false;
    bool verbose = false;
    bool profile = false;
    bool profile_per_window = false;
    std::string beatthis_python = "python3";
    std::string beatthis_script;
    std::string beatthis_checkpoint;
    bool beatthis_use_dbn = false;
    float beatthis_fps = 100.0f;
    std::string torch_model_path;
    std::string torch_device = "cpu";
    float torch_fps = 100.0f;
    std::size_t torch_batch_size = 1;
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

std::vector<float> compute_mel_features(const std::vector<float>& samples,
                                        double sample_rate,
                                        const CoreMLConfig& config,
                                        std::size_t* out_frames);

} // namespace beatit
