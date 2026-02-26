//
//  coreml.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#ifndef BEATIT_COREML_H
#define BEATIT_COREML_H

#include "beatit/logging.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace beatit {

struct BeatitConfig {
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
    std::size_t sample_rate = 22050;
    std::size_t frame_size = 1024;
    std::size_t hop_size = 441;
    double output_latency_seconds = 0.016;
    std::size_t mel_bins = 128;
    bool use_log_mel = true;
    float log_multiplier = 1000.0f;
    float f_min = 30.0f;
    float f_max = 11000.0f;
    float power = 1.0f;
    enum class MelScale {
        Htk,
        Slaney,
    };
    MelScale mel_scale = MelScale::Slaney;
    enum class SpectrogramNorm {
        None,
        FrameLength,
    };
    SpectrogramNorm spectrogram_norm = SpectrogramNorm::FrameLength;
    enum class MelBackend {
        Cpu,
        Torch,
    };
    MelBackend mel_backend = MelBackend::Cpu;
    enum class InputLayout {
        FramesByMels,
        ChannelsFramesMels,
    };
    InputLayout input_layout = InputLayout::FramesByMels;
    enum class ComputeUnits {
        All,
        CPUAndGPU,
        CPUOnly,
        CPUAndNeuralEngine,
    };
    ComputeUnits compute_units = ComputeUnits::All;
    std::size_t fixed_frames = 1500;
    std::size_t window_hop_frames = 1488;
    float min_bpm = 70.0f;
    float max_bpm = 180.0f;
    float tempo_window_percent = 20.0f;
    bool prefer_double_time = true;
    float activation_threshold = 0.5f;
    bool synthetic_fill = false;
    bool pad_final_window = true;
    std::size_t window_border_frames = 6;
    float gap_tolerance = 0.05f;
    float offbeat_tolerance = 0.10f;
    std::size_t tempo_window_beats = 8;
    bool use_dbn = true;
    double logit_phase_window_seconds = 2.0;
    float logit_phase_max_shift_seconds = 0.03f;
    std::size_t logit_min_peaks = 8;
    bool disable_silence_trimming = false;
    enum class DBNMode {
        Beatit,
        Calmdad,
    };
    DBNMode dbn_mode = DBNMode::Calmdad;
    float dbn_bpm_step = 1.0f;
    float dbn_interval_tolerance = 0.05f;
    float dbn_activation_floor = 0.7f;
    std::size_t dbn_beats_per_bar = 4;
    bool dbn_use_downbeat = true;
    float dbn_downbeat_weight = 1.0f;
    float dbn_downbeat_phase_peak_ratio = 0.2f;
    double dbn_downbeat_phase_window_seconds = 2.0;
    double dbn_downbeat_phase_max_delay_seconds = 0.3;
    float dbn_tempo_change_penalty = 0.05f;
    float dbn_tempo_prior_weight = 0.0f;
    std::size_t dbn_max_candidates = 4096;
    float dbn_transition_reward = 0.7f;
    bool dbn_use_all_candidates = true;
    float dbn_transition_lambda = 100.0f;
    bool dbn_trace = false;
    bool use_minimal_postprocess = true;
    double dbn_window_seconds = 60.0;
    // Global analysis start offset applied before any windowing (seconds).
    double analysis_start_seconds = 0.0;

    // DBN-only window start offset (seconds).
    double dbn_window_start_seconds = 0.0;
    // Sparse tempo anchors for global BPM fit when multi-window mode is enabled.
    double dbn_tempo_anchor_intro_seconds = 10.0;
    double dbn_tempo_anchor_outro_offset_seconds = 10.0;
    bool dbn_project_grid = true;
    // Run sparse multi-window probing (single-switch mode) instead of full activation inference.
    // When enabled, BeatIt auto-enables the required internal DBN/consensus logic.
    bool sparse_probe_mode = false;
    // Fit one global linear beat grid (frame ~= a + i*b) before projection.
    bool dbn_grid_global_fit = false;
    bool dbn_grid_align_downbeat_peak = true;
    bool dbn_grid_start_strong_peak = true;
    float dbn_grid_start_advance_seconds = 0.06f;
    double max_analysis_seconds = 60.0;
    double prepend_silence_seconds = 1.0;
    double debug_activations_start_s = -1.0;
    double debug_activations_end_s = -1.0;
    std::size_t debug_activations_max = 0;
    LogVerbosity log_verbosity = LogVerbosity::Info;
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
    bool coreml_output_logits = false;
    float coreml_logit_temperature = 1.0f;
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
    // Projected grid (feature frames) when dbn_project_grid is enabled.
    std::vector<unsigned long long> beat_projected_feature_frames;
    // Projected grid (sample frames) when dbn_project_grid is enabled.
    std::vector<unsigned long long> beat_projected_sample_frames;
    std::vector<float> beat_projected_strengths;
    // Projected grid downbeats (feature frames) when dbn_project_grid is enabled.
    std::vector<unsigned long long> downbeat_projected_feature_frames;
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
                                 const BeatitConfig& config,
                                 float reference_bpm);

CoreMLResult postprocess_coreml_activations(const std::vector<float>& beat_activation,
                                            const std::vector<float>& downbeat_activation,
                                            const std::vector<float>* phase_energy,
                                            const BeatitConfig& config,
                                            double sample_rate,
                                            float reference_bpm,
                                            std::size_t last_active_frame = 0,
                                            std::size_t total_frames_full = 0);

CoreMLMetadata load_coreml_metadata(const BeatitConfig& config);

std::vector<float> compute_mel_features(const std::vector<float>& samples,
                                        double sample_rate,
                                        const BeatitConfig& config,
                                        std::size_t* out_frames);

} // namespace beatit

#endif  // BEATIT_COREML_H
