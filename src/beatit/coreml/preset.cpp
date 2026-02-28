//
//  preset.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/coreml_preset.h"

#include <algorithm>
#include <cctype>

namespace beatit {
namespace {

class BeatTrackPreset : public CoreMLPreset {
public:
    const char* name() const override {
        return "beattrack";
    }

    void apply(BeatitConfig& config) const override {
        config.sample_rate = 44100;
        config.frame_size = 2048;
        config.hop_size = 441;
        config.mel_bins = 81;
        config.use_log_mel = false;
        config.log_multiplier = 1.0f;
        config.f_min = 0.0f;
        config.f_max = 0.0f;
        config.power = 2.0f;
        config.mel_scale = BeatitConfig::MelScale::Htk;
        config.spectrogram_norm = BeatitConfig::SpectrogramNorm::None;
        config.input_layout = BeatitConfig::InputLayout::ChannelsFramesMels;
        config.fixed_frames = 3000;
        config.window_hop_frames = 2500;
        config.window_border_frames = 0;
        config.analysis_start_seconds = 0.0;
    }
};

class BeatThisPreset : public CoreMLPreset {
public:
    const char* name() const override {
        return "beatthis";
    }

    void apply(BeatitConfig& config) const override {
        config.backend = BeatitConfig::Backend::CoreML;
        config.sample_rate = 22050;
        config.frame_size = 1024;
        config.hop_size = 441;
        config.mel_bins = 128;
        config.use_log_mel = true;
        config.log_multiplier = 1000.0f;
        config.f_min = 30.0f;
        config.f_max = 11000.0f;
        config.power = 1.0f;
        config.mel_scale = BeatitConfig::MelScale::Slaney;
        config.spectrogram_norm = BeatitConfig::SpectrogramNorm::FrameLength;
        config.input_layout = BeatitConfig::InputLayout::FramesByMels;
        config.fixed_frames = 1500;
        config.window_hop_frames = 1488;
        config.window_border_frames = 6;
        const double frame_size_d = static_cast<double>(config.frame_size);
        const double fixed_frames_d = static_cast<double>(config.fixed_frames);
        const double hop_size_d = static_cast<double>(config.hop_size);
        const double sample_rate_d = static_cast<double>(config.sample_rate);
        const double model_window_seconds =
            (frame_size_d + ((fixed_frames_d - 1.0) * hop_size_d)) / sample_rate_d;
        config.analysis_start_seconds = 0.0;
        config.min_bpm = 70.0f;
        config.max_bpm = 180.0f;
        config.output_latency_seconds = 0.016;
        config.use_dbn = true;
        config.logit_phase_window_seconds = 2.0;
        config.logit_phase_max_shift_seconds = 0.03f;
        config.logit_min_peaks = 8;
        config.disable_silence_trimming = true;
        config.dbn_use_downbeat = true;
        config.dbn_mode = BeatitConfig::DBNMode::Calmdad;
        config.dbn_activation_floor = 0.7f;
        config.dbn_downbeat_phase_peak_ratio = 0.2f;
        config.dbn_downbeat_phase_window_seconds = 2.0;
        config.dbn_downbeat_phase_max_delay_seconds = 0.3;
        config.use_minimal_postprocess = true;
        config.dbn_window_seconds = model_window_seconds;
        config.dbn_window_start_seconds = 0.0;
        config.dbn_tempo_anchor_intro_seconds = 10.0;
        config.dbn_tempo_anchor_outro_offset_seconds = 10.0;
        config.dbn_project_grid = true;
        config.dbn_grid_global_fit = true;
        config.dbn_trace = false;
        config.max_analysis_seconds = 0.0;
        config.dbn_window_start_seconds = 0.0;
        config.sparse_probe_mode = true;
        config.torch_batch_size = 1;
        config.prepend_silence_seconds = 1.0;
        config.model_path = "coreml_out/BeatThis_small0.mlpackage";
        config.input_name = "mel_spectrogram";
        config.beat_output_name = "beat_logits";
        config.downbeat_output_name = "downbeat_logits";
        config.coreml_output_logits = false;
        config.coreml_logit_temperature = 1.0f;
    }
};

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

std::unique_ptr<CoreMLPreset> make_coreml_preset(const std::string& name) {
    const std::string key = to_lower(name);
    if (key == "beattrack") {
        return std::make_unique<BeatTrackPreset>();
    }
    if (key == "beatthis") {
        return std::make_unique<BeatThisPreset>();
    }
    return nullptr;
}

std::vector<std::string> coreml_preset_names() {
    return {"beattrack", "beatthis"};
}

} // namespace beatit
