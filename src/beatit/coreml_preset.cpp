//
//  coreml_preset.cpp
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

    void apply(CoreMLConfig& config) const override {
        config.sample_rate = 44100;
        config.frame_size = 2048;
        config.hop_size = 441;
        config.mel_bins = 81;
        config.use_log_mel = false;
        config.log_multiplier = 1.0f;
        config.f_min = 0.0f;
        config.f_max = 0.0f;
        config.power = 2.0f;
        config.mel_scale = CoreMLConfig::MelScale::Htk;
        config.spectrogram_norm = CoreMLConfig::SpectrogramNorm::None;
        config.input_layout = CoreMLConfig::InputLayout::ChannelsFramesMels;
        config.fixed_frames = 3000;
        config.window_hop_frames = 2500;
        config.window_border_frames = 0;
    }
};

class BeatThisPreset : public CoreMLPreset {
public:
    const char* name() const override {
        return "beatthis";
    }

    void apply(CoreMLConfig& config) const override {
        config.sample_rate = 22050;
        config.frame_size = 1024;
        config.hop_size = 441;
        config.mel_bins = 128;
        config.use_log_mel = true;
        config.log_multiplier = 1000.0f;
        config.f_min = 30.0f;
        config.f_max = 11000.0f;
        config.power = 1.0f;
        config.mel_scale = CoreMLConfig::MelScale::Slaney;
        config.spectrogram_norm = CoreMLConfig::SpectrogramNorm::FrameLength;
        config.input_layout = CoreMLConfig::InputLayout::FramesByMels;
        config.fixed_frames = 1500;
        config.window_hop_frames = 1488;
        config.window_border_frames = 6;
        config.use_dbn = true;
        config.dbn_use_downbeat = true;
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
