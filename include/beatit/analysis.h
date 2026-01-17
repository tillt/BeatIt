//
//  analysis.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/coreml.h"

#include <cstddef>
#include <vector>

namespace beatit {

struct AnalysisResult {
    float estimated_bpm = 0.0f;
    std::vector<float> coreml_beat_activation;
    std::vector<float> coreml_downbeat_activation;
    // Low-band energy per feature frame (diagnostic/phase correction).
    std::vector<float> coreml_phase_energy;
    // Feature-frame indices (CoreML output timeline) for diagnostics/inspection.
    std::vector<unsigned long long> coreml_beat_feature_frames;
    // Sample-frame indices (audio timeline) intended for end-user consumption.
    std::vector<unsigned long long> coreml_beat_sample_frames;
    std::vector<float> coreml_beat_strengths;
    // Feature-frame indices for downbeats (diagnostic timeline).
    std::vector<unsigned long long> coreml_downbeat_feature_frames;
};

AnalysisResult analyze(const std::vector<float>& samples,
                       double sample_rate,
                       const CoreMLConfig& config);

} // namespace beatit
