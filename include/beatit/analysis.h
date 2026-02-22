//
//  analysis.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/coreml.h"
#include "beatit/refiner.h"

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
    // Projected grid feature-frame indices (if enabled).
    std::vector<unsigned long long> coreml_beat_projected_feature_frames;
    // Projected grid sample-frame indices (if enabled).
    std::vector<unsigned long long> coreml_beat_projected_sample_frames;
    // Beat events with markers derived from the beat grid.
    std::vector<BeatEvent> coreml_beat_events;
    std::vector<float> coreml_beat_strengths;
    // Feature-frame indices for downbeats (diagnostic timeline).
    std::vector<unsigned long long> coreml_downbeat_feature_frames;
    // Projected grid downbeat feature-frame indices (if enabled).
    std::vector<unsigned long long> coreml_downbeat_projected_feature_frames;
};

AnalysisResult analyze(const std::vector<float>& samples,
                       double sample_rate,
                       const CoreMLConfig& config);

// Diagnostics/helpers used by CLI tooling.
float estimate_bpm_from_activation(const std::vector<float>& activation,
                                   const CoreMLConfig& config,
                                   double sample_rate);
float estimate_bpm_from_activation_autocorr(const std::vector<float>& activation,
                                            const CoreMLConfig& config,
                                            double sample_rate);
float estimate_bpm_from_activation_comb(const std::vector<float>& activation,
                                        const CoreMLConfig& config,
                                        double sample_rate);
float estimate_bpm_from_beats(const std::vector<unsigned long long>& beat_samples,
                              double sample_rate);
float normalize_bpm_to_range(float bpm, float min_bpm, float max_bpm);

const std::vector<unsigned long long>& output_beat_feature_frames(const AnalysisResult& result);

const std::vector<unsigned long long>& output_beat_sample_frames(const AnalysisResult& result);

const std::vector<unsigned long long>& output_downbeat_feature_frames(const AnalysisResult& result);

void rebuild_output_beat_events(AnalysisResult* result,
                                double sample_rate,
                                const CoreMLConfig& config);

} // namespace beatit
