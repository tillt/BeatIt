//
//  stream.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/analysis.h"
#include "beatit/coreml.h"

#include <cstddef>
#include <vector>

namespace beatit {

class BeatitStream {
public:
    BeatitStream(double sample_rate,
                 const CoreMLConfig& coreml_config,
                 bool enable_coreml = true);

    void push(const float* samples, std::size_t count);
    void push(const std::vector<float>& samples) { push(samples.data(), samples.size()); }

    AnalysisResult finalize();

private:
    void process_coreml_windows();

    double sample_rate_ = 0.0;
    CoreMLConfig coreml_config_;
    bool coreml_enabled_ = true;

    struct LinearResampler {
        double ratio = 1.0;
        double src_index = 0.0;
        std::vector<float> buffer;

        void push(const float* input, std::size_t count, std::vector<float>* output);
    } resampler_;

    std::vector<float> resampled_buffer_;
    std::size_t resampled_offset_ = 0;
    std::size_t coreml_frame_offset_ = 0;
    std::vector<float> coreml_beat_activation_;
    std::vector<float> coreml_downbeat_activation_;
    std::vector<float> coreml_phase_energy_;
    std::size_t total_input_samples_ = 0;
    std::size_t last_active_sample_ = 0;
    bool has_active_sample_ = false;
    double phase_energy_alpha_ = 0.0;
    double phase_energy_state_ = 0.0;
    double phase_energy_sum_sq_ = 0.0;
    std::size_t phase_energy_sample_count_ = 0;
};

} // namespace beatit
