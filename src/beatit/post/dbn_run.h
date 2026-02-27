//
//  dbn_run.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/config.h"

#include <cstddef>
#include <vector>

namespace beatit::detail {

struct DBNRunRequest {
    CoreMLResult& result;
    const std::vector<float>* phase_energy = nullptr;
    const BeatitConfig& config;
    double sample_rate = 0.0;
    float reference_bpm = 0.0f;
    std::size_t grid_total_frames = 0;
    float min_bpm = 0.0f;
    float max_bpm = 0.0f;
    double fps = 0.0;
    double hop_scale = 1.0;
    std::size_t analysis_latency_frames = 0;
    double analysis_latency_frames_f = 0.0;
    double peaks_ms = 0.0;
    double& dbn_ms;
};

bool run_dbn_postprocess(const DBNRunRequest& request);

} // namespace beatit::detail
