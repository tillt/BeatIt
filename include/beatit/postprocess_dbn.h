//
//  postprocess_dbn.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/coreml.h"

#include <cstddef>
#include <vector>

namespace beatit::detail {

bool run_dbn_postprocess(CoreMLResult& result,
                         const std::vector<float>* phase_energy,
                         const CoreMLConfig& config,
                         double sample_rate,
                         float reference_bpm,
                         std::size_t grid_total_frames,
                         float min_bpm,
                         float max_bpm,
                         double fps,
                         double hop_scale,
                         std::size_t analysis_latency_frames,
                         double analysis_latency_frames_f,
                         double& dbn_ms,
                         double peaks_ms);

} // namespace beatit::detail
