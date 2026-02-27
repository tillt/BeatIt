//
//  logits.h
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

void run_logit_consensus_postprocess(CoreMLResult& result,
                                     const std::vector<float>* phase_energy,
                                     const BeatitConfig& config,
                                     double sample_rate,
                                     float min_bpm,
                                     float max_bpm,
                                     std::size_t grid_total_frames,
                                     double fps,
                                     double hop_scale,
                                     std::size_t analysis_latency_frames,
                                     double analysis_latency_frames_f,
                                     double dbn_ms,
                                     double peaks_ms);

} // namespace beatit::detail
