//
//  coreml_postprocess_dbn_grid_stages.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#ifndef BEATIT_INTERNAL_COREML_POSTPROCESS_DBN_GRID_STAGES_H
#define BEATIT_INTERNAL_COREML_POSTPROCESS_DBN_GRID_STAGES_H

#include "beatit/coreml.h"
#include "dbn_beatit.h"

#include <cstddef>
#include <limits>

namespace beatit::detail {

struct GridProjectionState {
    std::size_t bpb = 1;
    std::size_t phase = 0;
    std::size_t best_phase = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    double base_interval = 0.0;
    double step_frames = 0.0;
    std::size_t earliest_peak = 0;
    std::size_t earliest_downbeat_peak = 0;
    std::size_t strongest_peak = 0;
    float strongest_peak_value = -1.0f;
    float activation_floor = 0.0f;
    float max_downbeat = 0.0f;
};

void select_downbeat_phase(GridProjectionState& state,
                           DBNDecodeResult& decoded,
                           const CoreMLResult& result,
                           const CoreMLConfig& config,
                           bool quality_low,
                           bool downbeat_override_ok,
                           bool use_window,
                           std::size_t window_start,
                           std::size_t used_frames,
                           double fps);

void synthesize_uniform_grid(GridProjectionState& state,
                             DBNDecodeResult& decoded,
                             const CoreMLResult& result,
                             const CoreMLConfig& config,
                             std::size_t used_frames,
                             double fps);

} // namespace beatit::detail

#endif // BEATIT_INTERNAL_COREML_POSTPROCESS_DBN_GRID_STAGES_H
