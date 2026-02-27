//
//  grid_anchor.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#ifndef BEATIT_INTERNAL_POSTPROCESS_GRID_ANCHOR_H
#define BEATIT_INTERNAL_POSTPROCESS_GRID_ANCHOR_H

#include "beatit/config.h"
#include "../dbn/beatit.h"

#include <cstddef>

namespace beatit::detail {

struct GridAnchorSeed {
    std::size_t earliest_peak = 0;
    std::size_t earliest_downbeat_peak = 0;
    std::size_t strongest_peak = 0;
    float strongest_peak_value = -1.0f;
    float activation_floor = 0.0f;
};

GridAnchorSeed seed_grid_anchor(DBNDecodeResult& decoded,
                                const CoreMLResult& result,
                                const BeatitConfig& config,
                                bool use_window,
                                std::size_t window_start,
                                std::size_t used_frames,
                                double base_interval);

} // namespace beatit::detail

#endif // BEATIT_INTERNAL_POSTPROCESS_GRID_ANCHOR_H
