//
//  tempo_fit.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/config.h"

#include <cstddef>
#include <vector>

namespace beatit {
class CalmdadDecoder;
}

namespace beatit::detail {

double bpm_from_linear_fit(const std::vector<std::size_t>& beats, double fps);

double bpm_from_global_fit(const CoreMLResult& result,
                           const BeatitConfig& config,
                           const CalmdadDecoder& calmdad_decoder,
                           double fps,
                           float min_bpm,
                           float max_bpm,
                           std::size_t used_frames);

} // namespace beatit::detail
