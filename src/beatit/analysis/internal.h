//
//  internal.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/analysis.h"

#include <cstddef>
#include <vector>

namespace beatit {

AnalysisResult analyze_with_beatthis(const std::vector<float>& samples,
                                     double sample_rate,
                                     const CoreMLConfig& config);

std::vector<float> compute_phase_energy(const std::vector<float>& samples,
                                        double sample_rate,
                                        const CoreMLConfig& config);

std::size_t estimate_last_active_frame(const std::vector<float>& samples,
                                       double sample_rate,
                                       const CoreMLConfig& config);

float choose_candidate_bpm(float peaks,
                           float autocorr,
                           float comb,
                           float beats);

void assign_coreml_result(AnalysisResult& result,
                          CoreMLResult&& coreml_result,
                          std::vector<float> phase_energy,
                          double sample_rate,
                          const CoreMLConfig& config);

} // namespace beatit
