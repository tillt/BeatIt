//
//  result_ops.h
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

std::size_t refine_frame_to_peak(std::size_t frame,
                                 const std::vector<float>& activation,
                                 std::size_t window);

void dedupe_frames(std::vector<std::size_t>& frames);

void dedupe_frames_tolerant(std::vector<std::size_t>& frames, std::size_t tolerance);

std::vector<std::size_t> apply_latency_to_frames(const std::vector<std::size_t>& frames,
                                                 std::size_t analysis_latency_frames);

void fill_beats_from_frames(CoreMLResult& result,
                            const std::vector<std::size_t>& frames,
                            const CoreMLConfig& config,
                            double sample_rate,
                            double hop_scale,
                            std::size_t analysis_latency_frames,
                            double analysis_latency_frames_f,
                            std::size_t refine_window);

void fill_beats_from_bpm_grid_into(const std::vector<float>& beat_activation,
                                   const CoreMLConfig& config,
                                   double sample_rate,
                                   double fps,
                                   double hop_scale,
                                   std::size_t start_frame,
                                   double bpm,
                                   std::size_t total_frames,
                                   std::vector<unsigned long long>& out_feature_frames,
                                   std::vector<unsigned long long>& out_sample_frames,
                                   std::vector<float>& out_strengths);

} // namespace beatit::detail
