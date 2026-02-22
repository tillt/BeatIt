//
//  postprocess_window.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/coreml.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace beatit::detail {

struct WindowSummary {
    std::size_t frames = 0;
    std::size_t above = 0;
    float min = 0.0f;
    float max = 0.0f;
    double mean = 0.0;
};

double window_tempo_score(const std::vector<float>& activation,
                          std::size_t start,
                          std::size_t end,
                          float min_bpm,
                          float max_bpm,
                          float peak_threshold,
                          double fps);

std::pair<std::size_t, std::size_t> select_dbn_window(const std::vector<float>& activation,
                                                       double window_seconds,
                                                       bool intro_mid_outro,
                                                       float min_bpm,
                                                       float max_bpm,
                                                       float peak_threshold,
                                                       double fps);

std::pair<std::size_t, std::size_t> select_dbn_window_energy(const std::vector<float>& energy,
                                                              double window_seconds,
                                                              bool intro_mid_outro,
                                                              double fps);

std::vector<std::size_t> deduplicate_peaks(const std::vector<std::size_t>& peaks,
                                           std::size_t width);

std::vector<std::size_t> compute_minimal_peaks(const std::vector<float>& activation);

std::vector<std::size_t> align_downbeats_to_beats(const std::vector<std::size_t>& beats,
                                                  const std::vector<std::size_t>& downbeats);

std::pair<std::size_t, std::size_t> infer_bpb_phase(const std::vector<std::size_t>& beats,
                                                     const std::vector<std::size_t>& downbeats,
                                                     const std::vector<std::size_t>& candidates,
                                                     const CoreMLConfig& config);

std::vector<std::size_t> project_downbeats_from_beats(const std::vector<std::size_t>& beats,
                                                      std::size_t bpb,
                                                      std::size_t phase);

std::size_t guard_projected_downbeat_phase(const std::vector<std::size_t>& projected_frames,
                                           const std::vector<float>& downbeat_activation,
                                           std::size_t projected_bpb,
                                           std::size_t inferred_phase,
                                           bool verbose);

WindowSummary summarize_window(const std::vector<float>& activation,
                              std::size_t start,
                              std::size_t end,
                              float floor_value);

double median_interval_bpm(const std::vector<std::size_t>& frames, double fps);

} // namespace beatit::detail
