//
//  constant_internal.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/refiner.h"

#include <cstddef>
#include <vector>

namespace beatit {
namespace detail {

struct BeatConstRegion {
    std::size_t first_beat_frame = 0;
    double beat_length = 0.0;
};

struct DownbeatPhaseSource {
    bool use_model = false;
    const std::vector<float>* activation = nullptr;
    double coverage = 0.0;
};

double round_bpm_within_range(double min_bpm, double center_bpm, double max_bpm);

std::vector<BeatConstRegion> retrieve_constant_regions(const std::vector<unsigned long long>& coarse_beats,
                                                       double frame_rate,
                                                       const ConstantBeatRefinerConfig& config);

double make_constant_bpm(const std::vector<BeatConstRegion>& regions,
                         double frame_rate,
                         const ConstantBeatRefinerConfig& config,
                         long long* out_first_beat,
                         std::size_t initial_silence_frames);

long long adjust_phase(long long first_beat,
                       double bpm,
                       const std::vector<unsigned long long>& coarse_beats,
                       double frame_rate,
                       const ConstantBeatRefinerConfig& config);

double compute_local_peak(const std::vector<float>* activation,
                          unsigned long long frame,
                          std::size_t window);

bool is_found_peak(const std::vector<unsigned long long>& coarse_beats,
                   std::size_t* coarse_index,
                   unsigned long long frame,
                   unsigned long long tolerance_frames);

double score_phase_shift(const std::vector<unsigned long long>& beat_frames,
                         const std::vector<float>* energy,
                         long long shift_frames,
                         std::size_t total_frames);

long long choose_half_beat_shift(const std::vector<unsigned long long>& beat_frames,
                                 const std::vector<float>* energy,
                                 long long half_beat_frames,
                                 std::size_t total_frames,
                                 double margin);

std::size_t choose_downbeat_phase(const std::vector<unsigned long long>& beat_frames,
                                  const std::vector<float>* downbeat_activation,
                                  const std::vector<float>* phase_energy,
                                  double low_freq_weight);

std::size_t last_active_frame(const std::vector<float>* energy,
                              std::size_t total_frames,
                              double active_energy_ratio);

std::size_t first_active_frame(const std::vector<float>* energy,
                               std::size_t total_frames,
                               double active_energy_ratio);

DownbeatPhaseSource choose_downbeat_phase_source(
    const std::vector<unsigned long long>& beat_feature_frames,
    const std::vector<float>* downbeat_activation,
    const ConstantBeatRefinerConfig& refiner_config);

void apply_marker_style(BeatEvent& event,
                        std::size_t beat_index,
                        std::size_t intro_beats_starting_at,
                        std::size_t buildup_beats_starting_at,
                        std::size_t teardown_beats_starting_at,
                        std::size_t outro_beats_starting_at);

void apply_alarm_style(BeatEvent& event,
                       std::size_t beat_index,
                       std::size_t intro_beats_starting_at,
                       std::size_t buildup_beats_starting_at,
                       std::size_t teardown_beats_starting_at,
                       std::size_t outro_beats_starting_at);

void summarize_found_stats(ConstantBeatResult& result);

void fill_event_exports(ConstantBeatResult& result);

} // namespace detail
} // namespace beatit
