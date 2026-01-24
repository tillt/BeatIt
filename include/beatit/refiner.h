//
//  refiner.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/coreml.h"

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace beatit {

enum BeatEventStyle : unsigned long long {
    BeatEventStyleBeat = 1ULL << 0,
    BeatEventStyleBar = 1ULL << 1,

    BeatEventStyleFound = 1ULL << 2,

    BeatEventStyleAlarmIntro = 1ULL << 3,
    BeatEventStyleAlarmBuildup = 1ULL << 4,
    BeatEventStyleAlarmTeardown = 1ULL << 5,
    BeatEventStyleAlarmOutro = 1ULL << 6,

    BeatEventStyleMarkIntro = 1ULL << 7,
    BeatEventStyleMarkBuildup = 1ULL << 8,
    BeatEventStyleMarkTeardown = 1ULL << 9,
    BeatEventStyleMarkOutro = 1ULL << 10,

    BeatEventStyleMarkStart = 1ULL << 11,
    BeatEventStyleMarkEnd = 1ULL << 12,
    BeatEventStyleMarkChapter = 1ULL << 13,
};

extern const unsigned long long BeatEventMaskMarkers;

struct BeatEvent {
    BeatEventStyle style = BeatEventStyleBeat;
    // Sample-frame index on the audio timeline.
    unsigned long long frame = 0;
    double bpm = 0.0;
    std::size_t index = 0;
    double energy = 0.0;
    double peak = 0.0;
};

// CSV helpers for debug output.
std::string beat_event_csv_header();
std::string beat_event_to_csv(const BeatEvent& event);
std::string beat_event_csv_preamble(std::size_t downbeat_phase,
                                    long long phase_shift_frames,
                                    bool used_model_downbeat,
                                    double downbeat_coverage,
                                    std::size_t active_start_frame,
                                    std::size_t active_end_frame,
                                    std::size_t found_count,
                                    std::size_t total_count,
                                    std::size_t max_missing_run,
                                    double avg_missing_run);
void write_beat_events_csv(std::ostream& out,
                           const std::vector<BeatEvent>& events,
                           bool include_header = true,
                           std::size_t downbeat_phase = 0,
                           long long phase_shift_frames = 0,
                           bool used_model_downbeat = false,
                           double downbeat_coverage = 0.0,
                           std::size_t active_start_frame = 0,
                           std::size_t active_end_frame = 0,
                           std::size_t found_count = 0,
                           std::size_t total_count = 0,
                           std::size_t max_missing_run = 0,
                           double avg_missing_run = 0.0);

std::vector<BeatEvent> build_shakespear_markers(const std::vector<unsigned long long>& beat_feature_frames,
                                                const std::vector<unsigned long long>& beat_sample_frames,
                                                const std::vector<unsigned long long>& downbeat_feature_frames,
                                                const std::vector<float>* beat_activation,
                                                double bpm,
                                                double sample_rate,
                                                const CoreMLConfig& config);

struct ConstantBeatRefinerConfig {
    double max_phase_error_seconds = 0.025;
    double max_phase_error_sum_seconds = 0.1;
    std::size_t max_outliers = 1;
    std::size_t min_region_beats = 10;
    bool use_downbeat_anchor = false;
    bool use_half_beat_correction = false;
    double half_beat_score_margin = 0.05;
    bool auto_active_trim = true;
    bool auto_downbeat_fallback = true;
    double downbeat_min_coverage = 0.2;
    double downbeat_min_strength_ratio = 0.3;
    // Weight applied to low-frequency (phase) energy when selecting bar phase.
    double low_freq_weight = 0.5;
    // Ratio of peak energy used to detect the last active frame.
    double active_energy_ratio = 0.01;
};

struct ConstantBeatResult {
    float bpm = 0.0f;
    // Feature-frame indices (CoreML output timeline) for diagnostics/inspection.
    std::vector<unsigned long long> beat_feature_frames;
    // Sample-frame indices (audio timeline) intended for end-user consumption.
    std::vector<unsigned long long> beat_sample_frames;
    // Rich beat events with style flags and annotations.
    std::vector<BeatEvent> beat_events;
    std::vector<float> beat_strengths;
    // Feature-frame indices for downbeats (diagnostic timeline).
    std::vector<unsigned long long> downbeat_feature_frames;
    long long first_beat_feature_frame = 0;
    std::size_t downbeat_phase = 0;
    long long phase_shift_frames = 0;
    bool used_model_downbeat = false;
    double downbeat_coverage = 0.0;
    std::size_t active_start_frame = 0;
    std::size_t active_end_frame = 0;
    std::size_t found_count = 0;
    std::size_t total_count = 0;
    std::size_t max_missing_run = 0;
    double found_ratio = 0.0;
    double avg_missing_run = 0.0;
};

ConstantBeatResult refine_constant_beats(const std::vector<unsigned long long>& beat_feature_frames,
                                         std::size_t total_frames,
                                         const CoreMLConfig& config,
                                         double sample_rate,
                                         std::size_t initial_silence_frames = 0,
                                         const ConstantBeatRefinerConfig& refiner_config = ConstantBeatRefinerConfig(),
                                         const std::vector<float>* beat_activation = nullptr,
                                         const std::vector<float>* downbeat_activation = nullptr,
                                         const std::vector<float>* phase_energy = nullptr);

} // namespace beatit
