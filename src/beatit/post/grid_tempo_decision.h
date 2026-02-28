//
//  grid_tempo_decision.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#ifndef BEATIT_INTERNAL_POSTPROCESS_GRID_TEMPO_DECISION_H
#define BEATIT_INTERNAL_POSTPROCESS_GRID_TEMPO_DECISION_H

#include "beatit/config.h"
#include "beatit/post/helpers.h"
#include "beatit/dbn/calmdad.h"
#include "../dbn/beatit.h"

#include <cstddef>
#include <string>
#include <vector>

namespace beatit::detail {

struct GridTempoDecision {
    struct Diagnostics {
        IntervalStats tempo_stats;
        IntervalStats decoded_stats;
        IntervalStats decoded_filtered_stats;
        IntervalStats downbeat_stats;

        double bpm_from_peaks = 0.0;
        double bpm_from_peaks_median = 0.0;
        double bpm_from_peaks_reg_full = 0.0;
        double bpm_from_downbeats = 0.0;
        double bpm_from_fit = 0.0;
        double bpm_from_global_fit = 0.0;

        std::size_t downbeat_count = 0;
        double downbeat_cv = 0.0;

        std::string bpm_source = "none";
    };

    std::size_t bpb = 1;
    std::size_t phase = 0;
    double base_interval = 0.0;
    double bpm_for_grid = 0.0;
    double step_frames = 0.0;
    bool quality_low = false;
    bool downbeat_override_ok = false;
    Diagnostics diagnostics;
};

struct GridTempoDecisionInput {
    const DBNDecodeResult& decoded;
    const CoreMLResult& result;
    const BeatitConfig& config;
    const CalmdadDecoder& calmdad_decoder;

    bool use_window = false;
    const std::vector<float>& beat_slice;
    const std::vector<float>& downbeat_slice;

    bool quality_valid = false;
    double quality_qkur = 0.0;

    std::size_t used_frames = 0;
    float min_bpm = 0.0f;
    float max_bpm = 0.0f;
    float reference_bpm = 0.0f;
    double fps = 0.0;
};

GridTempoDecision compute_grid_tempo_decision(const GridTempoDecisionInput& input);

void log_grid_tempo_decision(const GridTempoDecision& decision,
                             const GridTempoDecisionInput& input);

} // namespace beatit::detail

#endif // BEATIT_INTERNAL_POSTPROCESS_GRID_TEMPO_DECISION_H
