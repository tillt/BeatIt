//
//  calmdad.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//
// CALMDAD Analyzes Latent Meter Dynamics And Downbeats.
// DBN-based beat and downbeat decoder.
//
#ifndef BEATIT_DBN_CALMDAD_H
#define BEATIT_DBN_CALMDAD_H

#include "beatit/coreml.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace beatit {

struct DBNDecodeResult {
    std::vector<std::size_t> beat_frames;
    std::vector<std::size_t> downbeat_frames;
    double bpm = 0.0;
};

struct DBNPathResult {
    DBNDecodeResult decoded;
    double best_score = std::numeric_limits<double>::lowest();
};

struct CalmdadDecodeRequest {
    const std::vector<float>& beat_activation;
    const std::vector<float>& downbeat_activation;
    double fps = 0.0;
    float min_bpm = 0.0f;
    float max_bpm = 0.0f;
    float bpm_step = 1.0f;
};

struct CalmdadSparseDecodeRequest {
    const std::vector<std::size_t>& candidate_frames;
    const std::vector<double>& beat_log;
    const std::vector<double>& downbeat_log;
    double fps = 0.0;
    float min_bpm = 0.0f;
    float max_bpm = 0.0f;
    float bpm_step = 1.0f;
    std::size_t beats_per_bar = 4;
    double tolerance = 0.0;
    bool use_downbeat = true;
    double transition_reward = 0.0;
    double tempo_change_penalty = 0.0;
};

class CalmdadDecoder {
public:
    explicit CalmdadDecoder(const BeatitConfig& config);

    DBNDecodeResult decode(const CalmdadDecodeRequest& request) const;
    DBNPathResult decode_sparse(const CalmdadSparseDecodeRequest& request) const;

    static std::vector<std::size_t> viterbi_beats(const std::vector<float>& activation,
                                                  double fps,
                                                  double bpm,
                                                  double interval_tolerance,
                                                  float activation_floor);

private:
    BeatitConfig config_;
};

}  // namespace beatit

#endif  // BEATIT_DBN_CALMDAD_H
