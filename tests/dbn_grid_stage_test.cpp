//
//  dbn_grid_stage_test.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "post/grid_projection.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool check_uniform_spacing(const std::vector<std::size_t>& frames, std::size_t spacing) {
    if (frames.size() < 4) {
        return false;
    }
    for (std::size_t i = 1; i < frames.size(); ++i) {
        if (frames[i] - frames[i - 1] != spacing) {
            return false;
        }
    }
    return true;
}

bool test_uniform_grid_synthesis() {
    beatit::CoreMLResult result;
    result.beat_activation.assign(160, 0.0f);
    result.downbeat_activation.assign(160, 0.0f);
    for (std::size_t frame = 10; frame < 160; frame += 10) {
        result.beat_activation[frame] = 1.0f;
    }

    beatit::DBNDecodeResult decoded;
    decoded.beat_frames = {10, 21, 31, 40, 50, 60, 70, 80, 90, 100, 110, 121, 130, 140, 150};
    decoded.downbeat_frames = {10, 50, 90, 130};

    beatit::BeatitConfig config;
    config.dbn_grid_global_fit = false;
    beatit::detail::GridProjectionState state;
    state.bpb = 4;
    state.best_phase = 0;
    state.step_frames = 10.0;
    state.activation_floor = 0.1f;
    state.earliest_peak = 10;
    state.earliest_downbeat_peak = 10;
    state.strongest_peak = 10;
    state.strongest_peak_value = 1.0f;
    state.max_downbeat = 0.0f;

    beatit::detail::synthesize_uniform_grid(state, decoded, result, config, 160, 50.0);

    if (!check_uniform_spacing(decoded.beat_frames, 10)) {
        std::cerr << "dbn_grid_stage_test: synthesize_uniform_grid produced non-uniform spacing.\n";
        return false;
    }
    if (decoded.downbeat_frames.size() < 3) {
        std::cerr << "dbn_grid_stage_test: synthesize_uniform_grid produced too few downbeats.\n";
        return false;
    }
    if ((decoded.downbeat_frames[1] - decoded.downbeat_frames[0]) != 40) {
        std::cerr << "dbn_grid_stage_test: downbeat spacing mismatch.\n";
        return false;
    }
    return true;
}

bool test_downbeat_phase_selection() {
    beatit::CoreMLResult result;
    result.beat_activation.assign(200, 0.0f);
    result.downbeat_activation.assign(200, 0.0f);
    for (std::size_t frame = 0; frame < 200; frame += 10) {
        result.beat_activation[frame] = 1.0f;
    }
    for (std::size_t frame = 20; frame < 200; frame += 40) {
        result.downbeat_activation[frame] = 1.0f;
    }

    beatit::DBNDecodeResult decoded;
    for (std::size_t frame = 0; frame < 200; frame += 10) {
        decoded.beat_frames.push_back(frame);
    }

    beatit::BeatitConfig config;
    config.sample_rate = 50;
    config.hop_size = 1;
    config.dbn_downbeat_phase_window_seconds = 3.0f;
    config.dbn_downbeat_phase_max_delay_seconds = 2.0f;
    config.dbn_downbeat_phase_peak_ratio = 0.3f;

    beatit::detail::GridProjectionState state;
    state.bpb = 4;
    state.best_phase = 0;
    state.activation_floor = 0.01f;

    beatit::detail::select_downbeat_phase(state,
                                          decoded,
                                          result,
                                          config,
                                          false,
                                          true,
                                          false,
                                          0,
                                          200,
                                          50.0);

    if (state.best_phase != 2) {
        std::cerr << "dbn_grid_stage_test: expected phase 2, got " << state.best_phase << ".\n";
        return false;
    }
    if (decoded.downbeat_frames.empty() || decoded.downbeat_frames.front() != 20) {
        std::cerr << "dbn_grid_stage_test: expected first downbeat at frame 20.\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!test_uniform_grid_synthesis()) {
        return 1;
    }
    if (!test_downbeat_phase_selection()) {
        return 1;
    }
    std::cout << "DBN grid stage test passed.\n";
    return 0;
}
