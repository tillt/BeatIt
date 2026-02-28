//
//  neural_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Neural";
    cfg.audio_filename = "neural.wav";
    cfg.dump_env_var = "BEATIT_NEURAL_DUMP_EVENTS";

    cfg.edge_window_beats = 64;
    cfg.alternation_window_beats = 24;
    cfg.tempo_edge_intervals = 64;
    cfg.drift_probe_count = 24;
    cfg.event_probe_count = 16;

    cfg.min_expected_bpm = 70.0;
    cfg.max_expected_bpm = 180.0;
    cfg.require_first_bar_complete = false;

    cfg.max_intro_median_abs_ms = 25.0;
    cfg.max_offset_slope_ms_per_beat = 0.04;
    cfg.max_unwrapped_slope_ms_per_beat = 0.06;
    cfg.max_start_end_delta_ms = 45.0;
    cfg.max_start_end_delta_beats = 0.08;
    cfg.max_unwrapped_start_end_delta_beats = 0.08;
    cfg.max_odd_even_median_gap_ms = 15.0;
    cfg.max_tempo_edge_bpm_delta = 0.01;

    cfg.use_interior_windows = true;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
