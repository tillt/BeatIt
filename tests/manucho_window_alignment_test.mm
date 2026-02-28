//
//  manucho_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Manucho";
    cfg.audio_filename = "manucho.wav";
    cfg.dump_env_var = "BEATIT_MANUCHO_DUMP_EVENTS";

    cfg.edge_window_beats = 64;
    cfg.alternation_window_beats = 24;
    cfg.tempo_edge_intervals = 64;
    cfg.event_probe_count = 16;

    cfg.expected_first_downbeat_sample_frame = 2309ULL;
    cfg.first_downbeat_sample_tolerance_ms = 10.0;

    cfg.target_bpm = 109.998;
    cfg.max_bpm_error = 0.01;

    cfg.max_intro_median_abs_ms = 9.0;
    cfg.max_offset_slope_ms_per_beat = 0.015;
    cfg.max_start_end_delta_ms = 9.0;
    cfg.max_start_end_delta_beats = 0.016;
    cfg.max_odd_even_median_gap_ms = 9.0;
    cfg.max_tempo_edge_bpm_delta = 0.0005;

    cfg.check_seed_order = true;
    cfg.max_seed_order_bpm_delta = 0.01;
    cfg.max_seed_order_grid_median_delta_frames = 1.0;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
