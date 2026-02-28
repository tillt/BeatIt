//
//  samerano_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Samerano";
    cfg.audio_filename = "samerano.wav";
    cfg.dump_env_var = "BEATIT_SAMERANO_DUMP_EVENTS";

    cfg.edge_window_beats = 64;
    cfg.alternation_window_beats = 24;
    cfg.tempo_edge_intervals = 64;
    cfg.event_probe_count = 16;

    cfg.min_expected_bpm = 122.3;
    cfg.max_expected_bpm = 122.7;

    cfg.max_intro_median_abs_ms = 45.0;
    cfg.max_offset_slope_ms_per_beat = 0.05;
    cfg.max_start_end_delta_ms = 60.0;
    cfg.max_start_end_delta_beats = 0.12;

    cfg.local_offset_windows = {
        {"start", 0.02, 45.0},
        {"quarter", 0.25, 90.0},
        {"middle", 0.50, 70.0},
        {"three_quarter", 0.75, 70.0},
        {"late", 0.90, 80.0},
    };

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
