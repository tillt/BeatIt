//
//  turbo_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Turbo";
    cfg.audio_filename = "turbo.wav";
    cfg.dump_env_var = "BEATIT_TURBO_DUMP_EVENTS";

    cfg.edge_window_beats = 16;
    cfg.edge_window_bars = 8;
    cfg.alternation_window_beats = 16;
    cfg.tempo_edge_intervals = 32;
    cfg.event_probe_count = 16;

    cfg.min_expected_bpm = 126.8;
    cfg.max_expected_bpm = 127.2;

    cfg.max_intro_median_abs_ms = 10.0;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
