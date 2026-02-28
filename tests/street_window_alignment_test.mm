//
//  street_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Street";
    cfg.audio_filename = "street.wav";
    cfg.dump_env_var = "BEATIT_STREET_DUMP_EVENTS";

    cfg.edge_window_beats = 64;
    cfg.alternation_window_beats = 24;
    cfg.tempo_edge_intervals = 64;
    cfg.event_probe_count = 16;

    cfg.min_expected_bpm = 121.8;
    cfg.max_expected_bpm = 122.2;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
