//
//  bazaar_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Bazaar";
    cfg.audio_filename = "bazaar.wav";
    cfg.dump_env_var = "BEATIT_BAZAAR_DUMP_EVENTS";

    cfg.edge_window_beats = 64;
    cfg.alternation_window_beats = 24;
    cfg.tempo_edge_intervals = 64;
    cfg.drift_probe_count = 24;
    cfg.event_probe_count = 16;

    cfg.expected_first_downbeat_sample_frame = 11994ULL;
    cfg.first_downbeat_sample_tolerance_ms = 10.0;

    cfg.min_expected_bpm = 122.9;
    cfg.max_expected_bpm = 123.1;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
