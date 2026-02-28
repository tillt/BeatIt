//
//  moving_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Moving";
    cfg.audio_filename = "moving.wav";
    cfg.dump_env_var = "BEATIT_MOVING_DUMP_EVENTS";

    cfg.event_probe_count = 16;

    cfg.expected_first_downbeat_sample_frame = 1104ULL;
    cfg.first_downbeat_sample_tolerance_ms = 10.0;

    cfg.min_expected_bpm = 124.8;
    cfg.max_expected_bpm = 125.2;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
