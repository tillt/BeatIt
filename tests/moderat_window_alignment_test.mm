//
//  moderat_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Moderat";
    cfg.audio_filename = "moderat.wav";
    cfg.dump_env_var = "BEATIT_MODERAT_DUMP_EVENTS";

    cfg.edge_window_beats = 64;
    cfg.alternation_window_beats = 24;
    cfg.tempo_edge_intervals = 64;
    cfg.event_probe_count = 16;

    cfg.expected_first_downbeat_feature_frame = 0ULL;
    cfg.first_downbeat_feature_frame_tolerance = 1ULL;
    cfg.expected_first_downbeat_sample_frame = 3750ULL;
    cfg.first_downbeat_sample_tolerance_ms = 10.0;

    cfg.target_bpm = 124.001;
    cfg.max_bpm_error = 0.02;

    cfg.max_intro_median_abs_ms = 4.0;
    cfg.max_offset_slope_ms_per_beat = 0.005;
    cfg.max_start_end_delta_ms = 2.0;
    cfg.max_start_end_delta_beats = 0.004;
    cfg.max_odd_even_median_gap_ms = 4.5;
    cfg.max_tempo_edge_bpm_delta = 0.0005;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
