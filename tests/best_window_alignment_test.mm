//
//  best_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Best";
    cfg.audio_filename = "best.wav";
    cfg.dump_env_var = "BEATIT_BEST_DUMP_EVENTS";

    cfg.edge_window_beats = 64;
    cfg.alternation_window_beats = 24;
    cfg.tempo_edge_intervals = 64;
    cfg.drift_probe_count = 24;
    cfg.event_probe_count = 16;

    cfg.min_beat_count = 64;
    cfg.expected_beat_count = 1163;
    cfg.expected_downbeat_count = 30;

    cfg.expected_first_downbeat_feature_frame = 26ULL;
    cfg.first_downbeat_feature_frame_tolerance = 1ULL;
    cfg.expected_first_downbeat_sample_frame = 3711ULL;
    cfg.first_downbeat_sample_tolerance_ms = 10.0;

    cfg.min_expected_bpm = 121.0;
    cfg.max_expected_bpm = 123.0;

    cfg.max_intro_median_abs_ms = 17.0;
    cfg.max_offset_slope_ms_per_beat = 0.024;
    cfg.max_start_end_delta_ms = 20.0;
    cfg.max_start_end_delta_beats = 0.04;
    cfg.max_odd_even_median_gap_ms = 4.5;
    cfg.max_tempo_edge_bpm_delta = 0.0005;

    cfg.use_interior_windows = true;
    cfg.fail_wrapped_middle_signature = true;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
