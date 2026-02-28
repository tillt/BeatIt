//
//  lethal_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Lethal";
    cfg.audio_filename = "lethal.wav";
    cfg.dump_env_var = "BEATIT_LETHAL_DUMP_EVENTS";

    cfg.edge_window_beats = 64;
    cfg.alternation_window_beats = 24;
    cfg.tempo_edge_intervals = 64;
    cfg.event_probe_count = 16;

    cfg.min_beat_count = 64;
    cfg.expected_beat_count = 781;
    cfg.expected_downbeat_count = 30;

    cfg.expected_first_downbeat_feature_frame = 32ULL;
    cfg.first_downbeat_feature_frame_tolerance = 1ULL;
    cfg.expected_first_downbeat_sample_frame = 14390ULL;
    cfg.first_downbeat_sample_tolerance_ms = 10.0;

    cfg.target_bpm = 122.009;
    cfg.max_bpm_error = 0.02;

    cfg.max_intro_median_abs_ms = 15.0;
    cfg.max_offset_slope_ms_per_beat = 0.042;
    cfg.max_start_end_delta_ms = 26.0;
    cfg.max_start_end_delta_beats = 0.052;
    cfg.max_odd_even_median_gap_ms = 0.2;
    cfg.max_tempo_edge_bpm_delta = 0.0005;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
