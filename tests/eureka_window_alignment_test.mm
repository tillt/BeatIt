//
//  eureka_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "window_alignment_case.h"

int main() {
    beatit::tests::window_alignment::WindowAlignmentCaseConfig cfg;
    cfg.name = "Eureka";
    cfg.audio_filename = "eureka.wav";
    cfg.dump_env_var = "BEATIT_EUREKA_DUMP_EVENTS";

    cfg.edge_window_beats = 64;
    cfg.alternation_window_beats = 24;
    cfg.tempo_edge_intervals = 64;
    cfg.drift_probe_count = 24;
    cfg.event_probe_count = 16;

    cfg.expected_first_downbeat_sample_frame = 241ULL;
    cfg.first_downbeat_sample_tolerance_ms = 10.0;

    cfg.min_expected_bpm = 119.95;
    cfg.max_expected_bpm = 120.30;

    cfg.max_intro_median_abs_ms = 52.0;
    cfg.max_offset_slope_ms_per_beat = 0.04;
    cfg.max_unwrapped_slope_ms_per_beat = 0.06;
    cfg.max_start_end_delta_ms = 65.0;
    cfg.max_start_end_delta_beats = 0.12;
    cfg.max_unwrapped_start_end_delta_beats = 0.13;
    cfg.max_odd_even_median_gap_ms = 11.0;
    cfg.max_tempo_edge_bpm_delta = 0.0005;

    cfg.use_interior_windows = true;
    cfg.fail_one_beat_linear_signature = true;
    cfg.fail_wrapped_middle_signature = true;
    cfg.one_beat_signature_min_beats = 0.70;
    cfg.one_beat_signature_max_beats = 1.30;

    return beatit::tests::window_alignment::run_window_alignment_case(cfg);
}
