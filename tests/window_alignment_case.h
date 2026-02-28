//
//  window_alignment_case.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "coreml_test_config.h"
#include "window_alignment_test_utils.h"

namespace beatit::tests::window_alignment {

struct WindowAlignmentCaseConfig {
    struct LocalOffsetWindowCheck {
        const char* label = "";
        double center_fraction = 0.0;
        double max_median_abs_ms = 0.0;
    };

    struct LocalBarOffsetWindowCheck {
        const char* label = "";
        double center_fraction = 0.0;
        double max_median_abs_ms = 0.0;
    };

    const char* name = "Window";
    const char* audio_filename = nullptr;
    const char* dump_env_var = nullptr;

    std::size_t edge_window_beats = 64;
    std::size_t edge_window_bars = 16;
    std::size_t alternation_window_beats = 24;
    std::size_t tempo_edge_intervals = 64;
    std::size_t drift_probe_count = 24;
    std::size_t event_probe_count = 16;

    std::size_t min_beat_count = 0;
    std::optional<std::size_t> expected_beat_count;

    std::optional<std::size_t> expected_downbeat_count;
    std::optional<std::size_t> min_downbeat_count;

    std::optional<unsigned long long> expected_first_downbeat_feature_frame;
    unsigned long long first_downbeat_feature_frame_tolerance = 0;
    std::optional<unsigned long long> expected_first_downbeat_sample_frame;
    double first_downbeat_sample_tolerance_ms = 10.0;

    std::optional<double> target_bpm;
    double max_bpm_error = 0.0;
    std::optional<double> min_expected_bpm;
    std::optional<double> max_expected_bpm;

    bool require_first_bar_complete = true;
    bool require_bars_repeat_every_four = true;

    std::optional<double> max_intro_median_abs_ms;
    std::optional<double> max_offset_slope_ms_per_beat;
    std::optional<double> max_start_end_delta_ms;
    std::optional<double> max_start_end_delta_beats;
    std::optional<double> max_odd_even_median_gap_ms;
    std::optional<double> max_tempo_edge_bpm_delta;

    bool use_interior_windows = false;
    std::optional<double> max_unwrapped_slope_ms_per_beat;
    std::optional<double> max_unwrapped_start_end_delta_beats;

    bool fail_one_beat_linear_signature = false;
    double one_beat_signature_min_beats = 0.70;
    double one_beat_signature_max_beats = 1.30;

    bool fail_wrapped_middle_signature = false;

    bool check_seed_order = false;
    double max_seed_order_bpm_delta = 0.01;
    double max_seed_order_grid_median_delta_frames = 1.0;

    std::vector<LocalOffsetWindowCheck> local_offset_windows;
    std::vector<LocalBarOffsetWindowCheck> local_bar_offset_windows;
};

inline int fail_case(const WindowAlignmentCaseConfig& cfg, const std::string& message) {
    std::cerr << cfg.name << " alignment test failed: " << message << "\n";
    return 1;
}

inline bool env_enabled(const char* name) {
    if (!name) {
        return false;
    }
    const char* value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
}

inline int run_window_alignment_case(const WindowAlignmentCaseConfig& cfg) {
    std::string source_model_path = beatit::tests::resolve_beatthis_coreml_model_path();
    if (source_model_path.empty()) {
        std::cerr << "SKIP: BeatThis CoreML model missing (set BEATIT_COREML_MODEL_PATH).\n";
        return 77;
    }

    std::string model_error;
    std::string model_path = compile_model_if_needed(source_model_path, &model_error);
    if (model_path.empty()) {
        std::cerr << "SKIP: Failed to prepare CoreML model: " << model_error << "\n";
        return 77;
    }

    std::filesystem::path test_root = std::filesystem::current_path();
#if defined(BEATIT_TEST_DATA_DIR)
    test_root = BEATIT_TEST_DATA_DIR;
#endif
    const std::filesystem::path audio_path = test_root / "training" / cfg.audio_filename;
    if (!std::filesystem::exists(audio_path)) {
        std::cerr << "SKIP: missing " << audio_path.string() << "\n";
        return 77;
    }

    beatit::BeatitConfig config;
    beatit::tests::apply_beatthis_coreml_test_config(config);
    config.model_path = model_path;
    config.use_dbn = true;
    config.max_analysis_seconds = 60.0;
    config.dbn_window_start_seconds = 0.0;
#if defined(BEATIT_TEST_SPARSE_PROBE_MODE)
    config.sparse_probe_mode = true;
#endif
    if (env_enabled("BEATIT_WINDOW_TRACE")) {
        config.log_verbosity = beatit::LogVerbosity::Debug;
        config.dbn_trace = true;
        config.profile = true;
    }
    if (env_enabled("BEATIT_TEST_CPU_ONLY")) {
        config.compute_units = beatit::BeatitConfig::ComputeUnits::CPUOnly;
    }

    std::vector<float> mono;
    double sample_rate = 0.0;
    std::string decode_error;
    if (!decode_audio_mono(audio_path.string(), &mono, &sample_rate, &decode_error)) {
        return fail_case(cfg, "decode error: " + decode_error);
    }
    if (sample_rate <= 0.0 || mono.empty()) {
        return fail_case(cfg, "decoded audio is empty.");
    }

    const double total_duration_s = static_cast<double>(mono.size()) / sample_rate;
    auto provider =
        [&](double start_seconds, double duration_seconds, std::vector<float>* out_samples)
            -> std::size_t {
        if (!out_samples) {
            return 0;
        }
        out_samples->clear();
        if (sample_rate <= 0.0 || mono.empty()) {
            return 0;
        }
        const std::size_t begin = static_cast<std::size_t>(
            std::llround(std::max(0.0, start_seconds) * sample_rate));
        const std::size_t count = static_cast<std::size_t>(
            std::llround(std::max(0.0, duration_seconds) * sample_rate));
        const std::size_t end = std::min(mono.size(), begin + count);
        if (begin >= end) {
            return 0;
        }
        out_samples->assign(mono.begin() + static_cast<long>(begin),
                            mono.begin() + static_cast<long>(end));
        return out_samples->size();
    };

    auto analyze_for_seed_order = [&](const char* order) -> std::optional<beatit::AnalysisResult> {
        if (order && order[0] != '\0') {
            setenv("BEATIT_SPARSE_SEED_ORDER", order, 1);
        } else {
            unsetenv("BEATIT_SPARSE_SEED_ORDER");
        }
        beatit::BeatitStream stream(sample_rate, config, true);
        double start_s = 0.0;
        double duration_s = 0.0;
        if (!stream.request_analysis_window(&start_s, &duration_s)) {
            return std::nullopt;
        }
        return stream.analyze_window(start_s, duration_s, total_duration_s, provider);
    };

    std::optional<beatit::AnalysisResult> baseline_result_opt = analyze_for_seed_order(nullptr);
    unsetenv("BEATIT_SPARSE_SEED_ORDER");
    if (!baseline_result_opt.has_value()) {
        return fail_case(cfg, "request_analysis_window returned false.");
    }
    beatit::AnalysisResult result = *baseline_result_opt;

    if (cfg.check_seed_order) {
        std::optional<beatit::AnalysisResult> right_first_result_opt =
            analyze_for_seed_order("right_first");
        unsetenv("BEATIT_SPARSE_SEED_ORDER");
        if (!right_first_result_opt.has_value()) {
            return fail_case(cfg, "request_analysis_window failed for seed-order check.");
        }

        const beatit::AnalysisResult& right_first_result = *right_first_result_opt;
        const auto& baseline_grid = result.coreml_beat_projected_sample_frames.empty()
            ? result.coreml_beat_sample_frames
            : result.coreml_beat_projected_sample_frames;
        const auto& right_first_grid = right_first_result.coreml_beat_projected_sample_frames.empty()
            ? right_first_result.coreml_beat_sample_frames
            : right_first_result.coreml_beat_projected_sample_frames;
        if (baseline_grid.size() != right_first_grid.size()) {
            return fail_case(cfg,
                             "probe seed order changed beat count: baseline=" +
                                 std::to_string(baseline_grid.size()) +
                                 " right_first=" + std::to_string(right_first_grid.size()));
        }
        const double order_bpm_delta =
            std::fabs(result.estimated_bpm - right_first_result.estimated_bpm);
        const double order_grid_median_delta_frames =
            median_abs_frame_delta(baseline_grid, right_first_grid);
        if (order_bpm_delta > cfg.max_seed_order_bpm_delta) {
            return fail_case(cfg,
                             "probe seed order changed BPM by " +
                                 std::to_string(order_bpm_delta));
        }
        if (order_grid_median_delta_frames > cfg.max_seed_order_grid_median_delta_frames) {
            return fail_case(cfg,
                             "probe seed order changed grid by median " +
                                 std::to_string(order_grid_median_delta_frames) + " frames.");
        }
    }

    if (result.coreml_beat_events.size() < cfg.min_beat_count) {
        return fail_case(cfg,
                         "too few beat events: " +
                             std::to_string(result.coreml_beat_events.size()));
    }
    if (cfg.expected_beat_count.has_value() &&
        result.coreml_beat_events.size() != *cfg.expected_beat_count) {
        return fail_case(cfg,
                         "beat event count " +
                             std::to_string(result.coreml_beat_events.size()) +
                             " != " + std::to_string(*cfg.expected_beat_count) + ".");
    }

    if (cfg.require_first_bar_complete && !first_bar_is_complete_four_four(result)) {
        return fail_case(cfg,
                         "opening bar is not complete 4/4 (expected first downbeats at beat "
                         "indices 0 and 4).");
    }
    if (cfg.require_bars_repeat_every_four && !bars_repeat_every_four_beats(result)) {
        return fail_case(cfg, "bar markers are not consistently every 4 beats.");
    }

    if (!(result.estimated_bpm > 0.0)) {
        return fail_case(cfg, "non-positive BPM.");
    }
    if (cfg.target_bpm.has_value()) {
        if (std::fabs(result.estimated_bpm - *cfg.target_bpm) > cfg.max_bpm_error) {
            return fail_case(cfg,
                             "estimated BPM " + std::to_string(result.estimated_bpm) +
                                 " outside [" + std::to_string(*cfg.target_bpm - cfg.max_bpm_error) +
                                 "," + std::to_string(*cfg.target_bpm + cfg.max_bpm_error) + "].");
        }
    } else {
        if (cfg.min_expected_bpm.has_value() && result.estimated_bpm < *cfg.min_expected_bpm) {
            return fail_case(cfg,
                             "estimated BPM " + std::to_string(result.estimated_bpm) +
                                 " < " + std::to_string(*cfg.min_expected_bpm) + ".");
        }
        if (cfg.max_expected_bpm.has_value() && result.estimated_bpm > *cfg.max_expected_bpm) {
            return fail_case(cfg,
                             "estimated BPM " + std::to_string(result.estimated_bpm) +
                                 " > " + std::to_string(*cfg.max_expected_bpm) + ".");
        }
    }

    if (cfg.expected_downbeat_count.has_value() &&
        result.coreml_downbeat_feature_frames.size() != *cfg.expected_downbeat_count) {
        return fail_case(cfg,
                         "downbeat count " +
                             std::to_string(result.coreml_downbeat_feature_frames.size()) +
                             " != " + std::to_string(*cfg.expected_downbeat_count) + ".");
    }
    if (cfg.min_downbeat_count.has_value() &&
        result.coreml_downbeat_feature_frames.size() < *cfg.min_downbeat_count) {
        return fail_case(cfg,
                         "downbeat count " +
                             std::to_string(result.coreml_downbeat_feature_frames.size()) +
                             " < " + std::to_string(*cfg.min_downbeat_count) + ".");
    }

    unsigned long long first_downbeat_feature_ff = 0;
    unsigned long long first_downbeat_sample_sf = 0;
    if (cfg.expected_first_downbeat_feature_frame.has_value()) {
        if (!first_downbeat_feature_frame(result, &first_downbeat_feature_ff)) {
            return fail_case(cfg, "missing first downbeat feature frame.");
        }
        const auto downbeat_feature_delta =
            (first_downbeat_feature_ff > *cfg.expected_first_downbeat_feature_frame)
                ? (first_downbeat_feature_ff - *cfg.expected_first_downbeat_feature_frame)
                : (*cfg.expected_first_downbeat_feature_frame - first_downbeat_feature_ff);
        if (downbeat_feature_delta > cfg.first_downbeat_feature_frame_tolerance) {
            const auto lower_feature_frame =
                (*cfg.expected_first_downbeat_feature_frame > cfg.first_downbeat_feature_frame_tolerance)
                    ? (*cfg.expected_first_downbeat_feature_frame -
                       cfg.first_downbeat_feature_frame_tolerance)
                    : 0ULL;
            return fail_case(cfg,
                             "first downbeat feature frame " +
                                 std::to_string(first_downbeat_feature_ff) + " outside [" +
                                 std::to_string(lower_feature_frame) + "," +
                                 std::to_string(*cfg.expected_first_downbeat_feature_frame +
                                                cfg.first_downbeat_feature_frame_tolerance) +
                                 "].");
        }
    }

    if (cfg.expected_first_downbeat_sample_frame.has_value()) {
        if (!first_downbeat_sample_frame(result, &first_downbeat_sample_sf)) {
            return fail_case(cfg, "missing first downbeat sample frame.");
        }
        const auto downbeat_sample_tolerance_frames = static_cast<unsigned long long>(
            std::llround((cfg.first_downbeat_sample_tolerance_ms / 1000.0) * sample_rate));
        const auto downbeat_sample_delta =
            (first_downbeat_sample_sf > *cfg.expected_first_downbeat_sample_frame)
                ? (first_downbeat_sample_sf - *cfg.expected_first_downbeat_sample_frame)
                : (*cfg.expected_first_downbeat_sample_frame - first_downbeat_sample_sf);
        if (downbeat_sample_delta > downbeat_sample_tolerance_frames) {
            const auto lower_sample_frame =
                (*cfg.expected_first_downbeat_sample_frame > downbeat_sample_tolerance_frames)
                    ? (*cfg.expected_first_downbeat_sample_frame - downbeat_sample_tolerance_frames)
                    : 0ULL;
            return fail_case(cfg,
                             "first downbeat sample frame " +
                                 std::to_string(first_downbeat_sample_sf) + " outside [" +
                                 std::to_string(lower_sample_frame) + "," +
                                 std::to_string(*cfg.expected_first_downbeat_sample_frame +
                                                downbeat_sample_tolerance_frames) +
                                 "] (±" + std::to_string(cfg.first_downbeat_sample_tolerance_ms) +
                                 "ms).");
        }
    }

    if (cfg.expected_beat_count.has_value() &&
        !result.coreml_beat_projected_sample_frames.empty() &&
        result.coreml_beat_projected_sample_frames.size() != *cfg.expected_beat_count) {
        return fail_case(cfg,
                         "projected beat count " +
                             std::to_string(result.coreml_beat_projected_sample_frames.size()) +
                             " != " + std::to_string(*cfg.expected_beat_count) + ".");
    }

    std::vector<unsigned long long> beat_frames;
    std::vector<unsigned long long> beat_styles;
    beat_frames.reserve(result.coreml_beat_events.size());
    beat_styles.reserve(result.coreml_beat_events.size());
    for (const auto& event : result.coreml_beat_events) {
        beat_frames.push_back(event.frame);
        beat_styles.push_back(event.style);
    }

    const std::vector<double> offsets_ms =
        compute_strong_peak_offsets_ms(beat_frames, mono, sample_rate, result.estimated_bpm);
    if (offsets_ms.size() < cfg.edge_window_beats) {
        return fail_case(cfg, "too few offsets.");
    }

    std::vector<double> first(offsets_ms.begin(),
                              offsets_ms.begin() + static_cast<long>(cfg.edge_window_beats));
    std::vector<double> last(offsets_ms.end() - static_cast<long>(cfg.edge_window_beats),
                             offsets_ms.end());
    const double start_median_ms = median(first);
    std::vector<double> first_abs = first;
    for (double& v : first_abs) {
        v = std::fabs(v);
    }
    const double start_median_abs_ms = median(first_abs);
    const double end_median_ms = median(last);
    const double start_end_delta_ms = end_median_ms - start_median_ms;
    const double beat_period_ms =
        (result.estimated_bpm > 0.0f) ? (60000.0 / result.estimated_bpm) : 500.0;
    const double ms_per_beat =
        result.estimated_bpm > 0.0 ? (60000.0 / result.estimated_bpm) : 0.0;
    const double start_end_delta_beats =
        ms_per_beat > 0.0 ? (std::fabs(start_end_delta_ms) / ms_per_beat) : 0.0;
    const double slope_ms_per_beat = robust_linear_slope(offsets_ms, beat_period_ms);

    const std::size_t alt_n = std::min<std::size_t>(cfg.alternation_window_beats, offsets_ms.size());
    std::vector<double> odd;
    std::vector<double> even;
    odd.reserve(alt_n / 2);
    even.reserve((alt_n + 1) / 2);
    for (std::size_t i = 0; i < alt_n; ++i) {
        if ((i % 2) == 0) {
            even.push_back(offsets_ms[i]);
        } else {
            odd.push_back(offsets_ms[i]);
        }
    }
    const double odd_even_gap_ms = std::fabs(median(even) - median(odd));
    const double early_interval_s =
        median_interval_seconds(beat_frames, sample_rate, cfg.tempo_edge_intervals, false);
    const double late_interval_s =
        median_interval_seconds(beat_frames, sample_rate, cfg.tempo_edge_intervals, true);
    const double early_bpm = early_interval_s > 0.0 ? (60.0 / early_interval_s) : 0.0;
    const double late_bpm = late_interval_s > 0.0 ? (60.0 / late_interval_s) : 0.0;
    const double edge_bpm_delta = std::fabs(early_bpm - late_bpm);

    std::optional<double> between_median_ms;
    std::optional<double> between_median_abs_ms;
    std::optional<double> middle_median_ms;
    std::optional<double> middle_median_abs_ms;
    if (cfg.use_interior_windows || cfg.fail_wrapped_middle_signature) {
        const std::size_t middle_start = (offsets_ms.size() - cfg.edge_window_beats) / 2;
        const std::size_t between_start = middle_start / 2;
        std::vector<double> between(offsets_ms.begin() + static_cast<long>(between_start),
                                    offsets_ms.begin() + static_cast<long>(between_start + cfg.edge_window_beats));
        std::vector<double> middle(offsets_ms.begin() + static_cast<long>(middle_start),
                                   offsets_ms.begin() + static_cast<long>(middle_start + cfg.edge_window_beats));

        between_median_ms = median(between);
        middle_median_ms = median(middle);

        std::vector<double> between_abs = between;
        std::vector<double> middle_abs = middle;
        for (double& v : between_abs) {
            v = std::fabs(v);
        }
        for (double& v : middle_abs) {
            v = std::fabs(v);
        }
        between_median_abs_ms = median(between_abs);
        middle_median_abs_ms = median(middle_abs);
    }

    std::optional<double> unwrapped_slope_ms_per_beat;
    std::optional<double> unwrapped_start_end_delta_beats;
    bool one_beat_linear_signature = false;
    bool wrapped_middle_signature = false;
    if (cfg.max_unwrapped_slope_ms_per_beat.has_value() ||
        cfg.max_unwrapped_start_end_delta_beats.has_value() ||
        cfg.fail_one_beat_linear_signature ||
        cfg.fail_wrapped_middle_signature) {
        const std::vector<double> unwrapped_offsets_ms =
            unwrap_periodic_offsets_ms(offsets_ms, beat_period_ms);
        unwrapped_slope_ms_per_beat =
            robust_linear_slope(unwrapped_offsets_ms, beat_period_ms);

        std::vector<double> first_unwrapped(unwrapped_offsets_ms.begin(),
                                            unwrapped_offsets_ms.begin() + static_cast<long>(cfg.edge_window_beats));
        std::vector<double> last_unwrapped(unwrapped_offsets_ms.end() - static_cast<long>(cfg.edge_window_beats),
                                           unwrapped_offsets_ms.end());
        const double unwrapped_start_median_ms = median(first_unwrapped);
        const double unwrapped_end_median_ms = median(last_unwrapped);
        const double unwrapped_start_end_delta_ms =
            unwrapped_end_median_ms - unwrapped_start_median_ms;
        unwrapped_start_end_delta_beats =
            ms_per_beat > 0.0 ? (std::fabs(unwrapped_start_end_delta_ms) / ms_per_beat) : 0.0;

        one_beat_linear_signature =
            unwrapped_start_end_delta_beats.value_or(0.0) > cfg.one_beat_signature_min_beats &&
            unwrapped_start_end_delta_beats.value_or(0.0) < cfg.one_beat_signature_max_beats;

        const double phase_hot_threshold_ms = std::max(35.0, beat_period_ms * 0.22);
        const bool edges_calm =
            std::fabs(start_median_ms) <= (phase_hot_threshold_ms * 0.5) &&
            std::fabs(end_median_ms) <= (phase_hot_threshold_ms * 0.5);
        const bool between_hot =
            between_median_abs_ms.has_value() &&
            *between_median_abs_ms > phase_hot_threshold_ms;
        const bool middle_hot =
            middle_median_abs_ms.has_value() &&
            *middle_median_abs_ms > phase_hot_threshold_ms;
        wrapped_middle_signature = edges_calm && (between_hot || middle_hot);
    }

    for (const auto& local_window : cfg.local_offset_windows) {
        if (offsets_ms.size() < cfg.edge_window_beats) {
            return fail_case(cfg, "too few offsets for local window checks.");
        }
        const std::size_t max_start =
            offsets_ms.size() > cfg.edge_window_beats ? (offsets_ms.size() - cfg.edge_window_beats) : 0;
        const double center_fraction = std::clamp(local_window.center_fraction, 0.0, 1.0);
        const std::size_t center_index = static_cast<std::size_t>(
            std::llround(center_fraction * static_cast<double>(offsets_ms.size() - 1)));
        const std::size_t half_window = cfg.edge_window_beats / 2;
        const std::size_t start_index =
            std::min(max_start,
                     center_index > half_window ? (center_index - half_window) : std::size_t{0});
        std::vector<double> window(offsets_ms.begin() + static_cast<long>(start_index),
                                   offsets_ms.begin() + static_cast<long>(start_index + cfg.edge_window_beats));
        for (double& v : window) {
            v = std::fabs(v);
        }
        const double local_median_abs_ms = median(window);
        if (local_median_abs_ms > local_window.max_median_abs_ms) {
            return fail_case(cfg,
                             std::string("local window '") + local_window.label +
                                 "' median abs offset " + std::to_string(local_median_abs_ms) +
                                 "ms > " + std::to_string(local_window.max_median_abs_ms) + "ms.");
        }
    }

    if (!cfg.local_bar_offset_windows.empty()) {
        std::vector<unsigned long long> bar_frames;
        bar_frames.reserve(result.coreml_beat_events.size() / 4);
        for (const auto& event : result.coreml_beat_events) {
            if (is_bar_event(event)) {
                bar_frames.push_back(event.frame);
            }
        }

        const std::vector<double> bar_offsets_ms =
            compute_strong_peak_offsets_ms(bar_frames, mono, sample_rate, result.estimated_bpm);
        if (bar_offsets_ms.size() < cfg.edge_window_bars) {
            return fail_case(cfg, "too few bar offsets for local bar window checks.");
        }

        for (const auto& local_window : cfg.local_bar_offset_windows) {
            const std::size_t max_start =
                bar_offsets_ms.size() > cfg.edge_window_bars
                    ? (bar_offsets_ms.size() - cfg.edge_window_bars)
                    : 0;
            const double center_fraction = std::clamp(local_window.center_fraction, 0.0, 1.0);
            const std::size_t center_index = static_cast<std::size_t>(
                std::llround(center_fraction * static_cast<double>(bar_offsets_ms.size() - 1)));
            const std::size_t half_window = cfg.edge_window_bars / 2;
            const std::size_t start_index =
                std::min(max_start,
                         center_index > half_window ? (center_index - half_window) : std::size_t{0});
            std::vector<double> window(
                bar_offsets_ms.begin() + static_cast<long>(start_index),
                bar_offsets_ms.begin() + static_cast<long>(start_index + cfg.edge_window_bars));
            for (double& v : window) {
                v = std::fabs(v);
            }
            const double local_median_abs_ms = median(window);
            if (local_median_abs_ms > local_window.max_median_abs_ms) {
                return fail_case(cfg,
                                 std::string("local bar window '") + local_window.label +
                                     "' median abs offset " +
                                     std::to_string(local_median_abs_ms) + "ms > " +
                                     std::to_string(local_window.max_median_abs_ms) + "ms.");
            }
        }
    }

    if (env_enabled(cfg.dump_env_var)) {
        std::cout << cfg.name << " event probe: first_frames="
                  << format_slice(beat_frames, cfg.event_probe_count, false)
                  << " first_styles="
                  << format_slice(beat_styles, cfg.event_probe_count, false)
                  << " last_frames="
                  << format_slice(beat_frames, cfg.event_probe_count, true)
                  << " last_styles="
                  << format_slice(beat_styles, cfg.event_probe_count, true)
                  << "\n";
        std::cout << cfg.name << " downbeat probe: first="
                  << format_slice(result.coreml_downbeat_feature_frames, cfg.event_probe_count, false)
                  << " last="
                  << format_slice(result.coreml_downbeat_feature_frames, cfg.event_probe_count, true)
                  << "\n";
        if (cfg.expected_first_downbeat_feature_frame.has_value() ||
            cfg.expected_first_downbeat_sample_frame.has_value()) {
            const bool has_feature = first_downbeat_feature_frame(result, &first_downbeat_feature_ff);
            const bool has_sample = first_downbeat_sample_frame(result, &first_downbeat_sample_sf);
            std::cout << cfg.name << " downbeat timing probe: feature_frame="
                      << (has_feature ? std::to_string(first_downbeat_feature_ff) : std::string("none"))
                      << " sample_frame="
                      << (has_sample ? std::to_string(first_downbeat_sample_sf) : std::string("none"))
                      << "\n";
        }
    }

    std::cout << cfg.name << " alignment metrics: bpm=" << result.estimated_bpm
              << " beat_events=" << result.coreml_beat_events.size()
              << " downbeats=" << result.coreml_downbeat_feature_frames.size()
              << " projected_beats=" << result.coreml_beat_projected_sample_frames.size()
              << " start_median_ms=" << start_median_ms
              << " start_median_abs_ms=" << start_median_abs_ms
              << " end_median_ms=" << end_median_ms
              << " delta_ms=" << start_end_delta_ms
              << " delta_beats=" << start_end_delta_beats
              << " slope_ms_per_beat=" << slope_ms_per_beat
              << " odd_even_gap_ms=" << odd_even_gap_ms
              << " early_bpm=" << early_bpm
              << " late_bpm=" << late_bpm
              << " edge_bpm_delta=" << edge_bpm_delta;
    if (between_median_ms.has_value()) {
        std::cout << " between_median_ms=" << *between_median_ms
                  << " between_median_abs_ms=" << between_median_abs_ms.value_or(0.0);
    }
    if (middle_median_ms.has_value()) {
        std::cout << " middle_median_ms=" << *middle_median_ms
                  << " middle_median_abs_ms=" << middle_median_abs_ms.value_or(0.0);
    }
    if (unwrapped_slope_ms_per_beat.has_value()) {
        std::cout << " unwrapped_slope_ms_per_beat=" << *unwrapped_slope_ms_per_beat;
    }
    if (unwrapped_start_end_delta_beats.has_value()) {
        std::cout << " unwrapped_delta_beats=" << *unwrapped_start_end_delta_beats;
    }
    if (cfg.fail_one_beat_linear_signature) {
        std::cout << " one_beat_linear_signature=" << (one_beat_linear_signature ? 1 : 0);
    }
    if (cfg.fail_wrapped_middle_signature) {
        std::cout << " wrapped_middle_signature=" << (wrapped_middle_signature ? 1 : 0);
    }
    std::cout << "\n";

    if (cfg.max_intro_median_abs_ms.has_value() &&
        start_median_abs_ms > *cfg.max_intro_median_abs_ms) {
        return fail_case(cfg,
                         "intro median abs offset " + std::to_string(start_median_abs_ms) +
                             "ms > " + std::to_string(*cfg.max_intro_median_abs_ms) + "ms.");
    }
    if (cfg.max_offset_slope_ms_per_beat.has_value() &&
        std::fabs(slope_ms_per_beat) > *cfg.max_offset_slope_ms_per_beat) {
        return fail_case(cfg,
                         "slope " + std::to_string(slope_ms_per_beat) +
                             "ms/beat > " +
                             std::to_string(*cfg.max_offset_slope_ms_per_beat) + ".");
    }
    if (cfg.max_start_end_delta_ms.has_value() &&
        std::fabs(start_end_delta_ms) > *cfg.max_start_end_delta_ms) {
        return fail_case(cfg,
                         "start/end delta " + std::to_string(start_end_delta_ms) +
                             "ms > " + std::to_string(*cfg.max_start_end_delta_ms) + "ms.");
    }
    if (cfg.max_start_end_delta_beats.has_value() &&
        start_end_delta_beats > *cfg.max_start_end_delta_beats) {
        return fail_case(cfg,
                         "start/end delta " + std::to_string(start_end_delta_beats) +
                             " beats > " +
                             std::to_string(*cfg.max_start_end_delta_beats) + " beats.");
    }
    if (cfg.max_odd_even_median_gap_ms.has_value() &&
        odd_even_gap_ms > *cfg.max_odd_even_median_gap_ms) {
        return fail_case(cfg,
                         "odd/even median gap " + std::to_string(odd_even_gap_ms) +
                             "ms > " + std::to_string(*cfg.max_odd_even_median_gap_ms) + "ms.");
    }
    if (!(early_bpm > 0.0) || !(late_bpm > 0.0)) {
        return fail_case(cfg, "invalid edge BPM estimate.");
    }
    if (cfg.max_tempo_edge_bpm_delta.has_value() &&
        edge_bpm_delta > *cfg.max_tempo_edge_bpm_delta) {
        return fail_case(cfg,
                         "edge BPM delta " + std::to_string(edge_bpm_delta) +
                             " > " + std::to_string(*cfg.max_tempo_edge_bpm_delta) + ".");
    }
    if (cfg.max_unwrapped_slope_ms_per_beat.has_value()) {
        if (!unwrapped_slope_ms_per_beat.has_value()) {
            return fail_case(cfg, "missing unwrapped slope.");
        }
        if (std::fabs(*unwrapped_slope_ms_per_beat) > *cfg.max_unwrapped_slope_ms_per_beat) {
            return fail_case(cfg,
                             "unwrapped slope " + std::to_string(*unwrapped_slope_ms_per_beat) +
                                 "ms/beat > " +
                                 std::to_string(*cfg.max_unwrapped_slope_ms_per_beat) + ".");
        }
    }
    if (cfg.max_unwrapped_start_end_delta_beats.has_value()) {
        if (!unwrapped_start_end_delta_beats.has_value()) {
            return fail_case(cfg, "missing unwrapped start/end delta.");
        }
        if (*unwrapped_start_end_delta_beats > *cfg.max_unwrapped_start_end_delta_beats) {
            return fail_case(cfg,
                             "unwrapped start/end delta " +
                                 std::to_string(*unwrapped_start_end_delta_beats) +
                                 " beats > " +
                                 std::to_string(*cfg.max_unwrapped_start_end_delta_beats) +
                                 " beats.");
        }
    }
    if (cfg.fail_one_beat_linear_signature && one_beat_linear_signature) {
        return fail_case(cfg, "one-beat linear signature detected.");
    }
    if (cfg.fail_wrapped_middle_signature && wrapped_middle_signature) {
        return fail_case(cfg, "wrapped middle signature detected.");
    }

    return 0;
}

} // namespace beatit::tests::window_alignment
