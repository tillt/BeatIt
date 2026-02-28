//
//  sparse_usability_scan_test.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "sparse/usability_scan.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool check_close(double actual, double expected, double tolerance, const char* label) {
    if (std::fabs(actual - expected) <= tolerance) {
        return true;
    }
    std::cerr << "Sparse usability scan test failed: " << label
              << " expected " << expected << " got " << actual << ".\n";
    return false;
}

bool test_scoring_rewards_rhythmic_windows() {
    const beatit::detail::SparseUsabilityFeatures strong{
        0.05, 0.80, 0.90, 0.85, 0.10
    };
    const beatit::detail::SparseUsabilityFeatures weak{
        0.70, 0.10, 0.15, 0.10, 0.80
    };

    const double strong_score = beatit::detail::score_sparse_usability_window(strong);
    const double weak_score = beatit::detail::score_sparse_usability_window(weak);

    if (!(strong_score > weak_score)) {
        std::cerr << "Sparse usability scan test failed: strong window should score above weak window.\n";
        return false;
    }
    if (!beatit::detail::sparse_window_is_usable(strong)) {
        std::cerr << "Sparse usability scan test failed: strong window should be usable.\n";
        return false;
    }
    if (beatit::detail::sparse_window_is_usable(weak)) {
        std::cerr << "Sparse usability scan test failed: weak window should not be usable.\n";
        return false;
    }
    return true;
}

bool test_picker_prefers_nearby_when_scores_are_similar() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        20.0, 30.0, {0.08, 0.72, 0.84, 0.78, 0.14}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        82.0, 30.0, {0.06, 0.75, 0.86, 0.79, 0.13}));

    const beatit::detail::SparseUsabilityPickRequest request{
        18.0, 120.0, 0.01, 0.0
    };
    const std::size_t picked =
        beatit::detail::pick_sparse_usability_window(windows, request);
    if (picked != 0) {
        std::cerr << "Sparse usability scan test failed: expected nearby window to win.\n";
        return false;
    }
    return true;
}

bool test_picker_prefers_much_better_window_within_snap_budget() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        18.0, 30.0, {0.28, 0.35, 0.42, 0.33, 0.30}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        62.0, 30.0, {0.05, 0.82, 0.95, 0.90, 0.08}));

    const beatit::detail::SparseUsabilityPickRequest request{
        18.0, 80.0, 0.003, 0.0
    };
    const std::size_t picked =
        beatit::detail::pick_sparse_usability_window(windows, request);
    if (picked != 1) {
        std::cerr << "Sparse usability scan test failed: expected much better window to win.\n";
        return false;
    }
    return true;
}

bool test_picker_rejects_windows_beyond_snap_budget() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        90.0, 30.0, {0.05, 0.80, 0.90, 0.88, 0.10}));

    const beatit::detail::SparseUsabilityPickRequest request{
        10.0, 20.0, 0.01, 0.0
    };
    const std::size_t picked =
        beatit::detail::pick_sparse_usability_window(windows, request);
    if (picked != windows.size()) {
        std::cerr << "Sparse usability scan test failed: expected no window within snap budget.\n";
        return false;
    }
    return true;
}

bool test_window_builder_keeps_score_consistent() {
    const beatit::detail::SparseUsabilityFeatures features{
        0.10, 0.60, 0.75, 0.65, 0.20
    };
    const auto window =
        beatit::detail::build_sparse_usability_window(12.0, 30.0, features);
    return check_close(window.score,
                       beatit::detail::score_sparse_usability_window(features),
                       1e-9,
                       "window score");
}

bool test_span_builder_groups_contiguous_usable_windows() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        0.0, 30.0, {0.08, 0.72, 0.84, 0.78, 0.14}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        30.0, 30.0, {0.10, 0.70, 0.80, 0.76, 0.18}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        60.0, 30.0, {0.70, 0.10, 0.15, 0.10, 0.80}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        90.0, 30.0, {0.06, 0.78, 0.88, 0.82, 0.12}));

    const auto spans = beatit::detail::build_sparse_usability_spans(windows, 0.0);
    if (spans.size() != 2) {
        std::cerr << "Sparse usability scan test failed: expected 2 spans, got "
                  << spans.size() << ".\n";
        return false;
    }
    if (!check_close(spans[0].start_seconds, 0.0, 1e-9, "span 0 start")) {
        return false;
    }
    if (!check_close(spans[0].end_seconds, 60.0, 1e-9, "span 0 end")) {
        return false;
    }
    if (spans[0].first_index != 0 || spans[0].last_index != 1) {
        std::cerr << "Sparse usability scan test failed: first span index bounds mismatch.\n";
        return false;
    }
    if (!check_close(spans[1].start_seconds, 90.0, 1e-9, "span 1 start")) {
        return false;
    }
    if (!check_close(spans[1].end_seconds, 120.0, 1e-9, "span 1 end")) {
        return false;
    }
    return true;
}

std::vector<float> make_pulse_train(double sample_rate,
                                    double duration_seconds,
                                    double bpm) {
    const std::size_t total_samples =
        static_cast<std::size_t>(std::llround(sample_rate * duration_seconds));
    std::vector<float> samples(total_samples, 0.0f);
    const std::size_t interval =
        static_cast<std::size_t>(std::llround((60.0 / bpm) * sample_rate));
    const std::size_t width =
        static_cast<std::size_t>(std::llround(0.02 * sample_rate));
    for (std::size_t start = 0; start < total_samples; start += interval) {
        const std::size_t end = std::min(total_samples, start + width);
        for (std::size_t i = start; i < end; ++i) {
            samples[i] = 1.0f;
        }
    }
    return samples;
}

std::vector<float> make_sustained_pad(double sample_rate,
                                      double duration_seconds) {
    const std::size_t total_samples =
        static_cast<std::size_t>(std::llround(sample_rate * duration_seconds));
    return std::vector<float>(total_samples, 0.2f);
}

std::vector<float> concat_samples(const std::vector<std::vector<float>>& chunks) {
    std::size_t total = 0;
    for (const auto& chunk : chunks) {
        total += chunk.size();
    }
    std::vector<float> out;
    out.reserve(total);
    for (const auto& chunk : chunks) {
        out.insert(out.end(), chunk.begin(), chunk.end());
    }
    return out;
}

bool test_feature_measurement_distinguishes_pulses_from_pad() {
    constexpr double sample_rate = 1000.0;
    const auto pulses = make_pulse_train(sample_rate, 30.0, 120.0);
    const auto pad = make_sustained_pad(sample_rate, 30.0);

    const auto pulse_features =
        beatit::detail::measure_sparse_usability_features(pulses, sample_rate, 70.0, 180.0);
    const auto pad_features =
        beatit::detail::measure_sparse_usability_features(pad, sample_rate, 70.0, 180.0);

    if (!(pulse_features.periodicity > pad_features.periodicity)) {
        std::cerr << "Sparse usability scan test failed: pulse periodicity should exceed pad periodicity.\n";
        return false;
    }
    if (!(pulse_features.transient_strength > pad_features.transient_strength)) {
        std::cerr << "Sparse usability scan test failed: pulse transient strength should exceed pad.\n";
        return false;
    }
    if (!(pulse_features.instability < pad_features.instability)) {
        std::cerr << "Sparse usability scan test failed: pulse instability should be lower than pad instability.\n";
        return false;
    }
    return true;
}

bool test_provider_scan_finds_rhythmic_regions() {
    constexpr double sample_rate = 1000.0;
    const auto audio = concat_samples({
        make_sustained_pad(sample_rate, 20.0),
        make_pulse_train(sample_rate, 20.0, 120.0),
        make_sustained_pad(sample_rate, 20.0),
        make_pulse_train(sample_rate, 20.0, 120.0),
        make_sustained_pad(sample_rate, 20.0),
    });

    const beatit::detail::SparseSampleProvider provider =
        [&audio](double start_seconds,
                 double duration_seconds,
                 std::vector<float>* out_samples) -> std::size_t {
            const std::size_t total = audio.size();
            const std::size_t start =
                static_cast<std::size_t>(std::llround(std::max(0.0, start_seconds) * sample_rate));
            const std::size_t count =
                static_cast<std::size_t>(std::llround(std::max(0.0, duration_seconds) * sample_rate));
            if (start >= total) {
                out_samples->clear();
                return 0;
            }
            const std::size_t end = std::min(total, start + count);
            out_samples->assign(audio.begin() + static_cast<std::ptrdiff_t>(start),
                                audio.begin() + static_cast<std::ptrdiff_t>(end));
            return out_samples->size();
        };

    const beatit::detail::SparseUsabilityScanRequest request{
        100.0, 20.0, 20.0, sample_rate, 70.0, 180.0, &provider
    };
    const auto windows = beatit::detail::scan_sparse_usability_windows(request);
    if (windows.size() != 5) {
        std::cerr << "Sparse usability scan test failed: expected 5 windows, got "
                  << windows.size() << ".\n";
        return false;
    }

    if (!(windows[1].score > windows[0].score)) {
        std::cerr << "Sparse usability scan test failed: first rhythmic window should outscore intro pad.\n";
        return false;
    }
    if (!(windows[3].score > windows[2].score)) {
        std::cerr << "Sparse usability scan test failed: second rhythmic window should outscore mid pad.\n";
        return false;
    }

    const auto spans = beatit::detail::build_sparse_usability_spans(windows, 0.0);
    if (spans.size() != 2) {
        std::cerr << "Sparse usability scan test failed: expected 2 usable spans from provider scan, got "
                  << spans.size() << ".\n";
        return false;
    }
    if (spans[0].first_index != 1 || spans[0].last_index != 1 ||
        spans[1].first_index != 3 || spans[1].last_index != 3) {
        std::cerr << "Sparse usability scan test failed: span index bounds did not isolate rhythmic windows.\n";
        return false;
    }
    return true;
}

bool test_target_picker_selects_expected_windows() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        0.0, 20.0, {0.06, 0.75, 0.90, 0.84, 0.10}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        20.0, 20.0, {0.70, 0.10, 0.15, 0.10, 0.80}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        40.0, 20.0, {0.08, 0.78, 0.88, 0.82, 0.12}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        60.0, 20.0, {0.68, 0.10, 0.15, 0.10, 0.82}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        80.0, 20.0, {0.05, 0.80, 0.92, 0.86, 0.08}));

    const auto targets =
        beatit::detail::pick_sparse_usability_targets(windows, 0.0);
    if (targets.left_index != 0) {
        std::cerr << "Sparse usability scan test failed: expected left target index 0, got "
                  << targets.left_index << ".\n";
        return false;
    }
    if (targets.between_index != 0) {
        std::cerr << "Sparse usability scan test failed: expected between target index 0, got "
                  << targets.between_index << ".\n";
        return false;
    }
    if (targets.middle_index != 2) {
        std::cerr << "Sparse usability scan test failed: expected middle target index 2, got "
                  << targets.middle_index << ".\n";
        return false;
    }
    if (targets.right_index != 4) {
        std::cerr << "Sparse usability scan test failed: expected right target index 4, got "
                  << targets.right_index << ".\n";
        return false;
    }
    return true;
}

bool test_target_picker_respects_non_zero_window_origin() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        30.0, 20.0, {0.05, 0.78, 0.91, 0.85, 0.09}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        50.0, 20.0, {0.70, 0.10, 0.15, 0.10, 0.80}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        70.0, 20.0, {0.06, 0.79, 0.89, 0.83, 0.11}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        90.0, 20.0, {0.68, 0.10, 0.15, 0.10, 0.82}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        110.0, 20.0, {0.04, 0.81, 0.93, 0.87, 0.08}));

    const auto targets =
        beatit::detail::pick_sparse_usability_targets(windows, 0.0);
    if (targets.left_index != 0) {
        std::cerr << "Sparse usability scan test failed: expected non-zero-origin left target index 0, got "
                  << targets.left_index << ".\n";
        return false;
    }
    if (targets.between_index != 0) {
        std::cerr << "Sparse usability scan test failed: expected non-zero-origin between target index 0, got "
                  << targets.between_index << ".\n";
        return false;
    }
    if (targets.middle_index != 2) {
        std::cerr << "Sparse usability scan test failed: expected non-zero-origin middle target index 2, got "
                  << targets.middle_index << ".\n";
        return false;
    }
    if (targets.right_index != 4) {
        std::cerr << "Sparse usability scan test failed: expected non-zero-origin right target index 4, got "
                  << targets.right_index << ".\n";
        return false;
    }
    return true;
}

bool test_covering_window_picker_prefers_best_covering_score() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        30.0, 30.0, {0.08, 0.72, 0.76, 0.70, 0.16}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        40.0, 30.0, {0.05, 0.82, 0.90, 0.86, 0.08}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        50.0, 30.0, {0.06, 0.78, 0.84, 0.80, 0.10}));

    const std::size_t index =
        beatit::detail::find_covering_sparse_usability_window(windows, 55.0);
    if (index != 1) {
        std::cerr << "Sparse usability scan test failed: expected covering window index 1, got "
                  << index << ".\n";
        return false;
    }
    return true;
}

bool test_resolve_interior_targets_keeps_usable_windows() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        10.0, 20.0, {0.05, 0.78, 0.91, 0.85, 0.09}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        30.0, 20.0, {0.04, 0.80, 0.93, 0.87, 0.08}));

    const auto targets =
        beatit::detail::resolve_sparse_interior_targets(windows, 10.0, 10.0, 0.0);
    if (targets.middle_overridden || targets.between_overridden) {
        std::cerr << "Sparse usability scan test failed: usable interior windows should not be overridden.\n";
        return false;
    }
    if (targets.middle_start_seconds != 10.0 || targets.between_start_seconds != 10.0) {
        std::cerr << "Sparse usability scan test failed: usable interior window starts changed unexpectedly.\n";
        return false;
    }
    return true;
}

bool test_resolve_interior_targets_promotes_middle_and_keeps_distinct_between() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        10.0, 20.0, {0.70, 0.10, 0.15, 0.10, 0.82}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        30.0, 20.0, {0.06, 0.79, 0.90, 0.84, 0.10}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        50.0, 20.0, {0.05, 0.81, 0.92, 0.86, 0.08}));

    const auto targets =
        beatit::detail::resolve_sparse_interior_targets(windows, 10.0, 20.0, 0.0);
    if (!targets.middle_overridden) {
        std::cerr << "Sparse usability scan test failed: middle should have been overridden.\n";
        return false;
    }
    if (!targets.between_overridden) {
        std::cerr << "Sparse usability scan test failed: between should have been overridden.\n";
        return false;
    }
    if (targets.middle_start_seconds != 30.0 || targets.between_start_seconds != 50.0) {
        std::cerr << "Sparse usability scan test failed: promoted middle and distinct between starts not as expected.\n";
        return false;
    }
    return true;
}

bool test_resolve_interior_targets_collapse_between_when_no_distinct_window_exists() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        30.0, 20.0, {0.04, 0.80, 0.93, 0.87, 0.08}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        50.0, 20.0, {0.70, 0.10, 0.15, 0.10, 0.82}));

    const auto targets =
        beatit::detail::resolve_sparse_interior_targets(windows, 30.0, 20.0, 0.0);
    if (targets.middle_overridden) {
        std::cerr << "Sparse usability scan test failed: usable middle should not be overridden.\n";
        return false;
    }
    if (!targets.between_overridden) {
        std::cerr << "Sparse usability scan test failed: unusable between should have been overridden.\n";
        return false;
    }
    if (targets.middle_start_seconds != 30.0 || targets.between_start_seconds != 30.0) {
        std::cerr << "Sparse usability scan test failed: between should collapse to middle when no distinct usable region exists.\n";
        return false;
    }
    return true;
}

bool test_resolve_interior_targets_only_repairs_between_when_middle_is_usable() {
    std::vector<beatit::detail::SparseUsabilityWindow> windows;
    windows.push_back(beatit::detail::build_sparse_usability_window(
        10.0, 20.0, {0.05, 0.78, 0.91, 0.85, 0.09}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        30.0, 20.0, {0.70, 0.10, 0.15, 0.10, 0.82}));
    windows.push_back(beatit::detail::build_sparse_usability_window(
        50.0, 20.0, {0.04, 0.80, 0.93, 0.87, 0.08}));

    const auto targets =
        beatit::detail::resolve_sparse_interior_targets(windows, 10.0, 30.0, 0.0);
    if (targets.middle_overridden) {
        std::cerr << "Sparse usability scan test failed: usable middle should not be overridden.\n";
        return false;
    }
    if (!targets.between_overridden) {
        std::cerr << "Sparse usability scan test failed: unusable between should have been overridden.\n";
        return false;
    }
    if (targets.middle_start_seconds != 10.0 || targets.between_start_seconds != 50.0) {
        std::cerr << "Sparse usability scan test failed: between should have been redirected to a distinct usable region.\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!test_scoring_rewards_rhythmic_windows()) {
        return 1;
    }
    if (!test_picker_prefers_nearby_when_scores_are_similar()) {
        return 1;
    }
    if (!test_picker_prefers_much_better_window_within_snap_budget()) {
        return 1;
    }
    if (!test_picker_rejects_windows_beyond_snap_budget()) {
        return 1;
    }
    if (!test_window_builder_keeps_score_consistent()) {
        return 1;
    }
    if (!test_span_builder_groups_contiguous_usable_windows()) {
        return 1;
    }
    if (!test_feature_measurement_distinguishes_pulses_from_pad()) {
        return 1;
    }
    if (!test_provider_scan_finds_rhythmic_regions()) {
        return 1;
    }
    if (!test_target_picker_selects_expected_windows()) {
        return 1;
    }
    if (!test_target_picker_respects_non_zero_window_origin()) {
        return 1;
    }
    if (!test_covering_window_picker_prefers_best_covering_score()) {
        return 1;
    }
    if (!test_resolve_interior_targets_keeps_usable_windows()) {
        return 1;
    }
    if (!test_resolve_interior_targets_promotes_middle_and_keeps_distinct_between()) {
        return 1;
    }
    if (!test_resolve_interior_targets_collapse_between_when_no_distinct_window_exists()) {
        return 1;
    }
    if (!test_resolve_interior_targets_only_repairs_between_when_middle_is_usable()) {
        return 1;
    }

    std::cout << "Sparse usability scan test passed.\n";
    return 0;
}
