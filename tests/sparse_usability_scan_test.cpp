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

    std::cout << "Sparse usability scan test passed.\n";
    return 0;
}
