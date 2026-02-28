//
//  usability_scan.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace beatit {
namespace detail {

struct SparseUsabilityFeatures {
    double silence_ratio = 1.0;
    double onset_density = 0.0;
    double periodicity = 0.0;
    double transient_strength = 0.0;
    double instability = 1.0;
};

struct SparseUsabilityWindow {
    double start_seconds = 0.0;
    double duration_seconds = 0.0;
    SparseUsabilityFeatures features;
    double score = 0.0;
    bool usable = false;
};

struct SparseUsabilityPickRequest {
    double target_seconds = 0.0;
    double max_snap_seconds = std::numeric_limits<double>::infinity();
    double distance_weight = 0.01;
    double min_score = 0.0;
};

struct SparseUsabilitySpan {
    double start_seconds = 0.0;
    double end_seconds = 0.0;
    double mean_score = 0.0;
    std::size_t first_index = 0;
    std::size_t last_index = 0;
};

double score_sparse_usability_window(const SparseUsabilityFeatures& features);

bool sparse_window_is_usable(const SparseUsabilityFeatures& features);

SparseUsabilityWindow build_sparse_usability_window(double start_seconds,
                                                    double duration_seconds,
                                                    const SparseUsabilityFeatures& features);

SparseUsabilityFeatures measure_sparse_usability_features(const std::vector<float>& samples,
                                                          double sample_rate,
                                                          double min_bpm,
                                                          double max_bpm);

std::size_t pick_sparse_usability_window(const std::vector<SparseUsabilityWindow>& windows,
                                         const SparseUsabilityPickRequest& request);

std::vector<SparseUsabilitySpan> build_sparse_usability_spans(
    const std::vector<SparseUsabilityWindow>& windows,
    double min_score);

} // namespace detail
} // namespace beatit
