//
//  sparse_waveform.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/analysis.h"
#include "beatit/sparse_probe.h"

#include <cstddef>
#include <vector>

namespace beatit {
namespace detail {

enum class SparsePeakMode {
    AbsoluteMax,
    ThresholdedLocalMax,
};

const std::vector<unsigned long long>& sparse_select_beats(const AnalysisResult& result);

bool sparse_load_samples(const SparseSampleProvider& provider,
                         double start_seconds,
                         double duration_seconds,
                         std::vector<float>* out_samples);

std::size_t sparse_waveform_radius(double sample_rate, double bpm_hint);

void sparse_collect_offsets(const std::vector<unsigned long long>& beat_frames,
                            std::size_t first_idx,
                            std::size_t last_idx_exclusive,
                            std::size_t segment_start_frame,
                            const std::vector<float>& samples,
                            std::size_t radius,
                            SparsePeakMode mode,
                            double sample_rate,
                            std::vector<double>* signed_offsets_ms,
                            std::vector<double>* abs_offsets_ms);

double sparse_median_inplace(std::vector<double>* values);

} // namespace detail
} // namespace beatit
