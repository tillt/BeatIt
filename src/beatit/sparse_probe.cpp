//
//  sparse_probe.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/sparse_probe.h"
#include "beatit/sparse_probe_selection.h"
#include "beatit/sparse_refinement.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace beatit {
namespace detail {

AnalysisResult analyze_sparse_probe_window(const CoreMLConfig& original_config,
                                           double sample_rate,
                                           double total_duration_seconds,
                                           const SparseSampleProvider& provider,
                                           const SparseRunProbe& run_probe_fn,
                                           const SparseEstimateBpm& estimate_bpm_from_beats_fn,
                                           const SparseNormalizeBpm& normalize_bpm_fn) {
    AnalysisResult result;

    if (!provider || !run_probe_fn || sample_rate <= 0.0 || total_duration_seconds <= 0.0) {
        return result;
    }

    auto estimate_bpm_from_beats_local = [&](const std::vector<unsigned long long>& beat_samples,
                                             double sample_rate_for_beats) -> float {
        return estimate_bpm_from_beats_fn(beat_samples, sample_rate_for_beats);
    };
    auto normalize_bpm_to_range_local = [&](float bpm,
                                            float min_bpm,
                                            float max_bpm) -> float {
        return normalize_bpm_fn(bpm, min_bpm, max_bpm);
    };

    SparseProbeSelectionParams selection_params;
    selection_params.config = &original_config;
    selection_params.sample_rate = sample_rate;
    selection_params.total_duration_seconds = total_duration_seconds;
    selection_params.provider = &provider;
    selection_params.run_probe = &run_probe_fn;
    const SparseProbeSelectionResult selected = select_sparse_probe_result(selection_params);

    result = selected.result;
    const bool needs_bounded_refit =
        selected.low_confidence ||
        (std::isfinite(selected.selected_intro_median_abs_ms) &&
         selected.selected_intro_median_abs_ms > 60.0);
    if (needs_bounded_refit) {
        apply_sparse_bounded_grid_refit(&result, sample_rate);
    }
    apply_sparse_anchor_state_refit(&result,
                                    sample_rate,
                                    selected.probe_duration,
                                    selected.probes,
                                    original_config.verbose);

    SparseWaveformRefitParams waveform_refit_params;
    waveform_refit_params.config = &original_config;
    waveform_refit_params.provider = &provider;
    waveform_refit_params.estimate_bpm_from_beats = &estimate_bpm_from_beats_fn;
    waveform_refit_params.probes = &selected.probes;
    waveform_refit_params.sample_rate = sample_rate;
    waveform_refit_params.probe_duration = selected.probe_duration;
    waveform_refit_params.between_probe_start = selected.between_probe_start;
    waveform_refit_params.middle_probe_start = selected.middle_probe_start;
    apply_sparse_waveform_edge_refit(&result, waveform_refit_params);

    {
        // Keep reported BPM consistent with the returned beat grid.
        const float grid_bpm = normalize_bpm_to_range_local(
            estimate_bpm_from_beats_local(output_beat_sample_frames(result), sample_rate),
            std::max(1.0f, original_config.min_bpm),
            std::max(std::max(1.0f, original_config.min_bpm) + 1.0f, original_config.max_bpm));
        if (grid_bpm > 0.0f) {
            result.estimated_bpm = grid_bpm;
        } else if (selected.have_consensus && selected.consensus_bpm > 0.0) {
            result.estimated_bpm = static_cast<float>(selected.consensus_bpm);
        }
    }
    rebuild_output_beat_events(&result, sample_rate, original_config);

    return result;
}

} // namespace detail
} // namespace beatit
