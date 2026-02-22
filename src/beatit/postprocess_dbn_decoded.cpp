//
//  postprocess_dbn_decoded.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/postprocess_dbn_decoded.h"

#include "beatit/postprocess_helpers.h"
#include "beatit/postprocess_result_ops.h"
#include "beatit/postprocess_window.h"
#include "postprocess_dbn_grid_anchor.h"
#include "postprocess_dbn_grid_stages.h"
#include "postprocess_dbn_tempo_decision.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace beatit::detail {

bool run_dbn_decoded_postprocess(CoreMLResult& result,
                                 DBNDecodeResult& decoded,
                                 const DBNDecodedPostprocessContext& context) {
    const CoreMLConfig& config = context.processing.config;
    const CalmdadDecoder& calmdad_decoder = context.processing.calmdad_decoder;
    const double sample_rate = context.processing.sample_rate;
    const double fps = context.processing.fps;
    const double hop_scale = context.processing.hop_scale;
    const std::size_t analysis_latency_frames = context.processing.analysis_latency_frames;
    const double analysis_latency_frames_f = context.processing.analysis_latency_frames_f;
    const std::size_t refine_window = context.processing.refine_window;

    const std::size_t used_frames = context.window.used_frames;
    const bool use_window = context.window.use_window;
    const std::size_t window_start = context.window.window_start;
    const std::vector<float>& beat_slice = context.window.beat_slice;
    const std::vector<float>& downbeat_slice = context.window.downbeat_slice;

    const float reference_bpm = context.bpm.reference_bpm;
    const std::size_t grid_total_frames = context.bpm.grid_total_frames;
    const float min_bpm = context.bpm.min_bpm;
    const float max_bpm = context.bpm.max_bpm;

    const bool quality_valid = context.quality.valid;
    const double quality_qpar = context.quality.qpar;
    const double quality_qkur = context.quality.qkur;

    auto fill_beats_from_frames = [&](const std::vector<std::size_t>& frames) {
        detail::fill_beats_from_frames(result,
                                       frames,
                                       config,
                                       sample_rate,
                                       hop_scale,
                                       analysis_latency_frames,
                                       analysis_latency_frames_f,
                                       refine_window);
    };

    auto refine_frame_to_peak = [&](std::size_t frame,
                                    const std::vector<float>& activation) -> std::size_t {
        return detail::refine_frame_to_peak(frame, activation, refine_window);
    };

    auto fill_beats_from_bpm_grid_into = [&](std::size_t start_frame,
                                             double bpm,
                                             std::size_t total_frames,
                                             std::vector<unsigned long long>& out_feature_frames,
                                             std::vector<unsigned long long>& out_sample_frames,
                                             std::vector<float>& out_strengths) {
        detail::fill_beats_from_bpm_grid_into(result.beat_activation,
                                              config,
                                              sample_rate,
                                              fps,
                                              hop_scale,
                                              start_frame,
                                              bpm,
                                              total_frames,
                                              out_feature_frames,
                                              out_sample_frames,
                                              out_strengths);
    };

    auto apply_latency_to_frames = [&](const std::vector<std::size_t>& frames) {
        return detail::apply_latency_to_frames(frames, analysis_latency_frames);
    };

    auto infer_bpb_phase = [&](const std::vector<std::size_t>& beats,
                               const std::vector<std::size_t>& downbeats,
                               const std::vector<std::size_t>& candidates) {
        return detail::infer_bpb_phase(beats, downbeats, candidates, config);
    };
    double projected_bpm = 0.0;

    auto apply_window_offset = [&] {
        if (!use_window) {
            return;
        }
        for (std::size_t& frame : decoded.beat_frames) {
            frame += window_start;
        }
        for (std::size_t& frame : decoded.downbeat_frames) {
            frame += window_start;
        }
    };

    auto process_grid_projection = [&] {
        if (!config.dbn_project_grid) {
            return;
        }
        std::vector<std::size_t> refined_beats;
        refined_beats.reserve(decoded.beat_frames.size());
        for (std::size_t frame : decoded.beat_frames) {
            refined_beats.push_back(refine_frame_to_peak(frame, result.beat_activation));
        }
        decoded.beat_frames = std::move(refined_beats);
        detail::dedupe_frames(decoded.beat_frames);

        if (!decoded.downbeat_frames.empty()) {
            const std::vector<float>& downbeat_source =
                result.downbeat_activation.empty() ? result.beat_activation
                                                   : result.downbeat_activation;
            std::vector<std::size_t> refined_downbeats;
            refined_downbeats.reserve(decoded.downbeat_frames.size());
            for (std::size_t frame : decoded.downbeat_frames) {
                refined_downbeats.push_back(refine_frame_to_peak(frame, downbeat_source));
            }
            decoded.downbeat_frames = std::move(refined_downbeats);
            detail::dedupe_frames(decoded.downbeat_frames);
        }

        const GridTempoDecisionInput tempo_input{
            decoded,
            result,
            config,
            calmdad_decoder,
            use_window,
            beat_slice,
            downbeat_slice,
            quality_valid,
            quality_qpar,
            quality_qkur,
            used_frames,
            min_bpm,
            max_bpm,
            reference_bpm,
            fps,
        };
        const GridTempoDecision tempo_decision = compute_grid_tempo_decision(tempo_input);
        log_grid_tempo_decision(tempo_decision, tempo_input);
        projected_bpm = tempo_decision.bpm_for_grid;

        const GridAnchorSeed anchor_seed = seed_grid_anchor(decoded,
                                                            result,
                                                            config,
                                                            use_window,
                                                            window_start,
                                                            used_frames,
                                                            tempo_decision.base_interval);

        GridProjectionState grid_state;
        grid_state.bpb = tempo_decision.bpb;
        grid_state.phase = tempo_decision.phase;
        grid_state.best_phase = tempo_decision.phase;
        grid_state.best_score = -std::numeric_limits<double>::infinity();
        grid_state.base_interval = tempo_decision.base_interval;
        grid_state.step_frames = tempo_decision.step_frames;
        grid_state.earliest_peak = anchor_seed.earliest_peak;
        grid_state.earliest_downbeat_peak = anchor_seed.earliest_downbeat_peak;
        grid_state.strongest_peak = anchor_seed.strongest_peak;
        grid_state.strongest_peak_value = anchor_seed.strongest_peak_value;
        grid_state.activation_floor = anchor_seed.activation_floor;

        select_downbeat_phase(grid_state,
                              decoded,
                              result,
                              config,
                              tempo_decision.quality_low,
                              tempo_decision.downbeat_override_ok,
                              use_window,
                              window_start,
                              used_frames,
                              fps);
        synthesize_uniform_grid(grid_state, decoded, result, config, used_frames, fps);
    };

    auto emit_projected_grid_outputs = [&] {
        if (config.dbn_project_grid && decoded.beat_frames.size() >= 2 && projected_bpm > 0.0) {
            fill_beats_from_bpm_grid_into(decoded.beat_frames.front(),
                                          projected_bpm,
                                          grid_total_frames,
                                          result.beat_projected_feature_frames,
                                          result.beat_projected_sample_frames,
                                          result.beat_projected_strengths);
            std::vector<std::size_t> projected_frames;
            projected_frames.reserve(result.beat_projected_feature_frames.size());
            for (unsigned long long frame : result.beat_projected_feature_frames) {
                projected_frames.push_back(static_cast<std::size_t>(frame));
            }
            std::size_t projected_bpb = std::max<std::size_t>(1, config.dbn_beats_per_bar);
            std::size_t projected_phase = 0;
            if (!decoded.downbeat_frames.empty()) {
                const auto inferred =
                    infer_bpb_phase(decoded.beat_frames, decoded.downbeat_frames, {3, 4});
                projected_bpb = std::max<std::size_t>(1, inferred.first);
                projected_phase = inferred.second % projected_bpb;
            }
            projected_phase = guard_projected_downbeat_phase(projected_frames,
                                                             result.downbeat_activation,
                                                             projected_bpb,
                                                             projected_phase,
                                                             config.verbose);
            // Preserve DBN-selected bar phase on the projected beat grid.
            // Nearest-neighbor remapping can flip bars by one beat when
            // projection tempo differs slightly from the decoded beat list.
            const std::vector<std::size_t> projected_downbeats =
                project_downbeats_from_beats(projected_frames, projected_bpb, projected_phase);
            result.downbeat_projected_feature_frames.clear();
            result.downbeat_projected_feature_frames.reserve(projected_downbeats.size());
            for (std::size_t frame : projected_downbeats) {
                result.downbeat_projected_feature_frames.push_back(
                    static_cast<unsigned long long>(frame));
            }
        } else {
            result.beat_projected_feature_frames.clear();
            result.beat_projected_sample_frames.clear();
            result.beat_projected_strengths.clear();
            result.downbeat_projected_feature_frames.clear();
        }
    };

    auto emit_downbeat_outputs = [&] {
        result.downbeat_feature_frames.clear();
        const std::vector<std::size_t> adjusted_downbeats =
            apply_latency_to_frames(decoded.downbeat_frames);
        result.downbeat_feature_frames.reserve(adjusted_downbeats.size());
        for (std::size_t frame : adjusted_downbeats) {
            result.downbeat_feature_frames.push_back(static_cast<unsigned long long>(frame));
        }
    };

    apply_window_offset();
    process_grid_projection();

    // Use the refined peak interpolation path for decoded DBN beats as well,
    // so sample-frame timing is not quantized to integer feature frames.
    fill_beats_from_frames(decoded.beat_frames);
    emit_projected_grid_outputs();
    emit_downbeat_outputs();
    return true;
}

} // namespace beatit::detail
