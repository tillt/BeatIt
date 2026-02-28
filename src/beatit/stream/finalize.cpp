//
//  finalize.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/analysis/internal.h"
#include "beatit/post/window.h"
#include "beatit/stream.h"
#include "beatit/inference/window_merge.h"
#include "beatit/inference/backend.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace beatit {

namespace {

struct ProjectedPhaseState {
    std::size_t bpb = 1;
    std::size_t phase = 0;
    bool valid = false;
};

std::vector<std::size_t> as_size_t_frames(const std::vector<unsigned long long>& frames) {
    std::vector<std::size_t> converted;
    converted.reserve(frames.size());
    for (unsigned long long frame : frames) {
        converted.push_back(static_cast<std::size_t>(frame));
    }
    return converted;
}

ProjectedPhaseState preserve_projected_phase(const std::vector<unsigned long long>& projected_beats,
                                             const std::vector<unsigned long long>& projected_downbeats,
                                             const BeatitConfig& config) {
    if (projected_beats.size() < 2 || projected_downbeats.empty()) {
        return {};
    }

    const std::vector<std::size_t> beat_frames = as_size_t_frames(projected_beats);
    const std::vector<std::size_t> downbeat_frames = as_size_t_frames(projected_downbeats);
    const auto inferred =
        detail::infer_bpb_phase(beat_frames, downbeat_frames, {3, 4}, config);

    ProjectedPhaseState state;
    state.bpb = std::max<std::size_t>(1, inferred.first);
    state.phase = inferred.second % state.bpb;
    state.valid = true;
    return state;
}

void rebuild_projected_downbeats(std::vector<unsigned long long>* projected_downbeats,
                                 const std::vector<unsigned long long>& projected_beats_before_dedupe,
                                 const std::vector<unsigned long long>& projected_beats_after_dedupe,
                                 const ProjectedPhaseState& preserved_phase) {
    if (!projected_downbeats || !preserved_phase.valid || projected_beats_after_dedupe.size() < 2) {
        return;
    }

    const std::vector<std::size_t> beats_before = as_size_t_frames(projected_beats_before_dedupe);
    const std::vector<std::size_t> beats_after = as_size_t_frames(projected_beats_after_dedupe);

    std::size_t prefix_dropped = 0;
    if (!beats_after.empty()) {
        prefix_dropped = static_cast<std::size_t>(
            std::lower_bound(beats_before.begin(), beats_before.end(), beats_after.front()) -
            beats_before.begin());
    }

    const std::size_t adjusted_phase =
        (preserved_phase.phase + preserved_phase.bpb - (prefix_dropped % preserved_phase.bpb)) %
        preserved_phase.bpb;

    const std::vector<std::size_t> rebuilt =
        detail::project_downbeats_from_beats(beats_after, preserved_phase.bpb, adjusted_phase);

    projected_downbeats->clear();
    projected_downbeats->reserve(rebuilt.size());
    for (std::size_t frame : rebuilt) {
        projected_downbeats->push_back(static_cast<unsigned long long>(frame));
    }
}

} // namespace

AnalysisResult BeatitStream::finalize() {
    AnalysisResult result;

    const auto finalize_start = std::chrono::steady_clock::now();
    if (!coreml_enabled_) {
        return result;
    }

    if (coreml_config_.fixed_frames > 0 && coreml_config_.pad_final_window) {
        const std::size_t window_samples =
            coreml_config_.frame_size + (coreml_config_.fixed_frames - 1) * coreml_config_.hop_size;
        const std::size_t available =
            resampled_buffer_.size() > resampled_offset_
                ? resampled_buffer_.size() - resampled_offset_
                : 0;
        if (available > 0 && window_samples > 0) {
            std::vector<float> window(window_samples, 0.0f);
            const float* start_ptr = resampled_buffer_.data() + resampled_offset_;
            std::copy(start_ptr, start_ptr + std::min(available, window_samples), window.begin());

            BeatitConfig local_config = coreml_config_;
            local_config.tempo_window_percent = 0.0f;
            local_config.prefer_double_time = false;

            const auto infer_start = std::chrono::steady_clock::now();
            std::vector<float> beat_activation;
            std::vector<float> downbeat_activation;
            detail::InferenceTiming timing;
            const bool ok = inference_backend_ &&
                inference_backend_->infer_window(window,
                                                 local_config,
                                                 &beat_activation,
                                                 &downbeat_activation,
                                                 &timing);
            const auto infer_end = std::chrono::steady_clock::now();
            perf_.mel_ms += timing.mel_ms;
            perf_.torch_forward_ms += timing.torch_forward_ms;
            perf_.finalize_infer_ms +=
                std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
            if (!ok) {
                BEATIT_LOG_ERROR("Stream finalize inference failed."
                                 << " backend=" << static_cast<int>(local_config.backend)
                                 << " frame_offset=" << coreml_frame_offset_);
                return result;
            }
            detail::trim_activation_to_frames(&beat_activation, local_config.fixed_frames);
            detail::trim_activation_to_frames(&downbeat_activation, local_config.fixed_frames);
            detail::merge_window_activations(&coreml_beat_activation_,
                                             &coreml_downbeat_activation_,
                                             coreml_frame_offset_,
                                             local_config.fixed_frames,
                                             beat_activation,
                                             downbeat_activation,
                                             0);
        }
    }

    if (coreml_beat_activation_.empty()) {
        return result;
    }

    BeatitConfig base_config = coreml_config_;
    base_config.tempo_window_percent = 0.0f;
    base_config.prefer_double_time = false;
    base_config.dbn_window_seconds = 0.0;

    std::size_t last_active_frame = 0;
    std::size_t full_frame_count = 0;
    if (coreml_config_.hop_size > 0 && sample_rate_ > 0.0) {
        const double ratio = coreml_config_.sample_rate / sample_rate_;
        const double total_pos = static_cast<double>(total_seen_samples_) * ratio;
        full_frame_count =
            static_cast<std::size_t>(std::llround(total_pos / coreml_config_.hop_size));
        if (coreml_config_.disable_silence_trimming) {
            last_active_frame = full_frame_count;
        } else if (has_active_sample_) {
            const double sample_pos = static_cast<double>(last_active_sample_) * ratio;
            last_active_frame =
                static_cast<std::size_t>(std::llround(sample_pos / coreml_config_.hop_size));
        }
    }

    const auto postprocess_start = std::chrono::steady_clock::now();
    CoreMLResult base = postprocess_coreml_activations(coreml_beat_activation_,
                                                      coreml_downbeat_activation_,
                                                      &coreml_phase_energy_,
                                                      base_config,
                                                      sample_rate_,
                                                      0.0f,
                                                      last_active_frame,
                                                      full_frame_count);
    const float bpm_min = std::max(1.0f, coreml_config_.min_bpm);
    const float bpm_max = std::max(bpm_min + 1.0f, coreml_config_.max_bpm);
    const float peaks_bpm_raw = estimate_bpm_from_activation(
        coreml_beat_activation_, coreml_config_, sample_rate_);
    const float autocorr_bpm_raw = estimate_bpm_from_activation_autocorr(
        coreml_beat_activation_, coreml_config_, sample_rate_);
    const float comb_bpm_raw =
        estimate_bpm_from_activation_comb(coreml_beat_activation_, coreml_config_, sample_rate_);
    const float beats_bpm_raw = estimate_bpm_from_beats(base.beat_sample_frames, sample_rate_);
    const float peaks_bpm = normalize_bpm_to_range(peaks_bpm_raw, bpm_min, bpm_max);
    const float autocorr_bpm = normalize_bpm_to_range(autocorr_bpm_raw, bpm_min, bpm_max);
    const float comb_bpm = normalize_bpm_to_range(comb_bpm_raw, bpm_min, bpm_max);
    const float beats_bpm = normalize_bpm_to_range(beats_bpm_raw, bpm_min, bpm_max);
    const float candidate_bpm = choose_candidate_bpm(peaks_bpm, autocorr_bpm, comb_bpm, beats_bpm);
    const double prior_bpm = tempo_reference_valid_ ? tempo_reference_bpm_ : 0.0;
    float reference_bpm = candidate_bpm;
    std::string tempo_state = "init";
    float consensus_ratio = 0.0f;
    float prior_ratio = 0.0f;

    if (candidate_bpm > 0.0f) {
        std::vector<float> anchors;
        anchors.reserve(4);
        if (peaks_bpm > 0.0f) {
            anchors.push_back(peaks_bpm);
        }
        if (autocorr_bpm > 0.0f) {
            anchors.push_back(autocorr_bpm);
        }
        if (comb_bpm > 0.0f) {
            anchors.push_back(comb_bpm);
        }
        if (beats_bpm > 0.0f) {
            anchors.push_back(beats_bpm);
        }
        const float consensus_tol = 0.02f;
        std::size_t consensus = 0;
        for (float value : anchors) {
            if (std::abs(value - candidate_bpm) / std::max(candidate_bpm, 1e-6f) <= consensus_tol) {
                ++consensus;
            }
        }
        consensus_ratio = static_cast<float>(consensus) / std::max<std::size_t>(1, anchors.size());

        if (tempo_reference_valid_ && prior_bpm > 0.0) {
            prior_ratio =
                static_cast<float>(std::abs(candidate_bpm - prior_bpm) / prior_bpm);
            const float hold_tol = 0.02f;
            const float switch_tol = std::max(hold_tol, coreml_config_.dbn_interval_tolerance);
            if (prior_ratio <= hold_tol) {
                reference_bpm =
                    static_cast<float>(0.7 * prior_bpm + 0.3 * candidate_bpm);
                tempo_state = "blend";
            } else if (prior_ratio <= switch_tol || consensus >= 2) {
                reference_bpm = candidate_bpm;
                tempo_state = "switch";
            } else {
                reference_bpm = static_cast<float>(prior_bpm);
                tempo_state = "hold";
            }
        }
    }

    if (reference_bpm > 0.0f) {
        tempo_reference_bpm_ = reference_bpm;
        tempo_reference_valid_ = true;
    }
    BEATIT_LOG_DEBUG("Tempo anchor: peaks=" << peaks_bpm
                     << " autocorr=" << autocorr_bpm
                     << " comb=" << comb_bpm
                     << " beats=" << beats_bpm
                     << " chosen=" << reference_bpm
                     << " prior=" << prior_bpm
                     << " state=" << tempo_state
                     << " ratio=" << prior_ratio
                     << " consensus=" << consensus_ratio);

    CoreMLResult final_result = postprocess_coreml_activations(coreml_beat_activation_,
                                                              coreml_downbeat_activation_,
                                                              &coreml_phase_energy_,
                                                              coreml_config_,
                                                              sample_rate_,
                                                              reference_bpm,
                                                              last_active_frame,
                                                              full_frame_count);
    const auto postprocess_end = std::chrono::steady_clock::now();
    perf_.postprocess_ms +=
        std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();

    result.coreml_beat_activation = std::move(final_result.beat_activation);
    result.coreml_downbeat_activation = std::move(final_result.downbeat_activation);
    result.coreml_beat_feature_frames = std::move(final_result.beat_feature_frames);
    result.coreml_beat_sample_frames = std::move(final_result.beat_sample_frames);
    result.coreml_beat_projected_feature_frames =
        std::move(final_result.beat_projected_feature_frames);
    result.coreml_beat_projected_sample_frames =
        std::move(final_result.beat_projected_sample_frames);
    result.coreml_beat_strengths = std::move(final_result.beat_strengths);
    result.coreml_downbeat_feature_frames = std::move(final_result.downbeat_feature_frames);
    result.coreml_downbeat_projected_feature_frames =
        std::move(final_result.downbeat_projected_feature_frames);
    result.coreml_phase_energy = std::move(coreml_phase_energy_);
    // Keep reported BPM consistent with the returned beat grid.
    float estimated_bpm =
        normalize_bpm_to_range(estimate_bpm_from_beats(output_beat_sample_frames(result), sample_rate_),
                               bpm_min,
                               bpm_max);
    if (!(estimated_bpm > 0.0f)) {
        const float anchored_bpm = normalize_bpm_to_range(reference_bpm, bpm_min, bpm_max);
        if (anchored_bpm > 0.0f) {
            estimated_bpm = anchored_bpm;
        }
    }
    result.estimated_bpm = estimated_bpm;

    if (prepend_samples_ > 0) {
        for (auto& frame : result.coreml_beat_sample_frames) {
            frame = frame > prepend_samples_ ? frame - prepend_samples_ : 0;
        }
        for (auto& frame : result.coreml_beat_projected_sample_frames) {
            frame = frame > prepend_samples_ ? frame - prepend_samples_ : 0;
        }
    }

    auto dedupe_monotonic = [](std::vector<unsigned long long>& samples,
                               std::vector<unsigned long long>* feature_frames,
                               std::vector<float>* strengths) {
        if (samples.empty()) {
            return;
        }
        std::size_t write = 1;
        unsigned long long last = samples[0];
        for (std::size_t i = 1; i < samples.size(); ++i) {
            const unsigned long long current = samples[i];
            if (current <= last) {
                continue;
            }
            samples[write] = current;
            if (feature_frames && i < feature_frames->size()) {
                (*feature_frames)[write] = (*feature_frames)[i];
            }
            if (strengths && i < strengths->size()) {
                (*strengths)[write] = (*strengths)[i];
            }
            last = current;
            ++write;
        }
        samples.resize(write);
        if (feature_frames && feature_frames->size() >= write) {
            feature_frames->resize(write);
        }
        if (strengths && strengths->size() >= write) {
            strengths->resize(write);
        }
    };

    dedupe_monotonic(result.coreml_beat_sample_frames,
                     &result.coreml_beat_feature_frames,
                     &result.coreml_beat_strengths);
    const std::vector<unsigned long long> projected_beats_before_dedupe =
        result.coreml_beat_projected_feature_frames;
    const ProjectedPhaseState preserved_phase =
        preserve_projected_phase(result.coreml_beat_projected_feature_frames,
                                 result.coreml_downbeat_projected_feature_frames,
                                 coreml_config_);

    dedupe_monotonic(result.coreml_beat_projected_sample_frames,
                     &result.coreml_beat_projected_feature_frames,
                     nullptr);
    rebuild_projected_downbeats(&result.coreml_downbeat_projected_feature_frames,
                                projected_beats_before_dedupe,
                                result.coreml_beat_projected_feature_frames,
                                preserved_phase);
    const auto marker_start = std::chrono::steady_clock::now();
    rebuild_output_beat_events(result, sample_rate_, coreml_config_);
    const auto marker_end = std::chrono::steady_clock::now();
    perf_.marker_ms +=
        std::chrono::duration<double, std::milli>(marker_end - marker_start).count();

    const auto finalize_end = std::chrono::steady_clock::now();
    perf_.finalize_ms =
        std::chrono::duration<double, std::milli>(finalize_end - finalize_start).count();

    if (coreml_config_.profile) {
        BEATIT_LOG_INFO("Timing(stream): resample=" << perf_.resample_ms
                        << "ms process=" << perf_.process_ms
                        << "ms mel=" << perf_.mel_ms
                        << "ms torch=" << perf_.torch_forward_ms
                        << "ms window_infer=" << perf_.window_infer_ms
                        << "ms windows=" << perf_.window_count
                        << " finalize_infer=" << perf_.finalize_infer_ms
                        << "ms postprocess=" << perf_.postprocess_ms
                        << "ms markers=" << perf_.marker_ms
                        << "ms total_finalize=" << perf_.finalize_ms
                        << "ms");
    }

    return result;
}

} // namespace beatit
