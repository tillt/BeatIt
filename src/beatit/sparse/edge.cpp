//
//  edge.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/sparse/refinement.h"

#include "beatit/logging.hpp"
#include "beatit/sparse/phase_metrics.h"
#include "beatit/sparse/waveform.h"
#include "refine_common.h"
#include "edge_adjust.h"
#include "edge_metrics.h"
#include "edge_phase.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace beatit {
namespace detail {

void apply_sparse_waveform_edge_refit(AnalysisResult& result,
                                      const SparseWaveformRefitParams& params) {
    if (!params.config || !params.provider || !params.probes) {
        return;
    }
    const SparseSampleProvider& provider = *params.provider;
    const std::vector<SparseProbeObservation>& probes = *params.probes;
    const double sample_rate = params.sample_rate;
    const double probe_duration = params.probe_duration;
    if (sample_rate <= 0.0 || probes.empty() || probe_duration <= 0.0) {
        return;
    }

    const bool second_pass_enabled = []() {
        const char* v = std::getenv("BEATIT_EDGE_REFIT_SECOND_PASS");
        if (!v || v[0] == '\0') {
            return true;
        }
        return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' ||
                 v[0] == 'n' || v[0] == 'N');
    }();
    const auto has_valid_metrics = [](const EdgeOffsetMetrics& metrics, std::size_t min_count = 8) {
        return metrics.count >= min_count &&
               std::isfinite(metrics.median_ms) &&
               std::isfinite(metrics.mad_ms);
    };

    std::vector<unsigned long long>* projected = nullptr;
    if (!result.coreml_beat_projected_sample_frames.empty()) {
        projected = &result.coreml_beat_projected_sample_frames;
    } else if (!result.coreml_beat_sample_frames.empty()) {
        projected = &result.coreml_beat_sample_frames;
    }
    if (!projected || projected->size() < 64) {
        return;
    }
    const std::vector<unsigned long long> projected_before_refit = *projected;

    double bpm_hint = result.estimated_bpm > 0.0f
        ? static_cast<double>(result.estimated_bpm)
        : 0.0;
    if (!(bpm_hint > 0.0) &&
        params.estimate_bpm_from_beats &&
        *params.estimate_bpm_from_beats) {
        bpm_hint = (*params.estimate_bpm_from_beats)(*projected, sample_rate);
    }
    if (!(bpm_hint > 0.0)) {
        return;
    }

    auto measure_middle_windows = [&](const std::vector<unsigned long long>& beats) {
        SparseWindowPhaseMetrics between_metrics;
        SparseWindowPhaseMetrics middle_metrics;
        if (beats.size() < 32) {
            return std::pair<SparseWindowPhaseMetrics, SparseWindowPhaseMetrics>{
                between_metrics, middle_metrics};
        }
        AnalysisResult tmp;
        tmp.coreml_beat_projected_sample_frames = beats;
        between_metrics = measure_sparse_window_phase(tmp,
                                                      bpm_hint,
                                                      params.between_probe_start,
                                                      probe_duration,
                                                      sample_rate,
                                                      provider);
        middle_metrics = measure_sparse_window_phase(tmp,
                                                     bpm_hint,
                                                     params.middle_probe_start,
                                                     probe_duration,
                                                     sample_rate,
                                                     provider);
        return std::pair<SparseWindowPhaseMetrics, SparseWindowPhaseMetrics>{
            between_metrics, middle_metrics};
    };

    std::size_t first_probe_index = 0;
    std::size_t last_probe_index = 0;
    for (std::size_t i = 1; i < probes.size(); ++i) {
        if (probes[i].start < probes[first_probe_index].start) {
            first_probe_index = i;
        }
        if (probes[i].start > probes[last_probe_index].start) {
            last_probe_index = i;
        }
    }
    const double first_probe_start = probes[first_probe_index].start;
    const double last_probe_start = probes[last_probe_index].start;
    if (std::abs(last_probe_start - first_probe_start) < 1.0) {
        return;
    }
    const auto clamp_window_start = [&](double s) {
        return std::clamp(s, first_probe_start, last_probe_start);
    };

    auto select_window_beats = [&](const std::vector<unsigned long long>& beats,
                                   double window_start_s) {
        std::vector<unsigned long long> selected;
        if (sample_rate <= 0.0 || beats.empty()) {
            return selected;
        }
        const unsigned long long window_start_frame = static_cast<unsigned long long>(
            std::llround(std::max(0.0, window_start_s) * sample_rate));
        const unsigned long long window_end_frame = static_cast<unsigned long long>(
            std::llround(std::max(0.0, window_start_s + probe_duration) * sample_rate));
        auto begin_it = std::lower_bound(beats.begin(), beats.end(), window_start_frame);
        auto end_it = std::upper_bound(beats.begin(), beats.end(), window_end_frame);
        if (begin_it < end_it) {
            selected.insert(selected.end(), begin_it, end_it);
        }
        return selected;
    };
    auto measure_window_pair = [&](const std::vector<unsigned long long>& beats,
                                   double first_window_start_s,
                                   double last_window_start_s) {
        const std::vector<unsigned long long> first_window_beats =
            select_window_beats(beats, first_window_start_s);
        const std::vector<unsigned long long> last_window_beats =
            select_window_beats(beats, last_window_start_s);
        return std::pair<EdgeOffsetMetrics, EdgeOffsetMetrics>{
            measure_edge_offsets(first_window_beats, bpm_hint, false, sample_rate, provider),
            measure_edge_offsets(last_window_beats, bpm_hint, true, sample_rate, provider)};
    };
    auto window_usable = [](const EdgeOffsetMetrics& m) {
        return m.count >= 10 &&
               std::isfinite(m.median_ms) &&
               std::isfinite(m.mad_ms) &&
               m.mad_ms <= 120.0;
    };
    const double shift_step = std::clamp(probe_duration * 0.25, 5.0, 20.0);
    double first_window_start = first_probe_start;
    double last_window_start = last_probe_start;
    EdgeOffsetMetrics intro;
    EdgeOffsetMetrics outro;
    std::size_t quality_shift_rounds = 0;
    for (; quality_shift_rounds < 6; ++quality_shift_rounds) {
        auto pair = measure_window_pair(*projected, first_window_start, last_window_start);
        intro = pair.first;
        outro = pair.second;
        const bool intro_ok = window_usable(intro);
        const bool outro_ok = window_usable(outro);
        if (intro_ok && outro_ok) {
            break;
        }
        bool moved = false;
        if (!intro_ok) {
            const double next = clamp_window_start(first_window_start + shift_step);
            if (next > first_window_start + 0.5) {
                first_window_start = next;
                moved = true;
            }
        }
        if (!outro_ok) {
            const double next = clamp_window_start(last_window_start - shift_step);
            if (next + 0.5 < last_window_start) {
                last_window_start = next;
                moved = true;
            }
        }
        if (!moved || (last_window_start - first_window_start) < std::max(1.0, shift_step)) {
            break;
        }
    }
    if (!has_valid_metrics(intro) || !has_valid_metrics(outro)) {
        return;
    }

    const double ratio = compute_sparse_edge_ratio(*projected, intro, outro, sample_rate);
    const double applied_ratio =
        apply_sparse_edge_scale(*projected, ratio, 0.9995, 1.0005, 1e-5);
    if (applied_ratio == 1.0) {
        return;
    }

    EdgeOffsetMetrics post_intro = intro;
    EdgeOffsetMetrics post_outro = outro;
    if (second_pass_enabled) {
        auto measured = measure_window_pair(*projected, first_window_start, last_window_start);
        post_intro = measured.first;
        post_outro = measured.second;
        if (has_valid_metrics(post_intro) && has_valid_metrics(post_outro)) {
            const double post_delta = std::abs(post_outro.median_ms - post_intro.median_ms);
            std::vector<unsigned long long> candidate = *projected;
            const double pass2_ratio =
                compute_sparse_edge_ratio(candidate, post_intro, post_outro, sample_rate);
            const double pass2_applied =
                apply_sparse_edge_scale(candidate, pass2_ratio, 0.9997, 1.0003, 1e-6);
            if (pass2_applied != 1.0) {
                auto candidate_measured =
                    measure_window_pair(candidate, first_window_start, last_window_start);
                const EdgeOffsetMetrics cand_intro = candidate_measured.first;
                const EdgeOffsetMetrics cand_outro = candidate_measured.second;
                if (has_valid_metrics(cand_intro) && has_valid_metrics(cand_outro)) {
                    const double cand_delta = std::abs(cand_outro.median_ms - cand_intro.median_ms);
                    const bool improves_delta = cand_delta <= post_delta;
                    const bool keeps_intro =
                        std::abs(cand_intro.median_ms) <= (std::abs(post_intro.median_ms) + 3.0);
                    if (improves_delta && keeps_intro) {
                        *projected = std::move(candidate);
                        post_intro = cand_intro;
                        post_outro = cand_outro;
                    }
                }
            }
        }
    }

    auto try_uniform_shift_on_windows =
        [&](const EdgeOffsetMetrics& base_intro,
            const EdgeOffsetMetrics& base_outro,
            double max_beat_fraction) {
            if (!has_valid_metrics(base_intro) || !has_valid_metrics(base_outro)) {
                return false;
            }
            if ((base_intro.median_ms * base_outro.median_ms) <= 0.0) {
                return false;
            }
            const double mean_ms = 0.5 * (base_intro.median_ms + base_outro.median_ms);
            const double beat_ms = 60000.0 / std::max(1e-6, bpm_hint);
            const double max_shift_ms = std::max(25.0, beat_ms * max_beat_fraction);
            const double clamped_shift_ms = std::clamp(mean_ms, -max_shift_ms, max_shift_ms);
            const long long shift_frames = static_cast<long long>(
                std::llround((clamped_shift_ms * sample_rate) / 1000.0));
            if (shift_frames == 0) {
                return false;
            }

            std::vector<unsigned long long> candidate = *projected;
            apply_sparse_uniform_shift(candidate, shift_frames);
            const auto measured =
                measure_window_pair(candidate, first_window_start, last_window_start);
            const EdgeOffsetMetrics cand_intro = measured.first;
            const EdgeOffsetMetrics cand_outro = measured.second;
            if (!has_valid_metrics(cand_intro) || !has_valid_metrics(cand_outro)) {
                return false;
            }

            const double base_worst =
                std::max(std::abs(base_intro.median_ms), std::abs(base_outro.median_ms));
            const double cand_worst =
                std::max(std::abs(cand_intro.median_ms), std::abs(cand_outro.median_ms));
            if (cand_worst + 5.0 < base_worst) {
                *projected = std::move(candidate);
                return true;
            }
            return false;
        };
    if (try_uniform_shift_on_windows(post_intro, post_outro, 0.30)) {
        auto measured = measure_window_pair(*projected, first_window_start, last_window_start);
        post_intro = measured.first;
        post_outro = measured.second;
    }

    EdgeOffsetMetrics global_intro = measure_edge_offsets(*projected, bpm_hint, false, sample_rate, provider);
    EdgeOffsetMetrics global_outro = measure_edge_offsets(*projected, bpm_hint, true, sample_rate, provider);
    double global_guard_ratio = 1.0;
    const auto try_global_guard = [&](double min_delta_ms,
                                      double max_delta_ms,
                                      double ratio_min,
                                      double ratio_max,
                                      double improve_margin_ms,
                                      double intro_margin_ms,
                                      double worst_margin_ms,
                                      bool check_worst) {
        if (!has_valid_metrics(global_intro) || !has_valid_metrics(global_outro)) {
            return;
        }
        const double global_delta = std::abs(global_outro.median_ms - global_intro.median_ms);
        if (global_delta <= min_delta_ms ||
            (std::isfinite(max_delta_ms) && global_delta > max_delta_ms)) {
            return;
        }

        std::vector<unsigned long long> candidate = *projected;
        const double guard_ratio =
            compute_sparse_edge_ratio(candidate, global_intro, global_outro, sample_rate);
        const double guard_applied =
            apply_sparse_edge_scale(candidate, guard_ratio, ratio_min, ratio_max, 1e-6);
        if (guard_applied == 1.0) {
            return;
        }

        const EdgeOffsetMetrics cand_intro =
            measure_edge_offsets(candidate, bpm_hint, false, sample_rate, provider);
        const EdgeOffsetMetrics cand_outro =
            measure_edge_offsets(candidate, bpm_hint, true, sample_rate, provider);
        if (!has_valid_metrics(cand_intro) || !has_valid_metrics(cand_outro)) {
            return;
        }

        const double cand_delta = std::abs(cand_outro.median_ms - cand_intro.median_ms);
        const bool improves_delta = cand_delta <= (global_delta - improve_margin_ms);
        bool keeps_shape = false;
        if (check_worst) {
            const double base_worst =
                std::max(std::abs(global_intro.median_ms), std::abs(global_outro.median_ms));
            const double cand_worst =
                std::max(std::abs(cand_intro.median_ms), std::abs(cand_outro.median_ms));
            keeps_shape = cand_worst <= (base_worst + worst_margin_ms);
        } else {
            keeps_shape = std::abs(cand_intro.median_ms) <=
                          (std::abs(global_intro.median_ms) + intro_margin_ms);
        }

        if (!improves_delta || !keeps_shape) {
            return;
        }

        *projected = std::move(candidate);
        global_intro = cand_intro;
        global_outro = cand_outro;
        global_guard_ratio = guard_applied;
    };
    try_global_guard(30.0, std::numeric_limits<double>::infinity(), 0.99985, 1.00015, 1.0, 4.0, 0.0, false);
    try_global_guard(60.0, 120.0, 0.9996, 1.0004, 2.0, 0.0, 10.0, true);

    if (has_valid_metrics(global_intro) && has_valid_metrics(global_outro) &&
        (global_intro.median_ms * global_outro.median_ms) > 0.0) {
        const double mean_ms = 0.5 * (global_intro.median_ms + global_outro.median_ms);
        const double beat_ms = 60000.0 / std::max(1e-6, bpm_hint);
        const double max_shift_ms = std::max(40.0, beat_ms * 0.35);
        const double clamped_shift_ms = std::clamp(mean_ms, -max_shift_ms, max_shift_ms);
        const long long shift_frames = static_cast<long long>(
            std::llround((clamped_shift_ms * sample_rate) / 1000.0));
        if (shift_frames != 0) {
            std::vector<unsigned long long> candidate = *projected;
            apply_sparse_uniform_shift(candidate, shift_frames);
            const EdgeOffsetMetrics cand_intro =
                measure_edge_offsets(candidate, bpm_hint, false, sample_rate, provider);
            const EdgeOffsetMetrics cand_outro =
                measure_edge_offsets(candidate, bpm_hint, true, sample_rate, provider);
            if (has_valid_metrics(cand_intro) && has_valid_metrics(cand_outro)) {
                const double base_worst =
                    std::max(std::abs(global_intro.median_ms), std::abs(global_outro.median_ms));
                const double cand_worst =
                    std::max(std::abs(cand_intro.median_ms), std::abs(cand_outro.median_ms));
                if (cand_worst + 5.0 < base_worst) {
                    *projected = std::move(candidate);
                    global_intro = cand_intro;
                    global_outro = cand_outro;
                }
            }
        }
    }

    const SparseEdgePhaseTryResult phase_try = apply_sparse_edge_phase_try(
        projected,
        bpm_hint,
        provider,
        sample_rate,
        probe_duration,
        params.between_probe_start,
        params.middle_probe_start,
        first_window_start,
        last_window_start);

    const EdgeOffsetMetrics pre_global_intro =
        measure_edge_offsets(projected_before_refit, bpm_hint, false, sample_rate, provider);
    const EdgeOffsetMetrics pre_global_outro =
        measure_edge_offsets(projected_before_refit, bpm_hint, true, sample_rate, provider);
    const auto pre_middle_pair = measure_middle_windows(projected_before_refit);
    const auto post_middle_pair = measure_middle_windows(*projected);
    const double pre_global_delta_ms =
        (has_valid_metrics(pre_global_intro) && has_valid_metrics(pre_global_outro))
            ? std::abs(pre_global_outro.median_ms - pre_global_intro.median_ms)
            : std::numeric_limits<double>::infinity();
    const double post_global_delta_ms =
        (has_valid_metrics(global_intro) && has_valid_metrics(global_outro))
            ? std::abs(global_outro.median_ms - global_intro.median_ms)
            : std::numeric_limits<double>::infinity();
    const double pre_between_abs_ms = pre_middle_pair.first.median_abs_ms;
    const double pre_middle_abs_ms = pre_middle_pair.second.median_abs_ms;
    const double post_between_abs_ms = post_middle_pair.first.median_abs_ms;
    const double post_middle_abs_ms = post_middle_pair.second.median_abs_ms;
    const double pre_phase_score =
        (std::isfinite(pre_global_delta_ms) &&
         std::isfinite(pre_between_abs_ms) &&
         std::isfinite(pre_middle_abs_ms))
            ? ((0.40 * pre_global_delta_ms) + pre_between_abs_ms + pre_middle_abs_ms)
            : std::numeric_limits<double>::infinity();
    const double post_phase_score =
        (std::isfinite(post_global_delta_ms) &&
         std::isfinite(post_between_abs_ms) &&
         std::isfinite(post_middle_abs_ms))
            ? ((0.40 * post_global_delta_ms) + post_between_abs_ms + post_middle_abs_ms)
            : std::numeric_limits<double>::infinity();

    const double err_delta_frames =
        ((outro.median_ms - intro.median_ms) * sample_rate) / 1000.0;
    BEATIT_LOG_DEBUG("Sparse edge refit:"
                     << " second_pass=" << (second_pass_enabled ? 1 : 0)
                     << " first_probe_start_s=" << first_probe_start
                     << " last_probe_start_s=" << last_probe_start
                     << " first_window_start_s=" << first_window_start
                     << " last_window_start_s=" << last_window_start
                     << " quality_shift_rounds=" << quality_shift_rounds
                     << " intro_ms=" << intro.median_ms
                     << " outro_ms=" << outro.median_ms
                     << " post_intro_ms=" << post_intro.median_ms
                     << " post_outro_ms=" << post_outro.median_ms
                     << " global_intro_ms=" << global_intro.median_ms
                     << " global_outro_ms=" << global_outro.median_ms
                     << " pre_global_delta_ms=" << pre_global_delta_ms
                     << " post_global_delta_ms=" << post_global_delta_ms
                     << " pre_between_abs_ms=" << pre_between_abs_ms
                     << " pre_middle_abs_ms=" << pre_middle_abs_ms
                     << " post_between_abs_ms=" << post_between_abs_ms
                     << " post_middle_abs_ms=" << post_middle_abs_ms
                     << " pre_phase_score=" << pre_phase_score
                     << " post_phase_score=" << post_phase_score
                     << " phase_try_base_score=" << phase_try.base_score
                     << " phase_try_minus_score=" << phase_try.minus_score
                     << " phase_try_plus_score=" << phase_try.plus_score
                     << " phase_try_selected=" << phase_try.selected
                     << " phase_try_applied=" << (phase_try.applied ? 1 : 0)
                     << " global_ratio_applied=" << global_guard_ratio
                     << " delta_frames=" << err_delta_frames
                     << " ratio=" << ratio
                     << " ratio_applied=" << applied_ratio
                     << " beats=" << projected->size());
}

} // namespace detail
} // namespace beatit
