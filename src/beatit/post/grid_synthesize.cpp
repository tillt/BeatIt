//
//  grid_synthesize.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "grid_projection.h"

#include "beatit/post/helpers.h"
#include "beatit/post/result_ops.h"
#include "beatit/post/window.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace beatit::detail {

namespace {

struct OffsetSample {
    std::size_t beat_index = 0;
    double offset = 0.0;
};

double median_offset_window(const std::vector<OffsetSample>& samples,
                            std::size_t begin,
                            std::size_t count) {
    std::vector<double> values;
    values.reserve(count);
    const std::size_t end = std::min(samples.size(), begin + count);
    for (std::size_t i = begin; i < end; ++i) {
        values.push_back(samples[i].offset);
    }
    if (values.empty()) {
        return 0.0;
    }
    auto mid = values.begin() + static_cast<long>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

double grid_phase_score(const std::vector<float>& activation,
                        double start_frame,
                        double step_frames,
                        std::size_t used_frames) {
    if (start_frame < 0.0 || step_frames <= 0.0) {
        return -1.0;
    }

    double score = 0.0;
    std::size_t hits = 0;
    double cursor = start_frame;
    while (cursor >= step_frames) {
        cursor -= step_frames;
    }
    while (cursor < static_cast<double>(used_frames) && hits < 128) {
        const long long idx_ll = static_cast<long long>(std::llround(cursor));
        if (idx_ll >= 0 && static_cast<std::size_t>(idx_ll) < activation.size()) {
            const std::size_t idx = static_cast<std::size_t>(idx_ll);
            float value = activation[idx];
            if (idx > 0) {
                value = std::max(value, activation[idx - 1]);
            }
            if (idx + 1 < activation.size()) {
                value = std::max(value, activation[idx + 1]);
            }
            score += static_cast<double>(value);
            hits += 1;
        }
        cursor += step_frames;
    }
    return hits > 0 ? (score / static_cast<double>(hits)) : -1.0;
}

} // namespace

void synthesize_uniform_grid(GridProjectionState& state,
                             DBNDecodeResult& decoded,
                             const CoreMLResult& result,
                             const BeatitConfig& config,
                             std::size_t used_frames,
                             double fps) {
    if (decoded.beat_frames.size() < 2 || state.step_frames <= 1.0) {
        return;
    }

    if (config.dbn_grid_global_fit && decoded.beat_frames.size() >= 8) {
        const double n = static_cast<double>(decoded.beat_frames.size());
        double sx = 0.0;
        double sy = 0.0;
        double sxx = 0.0;
        double sxy = 0.0;
        for (std::size_t i = 0; i < decoded.beat_frames.size(); ++i) {
            const double x = static_cast<double>(i);
            const double y = static_cast<double>(decoded.beat_frames[i]);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        const double den = n * sxx - sx * sx;
        if (std::abs(den) > 1e-9) {
            const double fit_step = (n * sxy - sx * sy) / den;
            if (fit_step > 1.0) {
                BEATIT_LOG_DEBUG("DBN grid fit: step_frames(raw)=" << state.step_frames
                                 << " step_frames(fit)=" << fit_step
                                 << " beats=" << decoded.beat_frames.size());
                state.step_frames = fit_step;
            }
        }
    }

    const bool have_downbeat_start = !decoded.downbeat_frames.empty();
    const bool reliable_downbeat_start =
        have_downbeat_start && state.max_downbeat > state.activation_floor;
    const std::size_t start = reliable_downbeat_start
        ? decoded.downbeat_frames.front()
        : std::min(decoded.beat_frames.front(),
                   std::min(state.earliest_peak, state.earliest_downbeat_peak));
    double grid_start = static_cast<double>(start);
    if (state.earliest_downbeat_peak > 0 && state.earliest_downbeat_peak < start) {
        grid_start = static_cast<double>(state.earliest_downbeat_peak);
    }
    if (!reliable_downbeat_start && config.dbn_grid_start_strong_peak &&
        state.strongest_peak_value >= state.activation_floor) {
        grid_start = static_cast<double>(state.strongest_peak);
    }
    if (!reliable_downbeat_start && config.dbn_grid_align_downbeat_peak &&
        state.earliest_downbeat_peak > 0 && state.step_frames > 1.0) {
        const double offset = static_cast<double>(state.earliest_downbeat_peak) - grid_start;
        const double half_step = state.step_frames * 0.5;
        if (std::abs(offset) <= half_step) {
            const double adjusted = grid_start + offset;
            if (adjusted >= 0.0) {
                grid_start = adjusted;
            }
        } else if (state.earliest_downbeat_peak < grid_start) {
            // If the first downbeat peak is clearly earlier, bias the grid to it.
            grid_start = static_cast<double>(state.earliest_downbeat_peak);
        }
    }
    if (config.dbn_grid_start_advance_seconds > 0.0f && fps > 0.0) {
        const double frames_per_second = fps;
        grid_start -=
            static_cast<double>(config.dbn_grid_start_advance_seconds) * frames_per_second;
    }
    if (!reliable_downbeat_start && state.step_frames > 1.0 &&
        result.beat_activation.size() >= 8) {
        const double alt_start = grid_start + (0.5 * state.step_frames);
        const double base_score =
            grid_phase_score(result.beat_activation, grid_start, state.step_frames, used_frames);
        const double alt_score =
            grid_phase_score(result.beat_activation, alt_start, state.step_frames, used_frames);
        if (alt_score > base_score) {
            grid_start = alt_start;
            BEATIT_LOG_DEBUG("DBN grid: half-step phase shift selected"
                             << " base_score=" << base_score
                             << " alt_score=" << alt_score);
        }
    }
    if (grid_start < 0.0) {
        grid_start = 0.0;
    }
    if (state.step_frames > 1.0 && result.beat_activation.size() >= 64) {
        std::vector<std::size_t> beat_peaks =
            collect_activation_peaks(result.beat_activation, state.activation_floor);
        if (beat_peaks.size() >= 16) {
            std::vector<OffsetSample> samples;
            samples.reserve(256);
            double cursor_fit = grid_start;
            std::size_t beat_index = 0;
            while (cursor_fit < static_cast<double>(used_frames) && beat_index < 512) {
                const long long frame_ll = static_cast<long long>(std::llround(cursor_fit));
                if (frame_ll >= 0) {
                    const std::size_t frame = static_cast<std::size_t>(frame_ll);
                    const std::size_t nearest = nearest_peak_frame(beat_peaks, frame);
                    const double max_dist = state.step_frames * 0.45;
                    const double dist =
                        std::abs(static_cast<double>(nearest) - static_cast<double>(frame));
                    if (dist <= max_dist) {
                        OffsetSample sample;
                        sample.beat_index = beat_index;
                        sample.offset = static_cast<double>(nearest) - static_cast<double>(frame);
                        samples.push_back(sample);
                    }
                }
                cursor_fit += state.step_frames;
                beat_index += 1;
            }
            if (samples.size() >= 32) {
                const std::size_t edge = std::min<std::size_t>(32, samples.size() / 2);
                const double start_offset = median_offset_window(samples, 0, edge);
                const std::size_t tail_begin = samples.size() - edge;
                const double end_offset = median_offset_window(samples, tail_begin, edge);
                const std::size_t start_index = samples.front().beat_index;
                const std::size_t end_index = samples.back().beat_index;
                const double index_delta = static_cast<double>(end_index - start_index);
                if (index_delta > 0.0) {
                    const double step_correction = (end_offset - start_offset) / index_delta;
                    const double max_step_correction = state.step_frames * 0.01;
                    const double clamped_correction = std::clamp(step_correction,
                                                                 -max_step_correction,
                                                                 max_step_correction);
                    if (std::abs(clamped_correction) > 1e-6) {
                        state.step_frames += clamped_correction;
                        grid_start += start_offset;
                        if (grid_start < 0.0) {
                            grid_start = 0.0;
                        }
                        BEATIT_LOG_DEBUG("DBN grid drift-correct:"
                                         << " start_offset=" << start_offset
                                         << " end_offset=" << end_offset
                                         << " start_index=" << start_index
                                         << " end_index=" << end_index
                                         << " step_correction=" << step_correction
                                         << " step_applied=" << clamped_correction
                                         << " step_frames=" << state.step_frames
                                         << " samples=" << samples.size());
                    }
                }
            }
        }
    }
    std::vector<std::size_t> uniform_beats;
    double cursor = grid_start;
    while (cursor >= state.step_frames) {
        cursor -= state.step_frames;
        uniform_beats.push_back(static_cast<std::size_t>(std::llround(cursor)));
    }
    std::reverse(uniform_beats.begin(), uniform_beats.end());
    cursor = grid_start;
    while (cursor < static_cast<double>(used_frames)) {
        uniform_beats.push_back(static_cast<std::size_t>(std::llround(cursor)));
        cursor += state.step_frames;
    }
    detail::dedupe_frames(uniform_beats);
    decoded.beat_frames = std::move(uniform_beats);
    decoded.downbeat_frames =
        project_downbeats_from_beats(decoded.beat_frames, state.bpb, state.best_phase);
    if (config.dbn_trace && beatit_should_log("debug") && fps > 0.0) {
        trace_grid_peak_alignment(decoded.beat_frames,
                                  decoded.downbeat_frames,
                                  result.beat_activation,
                                  result.downbeat_activation,
                                  state.activation_floor,
                                  fps);
    }
    BEATIT_LOG_DEBUG("DBN grid: start=" << start
                     << " grid_start=" << grid_start
                     << " strongest_peak=" << state.strongest_peak
                     << " strongest_peak_value=" << state.strongest_peak_value
                     << " earliest_downbeat_peak=" << state.earliest_downbeat_peak
                     << " advance_s=" << config.dbn_grid_start_advance_seconds);
}

} // namespace beatit::detail
