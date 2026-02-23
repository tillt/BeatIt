//
//  dbn_grid.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "dbn_grid.h"

#include "beatit/post/helpers.h"
#include "beatit/post/result_ops.h"
#include "beatit/post/window.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace beatit::detail {

void select_downbeat_phase(GridProjectionState& state,
                           DBNDecodeResult& decoded,
                           const CoreMLResult& result,
                           const CoreMLConfig& config,
                           bool quality_low,
                           bool downbeat_override_ok,
                           bool use_window,
                           std::size_t window_start,
                           std::size_t used_frames,
                           double fps) {
    const double local_frame_rate =
        config.hop_size > 0
            ? static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size)
            : 0.0;
    const std::size_t phase_window_frames =
        (config.dbn_downbeat_phase_window_seconds > 0.0 && local_frame_rate > 0.0)
            ? static_cast<std::size_t>(
                  std::llround(config.dbn_downbeat_phase_window_seconds * local_frame_rate))
            : 0;
    const std::size_t phase_window_start = use_window ? window_start : 0;
    const std::size_t phase_window_end =
        (phase_window_frames > 0) ? std::min(used_frames, phase_window_start + phase_window_frames)
                                  : phase_window_start;
    const bool allow_downbeat_phase =
        !result.downbeat_activation.empty() && !quality_low && downbeat_override_ok;
    const std::size_t max_delay_frames =
        (config.dbn_downbeat_phase_max_delay_seconds > 0.0 && local_frame_rate > 0.0)
            ? static_cast<std::size_t>(
                  std::llround(config.dbn_downbeat_phase_max_delay_seconds * local_frame_rate))
            : 0;
    const float onset_ratio = 0.35f;
    const std::size_t onset_max_back = max_delay_frames > 0 ? max_delay_frames : 8;
    auto onset_from_peak = [&](const std::vector<float>& activation,
                               std::size_t peak_frame) -> std::size_t {
        if (activation.empty() || peak_frame >= activation.size()) {
            return peak_frame;
        }
        const float peak_value = activation[peak_frame];
        if (peak_value <= 0.0f) {
            return peak_frame;
        }
        const float threshold = peak_value * onset_ratio;
        std::size_t frame = peak_frame;
        std::size_t steps = 0;
        while (frame > 0 && steps < onset_max_back) {
            if (activation[frame] < threshold) {
                return frame + 1;
            }
            --frame;
            ++steps;
        }
        return frame;
    };
    auto build_onset_frames = [&](const std::vector<std::size_t>& frames,
                                  const std::vector<float>& activation) {
        std::vector<std::size_t> out;
        out.reserve(frames.size());
        for (std::size_t frame : frames) {
            out.push_back(onset_from_peak(activation, frame));
        }
        detail::dedupe_frames(out);
        return out;
    };

    float max_beat = 0.0f;
    std::vector<uint8_t> phase_peak_mask;
    bool has_phase_peaks = false;
    if (phase_window_frames > 0) {
        for (std::size_t i = phase_window_start; i < phase_window_end; ++i) {
            if (allow_downbeat_phase) {
                state.max_downbeat = std::max(state.max_downbeat, result.downbeat_activation[i]);
            }
            if (i < result.beat_activation.size()) {
                max_beat = std::max(max_beat, result.beat_activation[i]);
            }
        }
        phase_peak_mask.assign(used_frames, 0);
        if (max_beat > 0.0f && phase_window_end > phase_window_start + 2) {
            const float beat_threshold =
                static_cast<float>(max_beat * config.dbn_downbeat_phase_peak_ratio);
            const float peak_eps = std::max(1e-6f, max_beat * 0.01f);
            for (std::size_t i = phase_window_start + 1; i + 1 < phase_window_end; ++i) {
                const float prev = result.beat_activation[i - 1];
                const float curr = result.beat_activation[i];
                const float next = result.beat_activation[i + 1];
                if (curr >= beat_threshold && curr >= prev + peak_eps && curr >= next + peak_eps) {
                    const std::size_t onset_frame = onset_from_peak(result.beat_activation, i);
                    if (onset_frame < phase_peak_mask.size()) {
                        phase_peak_mask[onset_frame] = 1;
                    }
                    has_phase_peaks = true;
                }
            }
        }
        if (config.dbn_trace && allow_downbeat_phase) {
            const std::size_t limit =
                phase_window_end > phase_window_start ? (phase_window_end - phase_window_start) : 0;
            const std::size_t beat_preview = std::min<std::size_t>(12, result.beat_activation.size());
            std::cerr << "DBN: beat head:";
            for (std::size_t i = 0; i < beat_preview; ++i) {
                std::cerr << " " << i << "->" << result.beat_activation[i];
            }
            std::cerr << "\n";
            const std::size_t preview = std::min<std::size_t>(12, result.downbeat_activation.size());
            std::cerr << "DBN: downbeat head:";
            for (std::size_t i = 0; i < preview; ++i) {
                std::cerr << " " << i << "->" << result.downbeat_activation[i];
            }
            std::cerr << "\n";
            std::cerr << "DBN: downbeat max=" << state.max_downbeat
                      << " beat max=" << max_beat
                      << " activation_floor=" << state.activation_floor
                      << "\n";
            struct Peak {
                float value;
                std::size_t frame;
            };
            std::vector<Peak> peaks;
            const std::size_t start = (phase_window_start + 1 < phase_window_end)
                ? (phase_window_start + 1)
                : phase_window_start;
            const std::size_t end =
                phase_window_end > 1 ? phase_window_end - 1 : phase_window_end;
            for (std::size_t i = start; i <= end; ++i) {
                const float prev = result.downbeat_activation[i - 1];
                const float curr = result.downbeat_activation[i];
                const float next = result.downbeat_activation[i + 1];
                if (curr >= prev && curr >= next) {
                    peaks.push_back({curr, i});
                }
            }
            std::sort(peaks.begin(),
                      peaks.end(),
                      [](const Peak& a, const Peak& b) { return a.value > b.value; });
            std::cerr << "DBN: downbeat peaks (phase window "
                      << phase_window_start << "-" << phase_window_end << "):";
            const std::size_t top = std::min<std::size_t>(5, peaks.size());
            for (std::size_t i = 0; i < top; ++i) {
                std::cerr << " " << peaks[i].frame << "->" << peaks[i].value;
            }
            std::cerr << "\n";
            std::vector<Peak> global_peaks;
            const std::size_t global_end =
                result.downbeat_activation.empty() ? 0 : used_frames > 1 ? used_frames - 1 : 0;
            for (std::size_t i = start; i <= global_end; ++i) {
                const float prev = result.downbeat_activation[i - 1];
                const float curr = result.downbeat_activation[i];
                const float next = result.downbeat_activation[i + 1];
                if (curr >= prev && curr >= next) {
                    global_peaks.push_back({curr, i});
                }
            }
            std::sort(global_peaks.begin(),
                      global_peaks.end(),
                      [](const Peak& a, const Peak& b) { return a.value > b.value; });
            std::cerr << "DBN: downbeat peaks (global top):";
            const std::size_t global_top = std::min<std::size_t>(6, global_peaks.size());
            for (std::size_t i = 0; i < global_top; ++i) {
                const std::size_t frame = global_peaks[i].frame;
                const double time_s = fps > 0.0 ? static_cast<double>(frame) / fps : 0.0;
                std::cerr << " " << frame << "(" << time_s << "s)"
                          << "->" << global_peaks[i].value;
            }
            std::cerr << "\n";
            std::cerr << "DBN: phase peaks for selection (beat-only, strict): "
                      << (has_phase_peaks ? "picked" : "none")
                      << " (limit=" << limit << ")\n";
        }
    }
    const float downbeat_threshold =
        state.max_downbeat > 0.0f
            ? static_cast<float>(state.max_downbeat * config.dbn_downbeat_phase_peak_ratio)
            : 0.0f;
    struct PhaseDebug {
        std::size_t phase = 0;
        double score = -std::numeric_limits<double>::infinity();
        std::size_t first_frame = 0;
        std::size_t hits = 0;
        double mean = 0.0;
        double delay_penalty = 0.0;
        const char* source = "none";
    };
    std::vector<PhaseDebug> phase_debug;
    if (config.dbn_trace) {
        phase_debug.reserve(state.bpb);
    }
    for (std::size_t candidate_phase = 0; candidate_phase < state.bpb; ++candidate_phase) {
        const auto projected =
            project_downbeats_from_beats(decoded.beat_frames, state.bpb, candidate_phase);
        if (projected.empty()) {
            continue;
        }
        const std::vector<float>& onset_activation =
            allow_downbeat_phase ? result.downbeat_activation : result.beat_activation;
        const auto projected_onsets = build_onset_frames(projected, onset_activation);
        const auto& phase_frames = projected_onsets.empty() ? projected : projected_onsets;
        double score = -std::numeric_limits<double>::infinity();
        std::size_t hits = 0;
        double mean = 0.0;
        const char* source = "none";

        if (phase_window_frames > 0 && !allow_downbeat_phase) {
            if (has_phase_peaks) {
                double sum = 0.0;
                double weight = 0.0;
                for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                    const std::size_t frame = phase_frames[i];
                    if (frame >= phase_window_frames) {
                        break;
                    }
                    if (frame < phase_peak_mask.size() && phase_peak_mask[frame]) {
                        const double value = (frame < result.beat_activation.size())
                            ? static_cast<double>(result.beat_activation[frame])
                            : 0.0;
                        sum += value;
                        weight += 1.0;
                    }
                }
                if (weight > 0.0) {
                    score = (sum / weight);
                    hits = static_cast<std::size_t>(weight);
                    mean = score;
                    source = "beat_peak_mask";
                }
            } else {
                double sum = 0.0;
                double weight = 0.0;
                const float threshold = max_beat > 0.0f
                    ? static_cast<float>(max_beat * config.dbn_downbeat_phase_peak_ratio)
                    : 0.0f;
                for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                    const std::size_t frame = phase_frames[i];
                    if (frame >= phase_window_frames) {
                        break;
                    }
                    if (frame < result.beat_activation.size()) {
                        const float value = result.beat_activation[frame];
                        if (value >= threshold) {
                            sum += static_cast<double>(value);
                            weight += 1.0;
                        }
                    }
                }
                if (weight > 0.0) {
                    score = (sum / weight);
                    hits = static_cast<std::size_t>(weight);
                    mean = score;
                    source = "beat_threshold";
                }
            }
        } else if (phase_window_frames > 0 && allow_downbeat_phase) {
            double sum = 0.0;
            double weight = 0.0;
            const float threshold = state.max_downbeat > 0.0f
                ? static_cast<float>(state.max_downbeat * config.dbn_downbeat_phase_peak_ratio)
                : 0.0f;
            const std::size_t end = std::min(phase_window_end, result.downbeat_activation.size());
            const double decay = std::max(1.0, phase_window_frames * 0.2);
            for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                const std::size_t frame = phase_frames[i];
                if (frame >= end) {
                    break;
                }
                const float value = result.downbeat_activation[frame];
                if (value >= threshold) {
                    const double w = std::exp(-static_cast<double>(frame) / decay);
                    sum += static_cast<double>(value) * w;
                    weight += w;
                }
            }
            if (weight > 0.0) {
                score = (sum / weight);
                hits = static_cast<std::size_t>(weight + 0.5);
                mean = score;
                source = "downbeat_window_decay";
            }
        }
        if (!std::isfinite(score)) {
            const std::size_t max_checks = std::min<std::size_t>(phase_frames.size(), 3);
            double sum = 0.0;
            if (allow_downbeat_phase) {
                const float threshold = state.max_downbeat > 0.0f
                    ? static_cast<float>(state.max_downbeat * config.dbn_downbeat_phase_peak_ratio)
                    : 0.0f;
                for (std::size_t i = 0; i < max_checks; ++i) {
                    const std::size_t frame = phase_frames[i];
                    if (frame < result.downbeat_activation.size()) {
                        const float value = result.downbeat_activation[frame];
                        if (value >= threshold) {
                            sum += value;
                            hits += 1;
                        }
                    }
                }
                if (hits == 0) {
                    for (std::size_t i = 0; i < max_checks; ++i) {
                        const std::size_t frame = phase_frames[i];
                        if (frame < result.downbeat_activation.size()) {
                            sum += result.downbeat_activation[frame];
                        }
                    }
                    hits = max_checks;
                }
            } else {
                for (std::size_t i = 0; i < max_checks; ++i) {
                    const std::size_t frame = phase_frames[i];
                    if (frame < result.beat_activation.size()) {
                        sum += result.beat_activation[frame];
                    }
                }
                hits = max_checks;
            }
            const double penalty = 0.01 * static_cast<double>(phase_frames.front());
            score = sum - penalty;
            mean = (max_checks > 0) ? (sum / static_cast<double>(max_checks)) : 0.0;
            source = allow_downbeat_phase ? "fallback_downbeat" : "fallback_beat";
        }
        double delay_penalty = 0.0;
        if (max_delay_frames > 0 && phase_frames.front() > max_delay_frames &&
            std::strncmp(source, "fallback", 8) != 0 &&
            !(quality_low &&
              (std::strncmp(source, "beat_peak_mask", 14) == 0 ||
               std::strncmp(source, "beat_threshold", 14) == 0))) {
            const double delay = static_cast<double>(phase_frames.front() - max_delay_frames);
            delay_penalty = delay * 1000.0;
            score -= delay_penalty;
        }
        if (score > state.best_score) {
            state.best_score = score;
            state.best_phase = candidate_phase;
        }
        if (config.dbn_trace) {
            PhaseDebug entry;
            entry.phase = candidate_phase;
            entry.score = score;
            entry.first_frame = phase_frames.front();
            entry.hits = hits;
            entry.mean = mean;
            entry.delay_penalty = delay_penalty;
            entry.source = source;
            phase_debug.push_back(entry);
        }
    }
    if (config.verbose) {
        std::cerr << "DBN: phase_window_frames=" << phase_window_frames
                  << " max_downbeat=" << state.max_downbeat
                  << " threshold=" << downbeat_threshold
                  << " best_phase=" << state.best_phase
                  << " best_score=" << state.best_score
                  << "\n";
    }
    if (config.dbn_trace && !phase_debug.empty()) {
        std::vector<PhaseDebug> sorted = phase_debug;
        std::sort(sorted.begin(),
                  sorted.end(),
                  [](const PhaseDebug& a, const PhaseDebug& b) { return a.score > b.score; });
        const PhaseDebug& top = sorted.front();
        const PhaseDebug* runner = (sorted.size() > 1) ? &sorted[1] : nullptr;
        std::cerr << "DBN: phase winner="
                  << top.phase
                  << " score=" << top.score
                  << " first=" << top.first_frame
                  << " hits=" << top.hits
                  << " mean=" << top.mean
                  << " penalty=" << top.delay_penalty
                  << " src=" << top.source;
        if (runner) {
            std::cerr << " runner=" << runner->phase
                      << " score=" << runner->score
                      << " first=" << runner->first_frame
                      << " hits=" << runner->hits
                      << " mean=" << runner->mean
                      << " penalty=" << runner->delay_penalty
                      << " src=" << runner->source;
        }
        std::cerr << "\n";
        std::cerr << "DBN: phase candidates:";
        for (const auto& entry : phase_debug) {
            std::cerr << " p" << entry.phase
                      << " score=" << entry.score
                      << " first=" << entry.first_frame
                      << " hits=" << entry.hits
                      << " mean=" << entry.mean
                      << " penalty=" << entry.delay_penalty
                      << " src=" << entry.source;
        }
        std::cerr << "\n";
    }
    decoded.downbeat_frames =
        project_downbeats_from_beats(decoded.beat_frames, state.bpb, state.best_phase);
    if (config.dbn_trace) {
        const std::size_t preview = std::min<std::size_t>(6, decoded.downbeat_frames.size());
        std::cerr << "DBN: downbeat frames head:";
        for (std::size_t i = 0; i < preview; ++i) {
            const std::size_t frame = decoded.downbeat_frames[i];
            const double time_s = fps > 0.0 ? static_cast<double>(frame) / fps : 0.0;
            std::cerr << " " << frame << "(" << time_s << "s)";
        }
        std::cerr << "\n";
        if (!decoded.downbeat_frames.empty()) {
            const std::size_t first_frame = decoded.downbeat_frames.front();
            const double first_time = fps > 0.0 ? static_cast<double>(first_frame) / fps : 0.0;
            std::cerr << "DBN: downbeat selection start="
                      << first_frame << " (" << first_time << "s)"
                      << " bpb=" << state.bpb
                      << " phase=" << state.best_phase
                      << " score=" << state.best_score
                      << "\n";
        }
    }
}

void synthesize_uniform_grid(GridProjectionState& state,
                             DBNDecodeResult& decoded,
                             const CoreMLResult& result,
                             const CoreMLConfig& config,
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
                if (config.verbose) {
                    std::cerr << "DBN grid fit: step_frames(raw)=" << state.step_frames
                              << " step_frames(fit)=" << fit_step
                              << " beats=" << decoded.beat_frames.size()
                              << "\n";
                }
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
        const auto phase_score = [&](double start_frame) -> double {
            if (start_frame < 0.0) {
                return -1.0;
            }
            double score = 0.0;
            std::size_t hits = 0;
            double cursor_local = start_frame;
            while (cursor_local >= state.step_frames) {
                cursor_local -= state.step_frames;
            }
            while (cursor_local < static_cast<double>(used_frames) && hits < 128) {
                const long long idx_ll = static_cast<long long>(std::llround(cursor_local));
                if (idx_ll >= 0 &&
                    static_cast<std::size_t>(idx_ll) < result.beat_activation.size()) {
                    const std::size_t idx = static_cast<std::size_t>(idx_ll);
                    float value = result.beat_activation[idx];
                    if (idx > 0) {
                        value = std::max(value, result.beat_activation[idx - 1]);
                    }
                    if (idx + 1 < result.beat_activation.size()) {
                        value = std::max(value, result.beat_activation[idx + 1]);
                    }
                    score += static_cast<double>(value);
                    hits += 1;
                }
                cursor_local += state.step_frames;
            }
            return hits > 0 ? (score / static_cast<double>(hits)) : -1.0;
        };
        const double alt_start = grid_start + (0.5 * state.step_frames);
        const double base_score = phase_score(grid_start);
        const double alt_score = phase_score(alt_start);
        if (alt_score > base_score) {
            grid_start = alt_start;
            if (config.verbose) {
                std::cerr << "DBN grid: half-step phase shift selected"
                          << " base_score=" << base_score
                          << " alt_score=" << alt_score
                          << "\n";
            }
        }
    }
    if (grid_start < 0.0) {
        grid_start = 0.0;
    }
    if (state.step_frames > 1.0 && result.beat_activation.size() >= 64) {
        std::vector<std::size_t> beat_peaks;
        beat_peaks.reserve(result.beat_activation.size() / 8);
        if (result.beat_activation.size() >= 3) {
            const std::size_t end = result.beat_activation.size() - 1;
            for (std::size_t i = 1; i < end; ++i) {
                const float prev = result.beat_activation[i - 1];
                const float curr = result.beat_activation[i];
                const float next = result.beat_activation[i + 1];
                if (curr >= state.activation_floor && curr >= prev && curr >= next) {
                    beat_peaks.push_back(i);
                }
            }
        }
        if (beat_peaks.size() >= 16) {
            struct OffsetSample {
                std::size_t beat_index = 0;
                double offset = 0.0;
            };
            std::vector<OffsetSample> samples;
            samples.reserve(256);
            double cursor_fit = grid_start;
            std::size_t beat_index = 0;
            while (cursor_fit < static_cast<double>(used_frames) && beat_index < 512) {
                const long long frame_ll = static_cast<long long>(std::llround(cursor_fit));
                if (frame_ll >= 0) {
                    const std::size_t frame = static_cast<std::size_t>(frame_ll);
                    auto it = std::lower_bound(beat_peaks.begin(), beat_peaks.end(), frame);
                    std::size_t nearest = beat_peaks.front();
                    if (it == beat_peaks.end()) {
                        nearest = beat_peaks.back();
                    } else {
                        nearest = *it;
                        if (it != beat_peaks.begin()) {
                            const std::size_t prev = *(it - 1);
                            if (frame - prev < nearest - frame) {
                                nearest = prev;
                            }
                        }
                    }
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
                auto median_of_offsets =
                    [](const std::vector<OffsetSample>& src, std::size_t begin, std::size_t count) {
                        std::vector<double> values;
                        values.reserve(count);
                        const std::size_t end = std::min(src.size(), begin + count);
                        for (std::size_t i = begin; i < end; ++i) {
                            values.push_back(src[i].offset);
                        }
                        if (values.empty()) {
                            return 0.0;
                        }
                        auto mid = values.begin() + static_cast<long>(values.size() / 2);
                        std::nth_element(values.begin(), mid, values.end());
                        return *mid;
                    };
                const std::size_t edge = std::min<std::size_t>(32, samples.size() / 2);
                const double start_offset = median_of_offsets(samples, 0, edge);
                const std::size_t tail_begin = samples.size() - edge;
                const double end_offset = median_of_offsets(samples, tail_begin, edge);
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
                        if (config.verbose) {
                            std::cerr << "DBN grid drift-correct:"
                                      << " start_offset=" << start_offset
                                      << " end_offset=" << end_offset
                                      << " start_index=" << start_index
                                      << " end_index=" << end_index
                                      << " step_correction=" << step_correction
                                      << " step_applied=" << clamped_correction
                                      << " step_frames=" << state.step_frames
                                      << " samples=" << samples.size()
                                      << "\n";
                        }
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
    if (config.dbn_trace && fps > 0.0) {
        trace_grid_peak_alignment(decoded.beat_frames,
                                  decoded.downbeat_frames,
                                  result.beat_activation,
                                  result.downbeat_activation,
                                  state.activation_floor,
                                  fps);
    }
    if (config.verbose) {
        std::cerr << "DBN grid: start=" << start
                  << " grid_start=" << grid_start
                  << " strongest_peak=" << state.strongest_peak
                  << " strongest_peak_value=" << state.strongest_peak_value
                  << " earliest_downbeat_peak=" << state.earliest_downbeat_peak
                  << " advance_s=" << config.dbn_grid_start_advance_seconds
                  << "\n";
    }
}

} // namespace beatit::detail
