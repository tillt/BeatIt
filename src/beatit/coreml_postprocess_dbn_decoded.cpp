//
//  coreml_postprocess_dbn_decoded.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/coreml_postprocess_dbn_decoded.h"

#include "beatit/coreml_postprocess_helpers.h"
#include "coreml_postprocess_dbn_grid_stages.h"
#include "beatit/coreml_postprocess_result_ops.h"
#include "beatit/coreml_postprocess_tempo_fit.h"
#include "beatit/coreml_postprocess_window.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
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

struct GridTempoDecision {
    struct Diagnostics {
        double min_interval_frames = 0.0;
        double short_interval_threshold = 0.0;

        IntervalStats tempo_stats;
        IntervalStats decoded_stats;
        IntervalStats decoded_filtered_stats;
        IntervalStats downbeat_stats;

        bool has_downbeat_stats = false;
        bool stats_computed = false;

        double bpm_from_peaks = 0.0;
        double bpm_from_peaks_median = 0.0;
        double bpm_from_peaks_reg = 0.0;
        double bpm_from_peaks_median_full = 0.0;
        double bpm_from_peaks_reg_full = 0.0;
        double bpm_from_downbeats = 0.0;
        double bpm_from_downbeats_median = 0.0;
        double bpm_from_downbeats_reg = 0.0;
        double bpm_from_fit = 0.0;
        double bpm_from_global_fit = 0.0;

        std::size_t downbeat_count = 0;
        double downbeat_cv = 0.0;

        bool drop_fit = false;
        bool drop_ref = false;
        bool global_fit_plausible = false;

        double bpm_before_downbeat = 0.0;
        std::string bpm_source = "none";
        std::string bpm_source_before_downbeat = "none";

        double quality_qpar = 0.0;
        double quality_qkur = 0.0;
    };

    std::size_t bpb = 1;
    std::size_t phase = 0;
    double base_interval = 0.0;
    double bpm_for_grid = 0.0;
    double step_frames = 0.0;
    bool quality_low = false;
    bool downbeat_override_ok = false;
    Diagnostics diagnostics;
};

void log_grid_tempo_decision(const GridTempoDecision& decision,
                             const DBNDecodeResult& decoded,
                             const CoreMLConfig& config,
                             float reference_bpm,
                             double fps) {
    const auto& d = decision.diagnostics;
    if (config.dbn_trace && d.stats_computed) {
        auto print_stats = [&](const char* label, const IntervalStats& stats) {
            if (stats.count == 0 || stats.median_interval <= 0.0) {
                std::cerr << "DBN stats: " << label << " empty\n";
                return;
            }
            const double bpm_median = (60.0 * fps) / stats.median_interval;
            const double bpm_mean = (60.0 * fps) / stats.mean_interval;
            const double interval_cv =
                stats.mean_interval > 0.0 ? (stats.stdev_interval / stats.mean_interval) : 0.0;
            std::cerr << "DBN stats: " << label
                      << " count=" << stats.count
                      << " bpm_median=" << bpm_median
                      << " bpm_mean=" << bpm_mean
                      << " interval_cv=" << interval_cv
                      << " interval_range=[" << stats.min_interval
                      << "," << stats.max_interval << "]";
            if (!stats.top_bpm_bins.empty()) {
                std::cerr << " bpm_bins:";
                for (const auto& bin : stats.top_bpm_bins) {
                    std::cerr << " " << bin.first << "(" << bin.second << ")";
                }
            }
            std::cerr << "\n";
        };
        print_stats("tempo_peaks", d.tempo_stats);
        if (d.has_downbeat_stats) {
            print_stats("downbeat_peaks", d.downbeat_stats);
        }
        print_stats("decoded_beats", d.decoded_stats);
        print_stats("decoded_beats_filtered", d.decoded_filtered_stats);
        if (d.short_interval_threshold > 0.0) {
            std::cerr << "DBN stats: filter_threshold=" << d.short_interval_threshold
                      << " min_interval=" << d.min_interval_frames << "\n";
        }
    }
    if (config.verbose) {
        std::cerr << "DBN grid: bpm=" << decoded.bpm
                  << " bpm_from_fit=" << d.bpm_from_fit
                  << " bpm_from_global_fit=" << d.bpm_from_global_fit
                  << " bpm_from_peaks=" << d.bpm_from_peaks
                  << " bpm_from_peaks_median=" << d.bpm_from_peaks_median
                  << " bpm_from_peaks_reg=" << d.bpm_from_peaks_reg
                  << " bpm_from_peaks_median_full=" << d.bpm_from_peaks_median_full
                  << " bpm_from_peaks_reg_full=" << d.bpm_from_peaks_reg_full
                  << " bpm_from_downbeats=" << d.bpm_from_downbeats
                  << " bpm_from_downbeats_median=" << d.bpm_from_downbeats_median
                  << " bpm_from_downbeats_reg=" << d.bpm_from_downbeats_reg
                  << " base_interval=" << decision.base_interval
                  << " bpm_reference=" << reference_bpm
                  << " quality_qpar=" << d.quality_qpar
                  << " quality_qkur=" << d.quality_qkur
                  << " quality_low=" << (decision.quality_low ? 1 : 0)
                  << " bpm_for_grid=" << decision.bpm_for_grid
                  << " step_frames=" << decision.step_frames
                  << " start_frame=" << decoded.beat_frames.front()
                  << "\n";
    }
    if (config.dbn_trace) {
        std::cerr << "DBN quality gate: low=" << (decision.quality_low ? 1 : 0)
                  << " drop_ref=" << (d.drop_ref ? 1 : 0)
                  << " drop_fit=" << (d.drop_fit ? 1 : 0)
                  << " downbeat_ok=" << (decision.downbeat_override_ok ? 1 : 0)
                  << " downbeat_cv=" << d.downbeat_cv
                  << " downbeat_count=" << d.downbeat_count
                  << " used=" << d.bpm_source
                  << " pre_override=" << d.bpm_source_before_downbeat
                  << " pre_bpm=" << d.bpm_before_downbeat
                  << "\n";
    }
}

GridTempoDecision compute_grid_tempo_decision(const DBNDecodeResult& decoded,
                                              const CoreMLResult& result,
                                              const CoreMLConfig& config,
                                              const CalmdadDecoder& calmdad_decoder,
                                              bool use_window,
                                              const std::vector<float>& beat_slice,
                                              const std::vector<float>& downbeat_slice,
                                              bool quality_valid,
                                              double quality_qpar,
                                              double quality_qkur,
                                              std::size_t used_frames,
                                              float min_bpm,
                                              float max_bpm,
                                              float reference_bpm,
                                              double fps) {
    GridTempoDecision decision;
    auto& diag = decision.diagnostics;
    diag.quality_qpar = quality_qpar;
    diag.quality_qkur = quality_qkur;
    if (decoded.beat_frames.empty()) {
        return decision;
    }

    const double min_interval_frames =
        (max_bpm > 1.0f && fps > 0.0) ? (60.0 * fps) / max_bpm : 0.0;
    const double short_interval_threshold =
        (min_interval_frames > 0.0) ? std::max(1.0, min_interval_frames * 0.5) : 0.0;
    diag.min_interval_frames = min_interval_frames;
    diag.short_interval_threshold = short_interval_threshold;
    const std::vector<std::size_t> filtered_beats =
        filter_short_intervals(decoded.beat_frames, short_interval_threshold);
    const std::vector<std::size_t> aligned_downbeats =
        align_downbeats_to_beats(filtered_beats, decoded.downbeat_frames);

    std::vector<std::size_t> bpb_candidates;
    if (config.dbn_mode == CoreMLConfig::DBNMode::Calmdad) {
        bpb_candidates = {3, 4};
    } else {
        bpb_candidates = {config.dbn_beats_per_bar};
    }
    const auto inferred_bpb_phase =
        infer_bpb_phase(filtered_beats, aligned_downbeats, bpb_candidates, config);
    decision.bpb = inferred_bpb_phase.first;
    decision.phase = inferred_bpb_phase.second;
    decision.base_interval = median_interval_frames(filtered_beats);

    const std::vector<float>& tempo_activation = use_window ? beat_slice : result.beat_activation;
    const float tempo_threshold =
        std::max(config.dbn_activation_floor, config.activation_threshold * 0.5f);
    const std::size_t tempo_min_interval =
        static_cast<std::size_t>(
            std::max(1.0, std::floor((60.0 * fps) / std::max(1.0f, max_bpm))));
    const std::size_t tempo_max_interval = static_cast<std::size_t>(
        std::max<double>(tempo_min_interval, std::ceil((60.0 * fps) / std::max(1.0f, min_bpm))));
    const std::vector<std::size_t> tempo_peaks =
        pick_peaks(tempo_activation, tempo_threshold, tempo_min_interval, tempo_max_interval);
    const std::vector<std::size_t> tempo_peaks_full =
        use_window ? pick_peaks(result.beat_activation,
                                tempo_threshold,
                                tempo_min_interval,
                                tempo_max_interval)
                   : tempo_peaks;
    double bpm_from_peaks = 0.0;
    double bpm_from_peaks_median = 0.0;
    double bpm_from_peaks_reg = 0.0;
    double bpm_from_peaks_median_full = 0.0;
    double bpm_from_peaks_reg_full = 0.0;
    if (tempo_peaks.size() >= 2) {
        const double interval_median =
            median_interval_frames_interpolated(tempo_activation, tempo_peaks);
        const double interval_reg =
            regression_interval_frames_interpolated(tempo_activation, tempo_peaks);
        if (interval_median > 0.0) {
            bpm_from_peaks_median = (60.0 * fps) / interval_median;
            bpm_from_peaks = bpm_from_peaks_median;
        }
        if (interval_reg > 0.0) {
            bpm_from_peaks_reg = (60.0 * fps) / interval_reg;
            if (bpm_from_peaks_median > 0.0) {
                const double ratio =
                    std::abs(bpm_from_peaks_reg - bpm_from_peaks_median) / bpm_from_peaks_median;
                if (ratio <= 0.02) {
                    bpm_from_peaks = bpm_from_peaks_reg;
                }
            } else {
                bpm_from_peaks = bpm_from_peaks_reg;
            }
        }
    }
    if (tempo_peaks_full.size() >= 2) {
        const double interval_median =
            median_interval_frames_interpolated(result.beat_activation, tempo_peaks_full);
        const double interval_reg =
            regression_interval_frames_interpolated(result.beat_activation, tempo_peaks_full);
        if (interval_median > 0.0) {
            bpm_from_peaks_median_full = (60.0 * fps) / interval_median;
        }
        if (interval_reg > 0.0) {
            bpm_from_peaks_reg_full = (60.0 * fps) / interval_reg;
        }
    }
    double bpm_from_downbeats = 0.0;
    double bpm_from_downbeats_median = 0.0;
    double bpm_from_downbeats_reg = 0.0;
    std::vector<std::size_t> downbeat_peaks;
    IntervalStats downbeat_stats;
    if (!result.downbeat_activation.empty() && decision.bpb > 0) {
        const std::vector<float>& downbeat_activation =
            use_window ? downbeat_slice : result.downbeat_activation;
        const float downbeat_min_bpm =
            std::max(1.0f, min_bpm / static_cast<float>(decision.bpb));
        const float downbeat_max_bpm =
            std::max(downbeat_min_bpm + 1.0f, max_bpm / static_cast<float>(decision.bpb));
        const std::size_t downbeat_min_interval = static_cast<std::size_t>(
            std::max(1.0, std::floor((60.0 * fps) / downbeat_max_bpm)));
        const std::size_t downbeat_max_interval = static_cast<std::size_t>(
            std::max<double>(downbeat_min_interval, std::ceil((60.0 * fps) / downbeat_min_bpm)));
        downbeat_peaks = pick_peaks(downbeat_activation,
                                    tempo_threshold,
                                    downbeat_min_interval,
                                    downbeat_max_interval);
        if (downbeat_peaks.size() >= 2) {
            const double interval_median =
                median_interval_frames_interpolated(downbeat_activation, downbeat_peaks);
            const double interval_reg =
                regression_interval_frames_interpolated(downbeat_activation, downbeat_peaks);
            if (interval_median > 0.0) {
                const double downbeat_bpm = (60.0 * fps) / interval_median;
                bpm_from_downbeats_median = downbeat_bpm * static_cast<double>(decision.bpb);
                bpm_from_downbeats = bpm_from_downbeats_median;
            }
            if (interval_reg > 0.0) {
                const double downbeat_bpm = (60.0 * fps) / interval_reg;
                bpm_from_downbeats_reg = downbeat_bpm * static_cast<double>(decision.bpb);
                if (bpm_from_downbeats_median > 0.0) {
                    const double ratio = std::abs(bpm_from_downbeats_reg - bpm_from_downbeats_median) /
                        bpm_from_downbeats_median;
                    if (ratio <= 0.02) {
                        bpm_from_downbeats = bpm_from_downbeats_reg;
                    }
                } else {
                    bpm_from_downbeats = bpm_from_downbeats_reg;
                }
            }
        }
    }
    if (!downbeat_peaks.empty()) {
        downbeat_stats = interval_stats_interpolated(use_window ? downbeat_slice
                                                                : result.downbeat_activation,
                                                     downbeat_peaks,
                                                     fps,
                                                     0.2);
        diag.has_downbeat_stats = true;
    }
    if (config.dbn_trace) {
        diag.stats_computed = true;
        diag.tempo_stats = interval_stats_interpolated(tempo_activation, tempo_peaks, fps, 0.2);
        diag.decoded_stats = interval_stats_frames(decoded.beat_frames, fps, 0.2);
        diag.decoded_filtered_stats = interval_stats_frames(filtered_beats, fps, 0.2);
        if (diag.has_downbeat_stats) {
            diag.downbeat_stats = downbeat_stats;
        }
    }

    const double bpm_from_fit = bpm_from_linear_fit(filtered_beats, fps);
    const double bpm_from_global_fit = detail::bpm_from_global_fit(result,
                                                                   config,
                                                                   calmdad_decoder,
                                                                   fps,
                                                                   min_bpm,
                                                                   max_bpm,
                                                                   used_frames);
    diag.bpm_from_fit = bpm_from_fit;
    diag.bpm_from_global_fit = bpm_from_global_fit;
    decision.quality_low = quality_valid && (quality_qkur < 3.6);
    const bool drop_fit = decision.quality_low && bpm_from_fit > 0.0;
    diag.drop_fit = drop_fit;
    const std::size_t downbeat_count = downbeat_stats.count;
    const double downbeat_cv = (downbeat_count > 0 && downbeat_stats.mean_interval > 0.0)
        ? (downbeat_stats.stdev_interval / downbeat_stats.mean_interval)
        : 0.0;
    diag.downbeat_count = downbeat_count;
    diag.downbeat_cv = downbeat_cv;
    decision.downbeat_override_ok = !decision.quality_low && downbeat_count >= 6 && downbeat_cv <= 0.25;
    const double ref_downbeat_ratio =
        (decision.downbeat_override_ok && bpm_from_downbeats > 0.0)
            ? (std::abs(reference_bpm - bpm_from_downbeats) / bpm_from_downbeats)
            : 0.0;
    const bool ref_mismatch =
        decision.downbeat_override_ok && bpm_from_downbeats > 0.0 && ref_downbeat_ratio > 0.005;
    const bool drop_ref = (decision.quality_low || ref_mismatch) && reference_bpm > 0.0f;
    diag.drop_ref = drop_ref;
    const bool allow_reference_grid_bpm =
        reference_bpm > 0.0f &&
        ((static_cast<double>(max_bpm) - static_cast<double>(min_bpm)) <=
         std::max(2.0, static_cast<double>(reference_bpm) * 0.05));
    bool global_fit_plausible = false;
    if (bpm_from_global_fit > 0.0 && bpm_from_fit > 0.0) {
        const double diff = std::abs(bpm_from_global_fit - bpm_from_fit);
        const double rel_diff = diff / bpm_from_fit;
        global_fit_plausible = rel_diff <= 0.08;
    }
    diag.global_fit_plausible = global_fit_plausible;
    std::string bpm_source = "none";
    if (global_fit_plausible) {
        decision.bpm_for_grid = bpm_from_global_fit;
        bpm_source = "global_fit";
    } else if (allow_reference_grid_bpm && !decision.quality_low && !ref_mismatch) {
        decision.bpm_for_grid = reference_bpm;
        bpm_source = "reference";
    } else if (decision.downbeat_override_ok && bpm_from_downbeats > 0.0) {
        if (bpm_from_fit > 0.0) {
            decision.bpm_for_grid = bpm_from_fit;
            bpm_source = "fit_primary";
        } else {
            decision.bpm_for_grid = bpm_from_downbeats;
            bpm_source = "downbeats_primary";
        }
    } else if (!decision.quality_low && bpm_from_peaks_reg_full > 0.0) {
        decision.bpm_for_grid = bpm_from_peaks_reg_full;
        bpm_source = "peaks_reg_full";
    } else if (!decision.quality_low && bpm_from_fit > 0.0) {
        decision.bpm_for_grid = bpm_from_fit;
        bpm_source = "fit";
    } else if (bpm_from_peaks_median > 0.0) {
        decision.bpm_for_grid = bpm_from_peaks_median;
        bpm_source = "peaks_median";
    } else if (bpm_from_peaks > 0.0) {
        decision.bpm_for_grid = bpm_from_peaks;
        bpm_source = "peaks";
    }
    if (decision.bpm_for_grid <= 0.0 && allow_reference_grid_bpm) {
        decision.bpm_for_grid = reference_bpm;
        bpm_source = "reference_fallback";
    }
    const double bpm_before_downbeat = decision.bpm_for_grid;
    const std::string bpm_source_before_downbeat = bpm_source;
    diag.bpm_before_downbeat = bpm_before_downbeat;
    diag.bpm_source_before_downbeat = bpm_source_before_downbeat;
    if (decision.downbeat_override_ok && bpm_from_downbeats > 0.0 && decision.bpm_for_grid > 0.0 &&
        bpm_source != "peaks_reg_full" && bpm_source != "downbeats_primary" &&
        bpm_source != "fit_primary") {
        const double ratio = std::abs(bpm_from_downbeats - decision.bpm_for_grid) / decision.bpm_for_grid;
        if (ratio <= 0.005) {
            decision.bpm_for_grid = bpm_from_downbeats;
            bpm_source = "downbeats_override";
        }
    }
    if (decision.bpm_for_grid <= 0.0 && decoded.bpm > 0.0) {
        decision.bpm_for_grid = decoded.bpm;
        bpm_source = "decoded";
    }
    if (decision.bpm_for_grid <= 0.0 && decision.base_interval > 0.0) {
        decision.bpm_for_grid = (60.0 * fps) / decision.base_interval;
        bpm_source = "base_interval";
    }
    diag.bpm_source = bpm_source;
    decision.step_frames = (decision.bpm_for_grid > 0.0)
        ? (60.0 * fps) / decision.bpm_for_grid
        : decision.base_interval;
    diag.bpm_from_peaks = bpm_from_peaks;
    diag.bpm_from_peaks_median = bpm_from_peaks_median;
    diag.bpm_from_peaks_reg = bpm_from_peaks_reg;
    diag.bpm_from_peaks_median_full = bpm_from_peaks_median_full;
    diag.bpm_from_peaks_reg_full = bpm_from_peaks_reg_full;
    diag.bpm_from_downbeats = bpm_from_downbeats;
    diag.bpm_from_downbeats_median = bpm_from_downbeats_median;
    diag.bpm_from_downbeats_reg = bpm_from_downbeats_reg;

    return decision;
}

struct GridAnchorSeed {
    std::size_t earliest_peak = 0;
    std::size_t earliest_downbeat_peak = 0;
    std::size_t strongest_peak = 0;
    float strongest_peak_value = -1.0f;
    float activation_floor = 0.0f;
};

GridAnchorSeed seed_grid_anchor(DBNDecodeResult& decoded,
                                const CoreMLResult& result,
                                const CoreMLConfig& config,
                                bool use_window,
                                std::size_t window_start,
                                std::size_t used_frames,
                                double base_interval) {
    GridAnchorSeed anchor;
    if (decoded.beat_frames.empty()) {
        return anchor;
    }
    anchor.earliest_peak = decoded.beat_frames.front();
    anchor.earliest_downbeat_peak = decoded.beat_frames.front();
    anchor.strongest_peak = decoded.beat_frames.front();
    anchor.activation_floor = std::max(0.01f, config.activation_threshold * 0.1f);
    float earliest_downbeat_value = 0.0f;
    if (base_interval <= 1.0 || decoded.beat_frames.empty()) {
        return anchor;
    }
    const std::size_t peak_search_start = use_window ? window_start : 0;
    const std::size_t peak_search_end = use_window
        ? std::min<std::size_t>(used_frames - 1,
                                window_start + static_cast<std::size_t>(std::llround(base_interval)))
        : std::min<std::size_t>(used_frames - 1,
                                static_cast<std::size_t>(std::llround(base_interval)));
    if (!result.beat_activation.empty()) {
        if (peak_search_start + 1 <= peak_search_end) {
            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                const float prev = result.beat_activation[i - 1];
                const float curr = result.beat_activation[i];
                const float next = result.beat_activation[i + 1];
                if (curr >= anchor.activation_floor && curr >= prev && curr >= next) {
                    anchor.earliest_peak = i;
                    break;
                }
            }
        }
        if (peak_search_start + 1 <= peak_search_end) {
            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                const float prev = result.beat_activation[i - 1];
                const float curr = result.beat_activation[i];
                const float next = result.beat_activation[i + 1];
                if (curr >= anchor.activation_floor && curr >= prev && curr >= next) {
                    if (curr > anchor.strongest_peak_value) {
                        anchor.strongest_peak_value = curr;
                        anchor.strongest_peak = i;
                    }
                }
            }
        }
        if (anchor.strongest_peak_value < 0.0f && anchor.earliest_peak < result.beat_activation.size()) {
            anchor.strongest_peak = anchor.earliest_peak;
            anchor.strongest_peak_value = result.beat_activation[anchor.earliest_peak];
        }
    }
    if (!result.downbeat_activation.empty()) {
        float max_downbeat = 0.0f;
        for (std::size_t i = peak_search_start; i <= peak_search_end; ++i) {
            max_downbeat = std::max(max_downbeat, result.downbeat_activation[i]);
        }
        const float onset_threshold =
            std::max(anchor.activation_floor, max_downbeat * config.dbn_downbeat_phase_peak_ratio);
        for (std::size_t i = peak_search_start; i <= peak_search_end; ++i) {
            const float curr = result.downbeat_activation[i];
            if (curr >= onset_threshold) {
                anchor.earliest_downbeat_peak = i;
                earliest_downbeat_value = curr;
                break;
            }
        }
    }
    if (config.verbose) {
        std::cerr << "DBN grid: earliest_peak=" << anchor.earliest_peak
                  << " earliest_downbeat_peak=" << anchor.earliest_downbeat_peak
                  << " earliest_downbeat_value=" << earliest_downbeat_value
                  << " strongest_peak=" << anchor.strongest_peak
                  << " strongest_peak_value=" << anchor.strongest_peak_value
                  << " activation_floor=" << anchor.activation_floor
                  << "\n";
    }
    std::size_t start_peak = decoded.beat_frames.front();
    if (!result.beat_activation.empty()) {
        std::size_t earliest = start_peak;
        if (peak_search_start + 1 <= peak_search_end) {
            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                const float prev = result.beat_activation[i - 1];
                const float curr = result.beat_activation[i];
                const float next = result.beat_activation[i + 1];
                if (curr >= anchor.activation_floor && curr >= prev && curr >= next) {
                    earliest = i;
                    break;
                }
            }
        }
        if (earliest < start_peak) {
            start_peak = earliest;
        }
        if (config.dbn_grid_start_strong_peak && anchor.strongest_peak_value >= anchor.activation_floor) {
            start_peak = anchor.strongest_peak;
        }
    }
    std::vector<std::size_t> forward = fill_peaks_with_grid(result.beat_activation,
                                                            start_peak,
                                                            used_frames - 1,
                                                            base_interval,
                                                            anchor.activation_floor);
    std::vector<std::size_t> backward;
    double cursor = static_cast<double>(start_peak) - base_interval;
    const std::size_t window =
        static_cast<std::size_t>(std::max(1.0, std::round(base_interval * 0.25)));
    while (cursor >= 0.0) {
        const std::size_t center = static_cast<std::size_t>(std::llround(cursor));
        const std::size_t start = center > window ? center - window : 0;
        const std::size_t end = std::min(used_frames - 1, center + window);
        float best_value = -1.0f;
        std::size_t best_index = center;
        for (std::size_t k = start; k <= end; ++k) {
            const float value = result.beat_activation[k];
            if (value > best_value) {
                best_value = value;
                best_index = k;
            }
        }
        std::size_t chosen = best_index;
        if (best_value < anchor.activation_floor) {
            chosen = center;
        }
        backward.push_back(chosen);
        if (cursor < base_interval) {
            break;
        }
        cursor -= base_interval;
    }
    std::sort(backward.begin(), backward.end());
    std::vector<std::size_t> combined;
    combined.reserve(backward.size() + forward.size());
    combined.insert(combined.end(), backward.begin(), backward.end());
    if (!forward.empty() && (combined.empty() || combined.back() != forward.front())) {
        combined.insert(combined.end(), forward.begin(), forward.end());
    } else if (forward.size() > 1) {
        combined.insert(combined.end(), forward.begin() + 1, forward.end());
    }
    decoded.beat_frames = std::move(combined);
    return anchor;
}

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

    auto dedupe_frames = [&](std::vector<std::size_t>& frames) {
        detail::dedupe_frames(frames);
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
        dedupe_frames(decoded.beat_frames);

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
            dedupe_frames(decoded.downbeat_frames);
        }

        const GridTempoDecision tempo_decision = compute_grid_tempo_decision(decoded,
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
                                                                             fps);
        log_grid_tempo_decision(tempo_decision, decoded, config, reference_bpm, fps);
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
