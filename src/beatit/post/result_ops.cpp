//
//  result_ops.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/result_ops.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>

namespace beatit::detail {

std::size_t refine_frame_to_peak(std::size_t frame,
                                 const std::vector<float>& activation,
                                 std::size_t window) {
    if (activation.empty()) {
        return frame;
    }
    const std::size_t start = frame > window ? frame - window : 0;
    const std::size_t end = std::min(frame + window, activation.size() - 1);
    std::size_t best_index = frame;
    float best_value = activation[frame];
    for (std::size_t i = start; i <= end; ++i) {
        const float value = activation[i];
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    return best_index;
}

void dedupe_frames(std::vector<std::size_t>& frames) {
    if (frames.empty()) {
        return;
    }
    std::size_t write = 1;
    std::size_t last = frames[0];
    for (std::size_t i = 1; i < frames.size(); ++i) {
        const std::size_t current = frames[i];
        if (current <= last) {
            continue;
        }
        frames[write++] = current;
        last = current;
    }
    frames.resize(write);
}

void dedupe_frames_tolerant(std::vector<std::size_t>& frames, std::size_t tolerance) {
    if (frames.empty()) {
        return;
    }
    if (tolerance == 0) {
        dedupe_frames(frames);
        return;
    }
    std::size_t write = 1;
    std::size_t last = frames[0];
    for (std::size_t i = 1; i < frames.size(); ++i) {
        const std::size_t current = frames[i];
        if (current <= last + tolerance) {
            continue;
        }
        frames[write++] = current;
        last = current;
    }
    frames.resize(write);
}

std::vector<std::size_t> apply_latency_to_frames(const std::vector<std::size_t>& frames,
                                                 std::size_t analysis_latency_frames) {
    if (analysis_latency_frames == 0 || frames.empty()) {
        return frames;
    }
    std::vector<std::size_t> adjusted;
    adjusted.reserve(frames.size());
    for (std::size_t frame : frames) {
        if (frame > analysis_latency_frames) {
            adjusted.push_back(frame - analysis_latency_frames);
        } else {
            adjusted.push_back(0);
        }
    }
    dedupe_frames(adjusted);
    return adjusted;
}

void fill_beats_from_frames(CoreMLResult& result,
                            const std::vector<std::size_t>& frames,
                            const BeatitConfig& config,
                            double sample_rate,
                            double hop_scale,
                            std::size_t analysis_latency_frames,
                            double analysis_latency_frames_f,
                            std::size_t refine_window) {
    result.beat_feature_frames.clear();
    result.beat_feature_frames.reserve(frames.size());
    result.beat_sample_frames.clear();
    result.beat_sample_frames.reserve(frames.size());
    result.beat_strengths.clear();
    result.beat_strengths.reserve(frames.size());

    const long long latency_samples = static_cast<long long>(
        std::llround(config.output_latency_seconds * sample_rate));

    for (std::size_t frame : frames) {
        const std::size_t output_frame =
            (analysis_latency_frames > 0 && frame > analysis_latency_frames)
                ? (frame - analysis_latency_frames)
                : (analysis_latency_frames > 0 ? 0 : frame);
        result.beat_feature_frames.push_back(static_cast<unsigned long long>(output_frame));
        const std::size_t peak_frame = refine_frame_to_peak(frame, result.beat_activation, refine_window);

        double frame_pos = static_cast<double>(peak_frame);
        if (peak_frame > 0 && peak_frame + 1 < result.beat_activation.size()) {
            const double prev = result.beat_activation[peak_frame - 1];
            const double curr = result.beat_activation[peak_frame];
            const double next = result.beat_activation[peak_frame + 1];
            const double denom = prev - 2.0 * curr + next;
            if (std::abs(denom) > 1e-9) {
                double offset = 0.5 * (prev - next) / denom;
                offset = std::max(-0.5, std::min(0.5, offset));
                frame_pos += offset;
            }
        }
        if (analysis_latency_frames > 0) {
            frame_pos = std::max(0.0, frame_pos - analysis_latency_frames_f);
        }
        const double sample_pos =
            (frame_pos * static_cast<double>(config.hop_size)) * hop_scale;
        long long sample_frame = static_cast<long long>(std::llround(sample_pos)) - latency_samples;
        if (sample_frame < 0) {
            sample_frame = 0;
        }
        result.beat_sample_frames.push_back(static_cast<unsigned long long>(sample_frame));
        if (!result.beat_activation.empty()) {
            result.beat_strengths.push_back(result.beat_activation[peak_frame]);
        } else {
            result.beat_strengths.push_back(0.0f);
        }
    }

    if (!result.beat_sample_frames.empty()) {
        std::size_t write = 1;
        unsigned long long last = result.beat_sample_frames[0];
        for (std::size_t i = 1; i < result.beat_sample_frames.size(); ++i) {
            const unsigned long long current = result.beat_sample_frames[i];
            if (current <= last) {
                continue;
            }
            result.beat_sample_frames[write] = current;
            result.beat_feature_frames[write] = result.beat_feature_frames[i];
            result.beat_strengths[write] = result.beat_strengths[i];
            last = current;
            ++write;
        }
        result.beat_sample_frames.resize(write);
        result.beat_feature_frames.resize(write);
        result.beat_strengths.resize(write);
    }
}

void fill_beats_from_bpm_grid_into(const std::vector<float>& beat_activation,
                                   const BeatitConfig& config,
                                   double sample_rate,
                                   double fps,
                                   double hop_scale,
                                   std::size_t start_frame,
                                   double bpm,
                                   std::size_t total_frames,
                                   std::vector<unsigned long long>& out_feature_frames,
                                   std::vector<unsigned long long>& out_sample_frames,
                                   std::vector<float>& out_strengths) {
    out_feature_frames.clear();
    out_sample_frames.clear();
    out_strengths.clear();

    if (bpm <= 0.0 || fps <= 0.0 || total_frames == 0) {
        return;
    }

    const double step_frames = (60.0 * fps) / bpm;
    if (step_frames <= 0.0) {
        return;
    }

    const double start_frame_adjusted = static_cast<double>(start_frame);

    if (config.dbn_trace) {
        const double start_time = start_frame_adjusted / fps;
        const double start_sample_pos =
            (start_frame_adjusted * static_cast<double>(config.hop_size)) * hop_scale;
        const long long start_sample_frame =
            static_cast<long long>(std::llround(start_sample_pos));
        const double start_time_after_latency =
            sample_rate > 0.0
                ? static_cast<double>(std::max<long long>(0, start_sample_frame)) /
                    sample_rate
                : 0.0;
        BEATIT_LOG_DEBUG("DBN grid project: start_frame=" << start_frame
                         << " start_time=" << start_time
                         << " bpm=" << bpm
                         << " step_frames=" << step_frames
                         << " total_frames=" << total_frames
                         << " hop_size=" << config.hop_size
                         << " hop_scale=" << hop_scale
                         << " start_sample_frame=" << start_sample_frame
                         << " start_time_adj=" << start_time_after_latency);
    }

    const double step_samples = (60.0 * sample_rate) / bpm;
    if (step_samples <= 0.0) {
        return;
    }
    const double start_sample_pos =
        (start_frame_adjusted * static_cast<double>(config.hop_size)) * hop_scale;
    const long long start_sample_frame = static_cast<long long>(std::llround(start_sample_pos));
    std::vector<unsigned long long> grid_samples;
    grid_samples.reserve(static_cast<std::size_t>(
        std::ceil(static_cast<double>(total_frames) / step_frames)) + 4);
    std::size_t backward_count = 0;
    std::size_t forward_count = 0;

    double sample_pos = static_cast<double>(start_sample_frame);
    while (sample_pos >= step_samples) {
        sample_pos -= step_samples;
        grid_samples.push_back(static_cast<unsigned long long>(
            std::llround(sample_pos)));
    }
    std::reverse(grid_samples.begin(), grid_samples.end());
    backward_count = grid_samples.size();
    grid_samples.push_back(static_cast<unsigned long long>(start_sample_frame));

    sample_pos = static_cast<double>(start_sample_frame) + step_samples;
    const double total_samples = static_cast<double>(total_frames) *
        static_cast<double>(config.hop_size) * hop_scale;
    while (sample_pos < total_samples) {
        grid_samples.push_back(static_cast<unsigned long long>(
            std::llround(sample_pos)));
        ++forward_count;
        sample_pos += step_samples;
    }

    out_feature_frames.reserve(grid_samples.size());
    out_sample_frames.reserve(grid_samples.size());
    out_strengths.reserve(grid_samples.size());

    for (unsigned long long sample_frame : grid_samples) {
        out_sample_frames.push_back(sample_frame);
        const double feature_pos =
            (static_cast<double>(sample_frame) / hop_scale) /
            static_cast<double>(config.hop_size);
        const std::size_t frame = static_cast<std::size_t>(std::llround(feature_pos));
        out_feature_frames.push_back(static_cast<unsigned long long>(frame));

        if (!beat_activation.empty() && frame < beat_activation.size()) {
            out_strengths.push_back(beat_activation[frame]);
        } else {
            out_strengths.push_back(0.0f);
        }
    }

    if (!out_sample_frames.empty()) {
        std::size_t write = 1;
        unsigned long long last = out_sample_frames[0];
        for (std::size_t i = 1; i < out_sample_frames.size(); ++i) {
            const unsigned long long current = out_sample_frames[i];
            if (current <= last) {
                continue;
            }
            out_sample_frames[write] = current;
            out_feature_frames[write] = out_feature_frames[i];
            out_strengths[write] = out_strengths[i];
            last = current;
            ++write;
        }
        out_sample_frames.resize(write);
        out_feature_frames.resize(write);
        out_strengths.resize(write);
    }

    if (config.dbn_trace) {
        const std::size_t preview = std::min<std::size_t>(6, out_feature_frames.size());
        auto debug_stream = BEATIT_LOG_DEBUG_STREAM();
        debug_stream << "DBN grid beats head:";
        for (std::size_t i = 0; i < preview; ++i) {
            const std::size_t frame = static_cast<std::size_t>(out_feature_frames[i]);
            const double time_s = static_cast<double>(frame) / fps;
            debug_stream << " " << frame << "(" << time_s << "s)";
        }
        debug_stream << "\n";
        debug_stream << "DBN grid beats total=" << out_feature_frames.size()
                     << " backward=" << backward_count
                     << " forward=" << forward_count;
    }
}

} // namespace beatit::detail
