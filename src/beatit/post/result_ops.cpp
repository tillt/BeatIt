//
//  result_ops.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/result_ops.h"
#include "beatit/post/helpers.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>

namespace beatit::detail {

namespace {

std::size_t clamp_subtract(std::size_t value, std::size_t amount) {
    return (value > amount) ? (value - amount) : 0;
}

void dedupe_frames_with_tolerance(std::vector<std::size_t>& frames, std::size_t tolerance) {
    if (frames.empty()) {
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

double feature_to_sample_position(double feature_frame,
                                  const BeatitConfig& config,
                                  double hop_scale) {
    return (feature_frame * static_cast<double>(config.hop_size)) * hop_scale;
}

long long quantized_sample_frame(double sample_position, long long latency_samples = 0) {
    const long long sample_frame = static_cast<long long>(std::llround(sample_position)) - latency_samples;
    return std::max<long long>(0, sample_frame);
}

std::size_t sample_to_feature_frame(unsigned long long sample_frame,
                                    const BeatitConfig& config,
                                    double hop_scale) {
    const double feature_pos =
        (static_cast<double>(sample_frame) / hop_scale) /
        static_cast<double>(config.hop_size);
    return static_cast<std::size_t>(std::llround(feature_pos));
}

void reset_aligned_beats(std::vector<unsigned long long>& feature_frames,
                         std::vector<unsigned long long>& sample_frames,
                         std::vector<float>& strengths,
                         std::size_t reserve_count) {
    feature_frames.clear();
    feature_frames.reserve(reserve_count);
    sample_frames.clear();
    sample_frames.reserve(reserve_count);
    strengths.clear();
    strengths.reserve(reserve_count);
}

void dedupe_aligned_beats(std::vector<unsigned long long>& sample_frames,
                          std::vector<unsigned long long>& feature_frames,
                          std::vector<float>& strengths) {
    if (sample_frames.empty()) {
        return;
    }

    std::size_t write = 1;
    unsigned long long last = sample_frames[0];
    for (std::size_t i = 1; i < sample_frames.size(); ++i) {
        const unsigned long long current = sample_frames[i];
        if (current <= last) {
            continue;
        }
        sample_frames[write] = current;
        feature_frames[write] = feature_frames[i];
        strengths[write] = strengths[i];
        last = current;
        ++write;
    }
    sample_frames.resize(write);
    feature_frames.resize(write);
    strengths.resize(write);
}

} // namespace

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
    dedupe_frames_with_tolerance(frames, 0);
}

void dedupe_frames_tolerant(std::vector<std::size_t>& frames, std::size_t tolerance) {
    if (tolerance == 0) {
        dedupe_frames(frames);
        return;
    }
    dedupe_frames_with_tolerance(frames, tolerance);
}

std::vector<std::size_t> apply_latency_to_frames(const std::vector<std::size_t>& frames,
                                                 std::size_t analysis_latency_frames) {
    if (analysis_latency_frames == 0 || frames.empty()) {
        return frames;
    }
    std::vector<std::size_t> adjusted;
    adjusted.reserve(frames.size());
    for (std::size_t frame : frames) {
        adjusted.push_back(clamp_subtract(frame, analysis_latency_frames));
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
    reset_aligned_beats(result.beat_feature_frames,
                        result.beat_sample_frames,
                        result.beat_strengths,
                        frames.size());

    const long long latency_samples = static_cast<long long>(
        std::llround(config.output_latency_seconds * sample_rate));

    for (std::size_t frame : frames) {
        const std::size_t output_frame = clamp_subtract(frame, analysis_latency_frames);
        result.beat_feature_frames.push_back(static_cast<unsigned long long>(output_frame));
        const std::size_t peak_frame = refine_frame_to_peak(frame, result.beat_activation, refine_window);
        double frame_pos = interpolate_peak_position(result.beat_activation, peak_frame);
        if (analysis_latency_frames > 0) {
            frame_pos = std::max(0.0, frame_pos - analysis_latency_frames_f);
        }
        const double sample_pos = feature_to_sample_position(frame_pos, config, hop_scale);
        const long long sample_frame = quantized_sample_frame(sample_pos, latency_samples);
        result.beat_sample_frames.push_back(static_cast<unsigned long long>(sample_frame));
        if (!result.beat_activation.empty()) {
            result.beat_strengths.push_back(result.beat_activation[peak_frame]);
        } else {
            result.beat_strengths.push_back(0.0f);
        }
    }

    dedupe_aligned_beats(result.beat_sample_frames,
                         result.beat_feature_frames,
                         result.beat_strengths);
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
    reset_aligned_beats(out_feature_frames, out_sample_frames, out_strengths, 0);

    if (bpm <= 0.0 || fps <= 0.0 || total_frames == 0) {
        return;
    }

    const double step_frames = (60.0 * fps) / bpm;
    if (step_frames <= 0.0) {
        return;
    }

    const double start_frame_adjusted = static_cast<double>(start_frame);
    const double start_sample_pos =
        feature_to_sample_position(start_frame_adjusted, config, hop_scale);
    const long long start_sample_frame = quantized_sample_frame(start_sample_pos);

    if (config.dbn_trace) {
        const double start_time = start_frame_adjusted / fps;
        const double start_time_after_latency =
            sample_rate > 0.0
                ? static_cast<double>(start_sample_frame) / sample_rate
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
        const std::size_t frame = sample_to_feature_frame(sample_frame, config, hop_scale);
        out_feature_frames.push_back(static_cast<unsigned long long>(frame));

        if (!beat_activation.empty() && frame < beat_activation.size()) {
            out_strengths.push_back(beat_activation[frame]);
        } else {
            out_strengths.push_back(0.0f);
        }
    }

    dedupe_aligned_beats(out_sample_frames, out_feature_frames, out_strengths);

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
