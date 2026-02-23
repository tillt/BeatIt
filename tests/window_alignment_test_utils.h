//
//  window_alignment_test_utils.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "beatit/stream.h"

namespace beatit::tests::window_alignment {

inline std::string compile_model_if_needed(const std::string& path, std::string* error) {
    NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:ns_path];
    if (!url) {
        if (error) {
            *error = "Failed to create model URL.";
        }
        return {};
    }

    NSString* ext = url.pathExtension.lowercaseString;
    if (![ext isEqualToString:@"mlpackage"] && ![ext isEqualToString:@"mlmodel"]) {
        return path;
    }

    std::string tmp_dir;
    try {
        tmp_dir = (std::filesystem::current_path() / "coreml_tmp").string();
        std::filesystem::create_directories(tmp_dir);
        setenv("TMPDIR", tmp_dir.c_str(), 1);
    } catch (const std::exception&) {
        if (error) {
            *error = "Failed to prepare temporary directory for CoreML compile.";
        }
        return {};
    }

    NSError* compile_error = nil;
    NSURL* compiled_url = [MLModel compileModelAtURL:url error:&compile_error];
    if (!compiled_url || compile_error) {
        if (error) {
            std::string message = "Failed to compile CoreML model.";
            if (compile_error) {
                message += " ";
                message += compile_error.localizedDescription.UTF8String;
            }
            *error = message;
        }
        return {};
    }

    return compiled_url.path.UTF8String ? compiled_url.path.UTF8String : std::string();
}

inline bool decode_audio_mono(const std::string& path,
                              std::vector<float>* mono_out,
                              double* sample_rate_out,
                              std::string* error) {
    if (!mono_out) {
        if (error) {
            *error = "Output pointer is null.";
        }
        return false;
    }

    @autoreleasepool {
        NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:ns_path];
        NSError* open_error = nil;
        AVAudioFile* file = [[AVAudioFile alloc] initForReading:url error:&open_error];
        if (!file || open_error) {
            if (error) {
                *error = "AVAudioFile failed to open.";
            }
            return false;
        }

        AVAudioFormat* input_format = file.processingFormat;
        const double sample_rate = input_format.sampleRate > 0.0 ? input_format.sampleRate : 44100.0;
        const AVAudioChannelCount channels = input_format.channelCount > 0 ? input_format.channelCount : 1;
        AVAudioFormat* render_format =
            [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                              sampleRate:sample_rate
                                                channels:channels
                                             interleaved:NO];

        AVAudioConverter* converter = [[AVAudioConverter alloc] initFromFormat:input_format
                                                                       toFormat:render_format];
        converter.sampleRateConverterQuality = AVAudioQualityMax;
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering;

        const AVAudioFrameCount chunk_frames = 4096;
        AVAudioPCMBuffer* input_buffer =
            [[AVAudioPCMBuffer alloc] initWithPCMFormat:input_format frameCapacity:chunk_frames];
        AVAudioPCMBuffer* output_buffer =
            [[AVAudioPCMBuffer alloc] initWithPCMFormat:render_format frameCapacity:chunk_frames];

        mono_out->clear();
        __block BOOL done_reading = NO;
        __block NSError* read_error = nil;
        while (!done_reading) {
            input_buffer.frameLength = 0;
            output_buffer.frameLength = 0;

            AVAudioConverterInputBlock input_block =
                ^AVAudioBuffer* _Nullable(AVAudioPacketCount, AVAudioConverterInputStatus* out_status) {
                    NSError* local_error = nil;
                    if (![file readIntoBuffer:input_buffer error:&local_error] ||
                        input_buffer.frameLength == 0) {
                        if (local_error) {
                            read_error = local_error;
                        }
                        *out_status = AVAudioConverterInputStatus_EndOfStream;
                        done_reading = YES;
                        return nil;
                    }
                    if (local_error) {
                        read_error = local_error;
                        done_reading = YES;
                        *out_status = AVAudioConverterInputStatus_EndOfStream;
                        return nil;
                    }
                    *out_status = AVAudioConverterInputStatus_HaveData;
                    return input_buffer;
                };

            NSError* convert_error = nil;
            AVAudioConverterOutputStatus status =
                [converter convertToBuffer:output_buffer
                                     error:&convert_error
                       withInputFromBlock:input_block];
            if (status == AVAudioConverterOutputStatus_Error || convert_error) {
                if (error) {
                    std::string message = "AVAudioConverter failed.";
                    if (convert_error) {
                        message += " ";
                        message += convert_error.localizedDescription.UTF8String;
                    }
                    *error = message;
                }
                return false;
            }
            if (status == AVAudioConverterOutputStatus_EndOfStream) {
                done_reading = YES;
            }
            if (status != AVAudioConverterOutputStatus_HaveData ||
                output_buffer.frameLength == 0) {
                continue;
            }

            const AVAudioFrameCount frames = output_buffer.frameLength;
            const std::size_t base = mono_out->size();
            mono_out->resize(base + frames, 0.0f);
            for (AVAudioChannelCount ch = 0; ch < channels; ++ch) {
                const float* channel_data = output_buffer.floatChannelData[ch];
                for (AVAudioFrameCount i = 0; i < frames; ++i) {
                    (*mono_out)[base + i] += channel_data[i];
                }
            }
            const float scale = 1.0f / static_cast<float>(channels);
            for (AVAudioFrameCount i = 0; i < frames; ++i) {
                (*mono_out)[base + i] *= scale;
            }
        }

        if (read_error) {
            if (error) {
                *error = read_error.localizedDescription.UTF8String;
            }
            return false;
        }

        if (sample_rate_out) {
            *sample_rate_out = sample_rate;
        }
    }

    return true;
}

inline double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    auto mid = values.begin() + static_cast<long>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

inline double linear_slope(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double n = static_cast<double>(values.size());
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double x = static_cast<double>(i);
        const double y = values[i];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double den = n * sxx - sx * sx;
    if (std::fabs(den) < 1e-12) {
        return 0.0;
    }
    return (n * sxy - sx * sy) / den;
}

inline double robust_linear_slope(const std::vector<double>& values, double beat_period_ms) {
    if (values.size() < 2) {
        return 0.0;
    }

    const double center = median(values);
    std::vector<double> abs_dev;
    abs_dev.reserve(values.size());
    for (double v : values) {
        abs_dev.push_back(std::fabs(v - center));
    }
    const double mad = median(abs_dev);
    const double scale = 1.4826 * mad;
    const double raw_limit = 3.0 * std::max(1.0, scale);
    const double min_limit = std::max(12.0, beat_period_ms * 0.03);
    const double max_limit = std::max(30.0, beat_period_ms * 0.08);
    const double inlier_limit = std::min(std::max(raw_limit, min_limit), max_limit);

    std::vector<std::size_t> inlier_idx;
    inlier_idx.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::fabs(values[i] - center) <= inlier_limit) {
            inlier_idx.push_back(i);
        }
    }
    if (inlier_idx.size() < 2) {
        return linear_slope(values);
    }

    const double n = static_cast<double>(inlier_idx.size());
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t idx : inlier_idx) {
        const double x = static_cast<double>(idx);
        const double y = values[idx];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double den = n * sxx - sx * sx;
    if (std::fabs(den) < 1e-12) {
        return 0.0;
    }
    return (n * sxy - sx * sy) / den;
}

inline std::vector<double> unwrap_periodic_offsets_ms(const std::vector<double>& wrapped,
                                                      double period_ms) {
    if (wrapped.empty() || period_ms <= 1e-6) {
        return wrapped;
    }

    auto fit_line = [](const std::vector<double>& values, double* intercept, double* slope) {
        const double n = static_cast<double>(values.size());
        if (values.size() < 2 || !intercept || !slope) {
            if (intercept) {
                *intercept = values.empty() ? 0.0 : values.front();
            }
            if (slope) {
                *slope = 0.0;
            }
            return;
        }

        double sx = 0.0;
        double sy = 0.0;
        double sxx = 0.0;
        double sxy = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double x = static_cast<double>(i);
            const double y = values[i];
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        const double den = n * sxx - sx * sx;
        if (std::fabs(den) < 1e-12) {
            *slope = 0.0;
            *intercept = sy / n;
            return;
        }
        *slope = (n * sxy - sx * sy) / den;
        *intercept = (sy - (*slope) * sx) / n;
    };

    std::vector<double> unwrapped = wrapped;
    double intercept = 0.0;
    double slope = 0.0;
    fit_line(unwrapped, &intercept, &slope);

    for (int iter = 0; iter < 8; ++iter) {
        for (std::size_t i = 0; i < wrapped.size(); ++i) {
            const double trend = intercept + slope * static_cast<double>(i);
            const double base = wrapped[i];
            const long long k =
                static_cast<long long>(std::llround((trend - base) / period_ms));
            unwrapped[i] = base + static_cast<double>(k) * period_ms;
        }

        double new_intercept = 0.0;
        double new_slope = 0.0;
        fit_line(unwrapped, &new_intercept, &new_slope);

        const bool converged =
            std::fabs(new_intercept - intercept) < 1e-3 &&
            std::fabs(new_slope - slope) < 1e-6;
        intercept = new_intercept;
        slope = new_slope;
        if (converged) {
            break;
        }
    }

    return unwrapped;
}

inline bool is_bar_event(const beatit::BeatEvent& event) {
    return (static_cast<unsigned long long>(event.style) &
            static_cast<unsigned long long>(beatit::BeatEventStyleBar)) != 0ULL;
}

inline bool first_bar_is_complete_four_four(const beatit::AnalysisResult& result) {
    std::vector<std::size_t> bar_indices;
    bar_indices.reserve(result.coreml_beat_events.size() / 4);
    for (std::size_t i = 0; i < result.coreml_beat_events.size(); ++i) {
        if (is_bar_event(result.coreml_beat_events[i])) {
            bar_indices.push_back(i);
        }
    }
    if (bar_indices.size() < 2) {
        return false;
    }
    // Accept a single lead-in beat before the first full 4/4 bar.
    const bool first_bar_not_too_late = bar_indices[0] <= 1;
    const bool full_first_bar = (bar_indices[1] - bar_indices[0]) == 4;
    return first_bar_not_too_late && full_first_bar;
}

inline bool bars_repeat_every_four_beats(const beatit::AnalysisResult& result) {
    std::vector<std::size_t> bar_indices;
    bar_indices.reserve(result.coreml_beat_events.size() / 4);
    for (std::size_t i = 0; i < result.coreml_beat_events.size(); ++i) {
        if (is_bar_event(result.coreml_beat_events[i])) {
            bar_indices.push_back(i);
        }
    }
    if (bar_indices.size() < 2) {
        return false;
    }
    for (std::size_t i = 1; i < bar_indices.size(); ++i) {
        if ((bar_indices[i] - bar_indices[i - 1]) != 4) {
            return false;
        }
    }
    return true;
}

inline bool first_downbeat_sample_frame(const beatit::AnalysisResult& result,
                                        unsigned long long* out_frame) {
    for (const auto& event : result.coreml_beat_events) {
        if (is_bar_event(event)) {
            if (out_frame) {
                *out_frame = event.frame;
            }
            return true;
        }
    }
    return false;
}

inline bool first_downbeat_feature_frame(const beatit::AnalysisResult& result,
                                         unsigned long long* out_frame) {
    if (result.coreml_downbeat_feature_frames.empty()) {
        return false;
    }
    if (out_frame) {
        *out_frame = result.coreml_downbeat_feature_frames.front();
    }
    return true;
}

inline double median_interval_seconds(const std::vector<unsigned long long>& beat_frames,
                                      double sample_rate,
                                      std::size_t interval_count,
                                      bool from_end) {
    if (beat_frames.size() < 3 || sample_rate <= 0.0 || interval_count == 0) {
        return 0.0;
    }
    std::vector<double> intervals;
    intervals.reserve(beat_frames.size() - 1);
    for (std::size_t i = 1; i < beat_frames.size(); ++i) {
        if (beat_frames[i] > beat_frames[i - 1]) {
            intervals.push_back(
                static_cast<double>(beat_frames[i] - beat_frames[i - 1]) / sample_rate);
        }
    }
    if (intervals.empty()) {
        return 0.0;
    }

    const std::size_t n = std::min(interval_count, intervals.size());
    std::vector<double> slice;
    if (from_end) {
        slice.assign(intervals.end() - static_cast<long>(n), intervals.end());
    } else {
        slice.assign(intervals.begin(), intervals.begin() + static_cast<long>(n));
    }
    return median(slice);
}

inline std::vector<double> compute_strong_peak_offsets_ms(const std::vector<unsigned long long>& beat_frames,
                                                          const std::vector<float>& mono,
                                                          double sample_rate,
                                                          double bpm) {
    std::vector<double> offsets;
    if (beat_frames.empty() || mono.empty() || sample_rate <= 0.0 || bpm <= 0.0) {
        return offsets;
    }

    const std::size_t radius = static_cast<std::size_t>(
        std::llround(sample_rate * (60.0 / bpm) * 0.6));
    offsets.reserve(beat_frames.size());

    for (unsigned long long frame_ull : beat_frames) {
        const std::size_t frame =
            static_cast<std::size_t>(std::min<unsigned long long>(frame_ull, mono.size() - 1));
        const std::size_t start = frame > radius ? frame - radius : 0;
        const std::size_t end = std::min(mono.size() - 1, frame + radius);
        if (end <= start + 2) {
            offsets.push_back(0.0);
            continue;
        }

        float window_max = 0.0f;
        for (std::size_t i = start; i <= end; ++i) {
            window_max = std::max(window_max, std::fabs(mono[i]));
        }
        const float threshold = window_max * 0.6f;

        std::size_t best_peak = frame;
        float best_value = 0.0f;
        for (std::size_t i = start + 1; i < end; ++i) {
            const float left = std::fabs(mono[i - 1]);
            const float value = std::fabs(mono[i]);
            const float right = std::fabs(mono[i + 1]);
            if (value < threshold) {
                continue;
            }
            if (value >= left && value > right && value > best_value) {
                best_value = value;
                best_peak = i;
            }
        }

        if (best_value <= 0.0f) {
            float max_value = 0.0f;
            for (std::size_t i = start; i <= end; ++i) {
                const float value = std::fabs(mono[i]);
                if (value > max_value) {
                    max_value = value;
                    best_peak = i;
                }
            }
        }

        const double delta_frames = static_cast<double>(static_cast<long long>(best_peak) -
                                                        static_cast<long long>(frame));
        offsets.push_back((delta_frames * 1000.0) / sample_rate);
    }
    return offsets;
}

inline double median_abs_frame_delta(const std::vector<unsigned long long>& a,
                                     const std::vector<unsigned long long>& b) {
    if (a.empty() || b.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const std::size_t n = std::min(a.size(), b.size());
    std::vector<double> deltas;
    deltas.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const long long ai = static_cast<long long>(a[i]);
        const long long bi = static_cast<long long>(b[i]);
        deltas.push_back(static_cast<double>(std::llabs(ai - bi)));
    }
    return median(std::move(deltas));
}

template <typename T>
inline std::string format_slice(const std::vector<T>& values, std::size_t count, bool from_end) {
    if (values.empty()) {
        return "()";
    }
    const std::size_t n = std::min(count, values.size());
    const std::size_t start = from_end ? (values.size() - n) : 0;
    std::ostringstream os;
    os << "(";
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << values[start + i];
    }
    os << ")";
    return os.str();
}

inline std::string format_double_slice(const std::vector<double>& values,
                                       std::size_t count,
                                       bool from_end,
                                       int precision = 3) {
    if (values.empty()) {
        return "()";
    }
    const std::size_t n = std::min(count, values.size());
    const std::size_t start = from_end ? (values.size() - n) : 0;
    std::ostringstream os;
    os << "(";
    os << std::fixed << std::setprecision(precision);
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << values[start + i];
    }
    os << ")";
    return os.str();
}

} // namespace beatit::tests::window_alignment
