//
//  file_drift_probe_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "beatit/stream.h"
#include "coreml_test_config.h"

namespace {

std::string compile_model_if_needed(const std::string& path, std::string* error) {
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

bool decode_audio_mono(const std::string& path,
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

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    auto mid = values.begin() + static_cast<long>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

double linear_slope(const std::vector<double>& values) {
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

double robust_linear_slope(const std::vector<double>& values, double beat_period_ms) {
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

std::vector<double> unwrap_periodic_offsets_ms(const std::vector<double>& wrapped, double period_ms) {
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
            const long long k = static_cast<long long>(std::llround((trend - base) / period_ms));
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

double median_interval_seconds(const std::vector<unsigned long long>& beat_frames,
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

std::vector<double> compute_strong_peak_offsets_ms(const std::vector<unsigned long long>& beat_frames,
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

}  // namespace

int main(int argc, char** argv) {
    std::string source_model_path = beatit::tests::resolve_beatthis_coreml_model_path();
    if (source_model_path.empty()) {
        std::cerr << "SKIP: BeatThis CoreML model missing (set BEATIT_COREML_MODEL_PATH).\n";
        return 77;
    }

    std::string model_error;
    std::string model_path = compile_model_if_needed(source_model_path, &model_error);
    if (model_path.empty()) {
        std::cerr << "SKIP: Failed to prepare CoreML model: " << model_error << "\n";
        return 77;
    }

    std::filesystem::path test_root = std::filesystem::current_path();
#if defined(BEATIT_TEST_DATA_DIR)
    test_root = BEATIT_TEST_DATA_DIR;
#endif

    std::filesystem::path audio_path;
    if (argc >= 2) {
        audio_path = std::filesystem::path(argv[1]);
    } else if (const char* env_path = std::getenv("BEATIT_DRIFT_AUDIO_PATH")) {
        if (env_path[0] != '\0') {
            audio_path = std::filesystem::path(env_path);
        }
    }
    if (audio_path.empty()) {
        audio_path = test_root / "training" / "neural.wav";
    }
    if (!std::filesystem::exists(audio_path)) {
        std::cerr << "Drift probe failed: missing " << audio_path.string() << "\n";
        return 1;
    }

    beatit::BeatitConfig config;
    beatit::tests::apply_beatthis_coreml_test_config(config);
    config.model_path = model_path;
    config.use_dbn = true;
    config.max_analysis_seconds = 60.0;
    config.dbn_window_start_seconds = 0.0;
    config.sparse_probe_mode = true;
    if (const char* trace = std::getenv("BEATIT_WINDOW_TRACE")) {
        if (trace[0] != '\0' && trace[0] != '0') {
            config.log_verbosity = beatit::LogVerbosity::Debug;
            config.dbn_trace = true;
            config.profile = true;
        }
    }
    if (const char* force_cpu = std::getenv("BEATIT_TEST_CPU_ONLY")) {
        if (force_cpu[0] != '\0' && force_cpu[0] != '0') {
            config.compute_units = beatit::BeatitConfig::ComputeUnits::CPUOnly;
        }
    }

    std::vector<float> mono;
    double sample_rate = 0.0;
    std::string decode_error;
    if (!decode_audio_mono(audio_path.string(), &mono, &sample_rate, &decode_error)) {
        std::cerr << "Drift probe failed: decode error: " << decode_error << "\n";
        return 1;
    }
    const double total_duration_s = static_cast<double>(mono.size()) / sample_rate;
    auto provider =
        [&](double start_seconds, double duration_seconds, std::vector<float>* out_samples) -> std::size_t {
            if (!out_samples) {
                return 0;
            }
            out_samples->clear();
            if (sample_rate <= 0.0 || mono.empty()) {
                return 0;
            }
            const std::size_t begin = static_cast<std::size_t>(
                std::llround(std::max(0.0, start_seconds) * sample_rate));
            const std::size_t count = static_cast<std::size_t>(
                std::llround(std::max(0.0, duration_seconds) * sample_rate));
            const std::size_t end = std::min(mono.size(), begin + count);
            if (begin >= end) {
                return 0;
            }
            out_samples->assign(mono.begin() + static_cast<long>(begin),
                                mono.begin() + static_cast<long>(end));
            return out_samples->size();
        };

    beatit::BeatitStream stream(sample_rate, config, true);
    double start_s = 0.0;
    double duration_s = 0.0;
    if (!stream.request_analysis_window(&start_s, &duration_s)) {
        std::cerr << "Drift probe failed: request_analysis_window returned false.\n";
        return 1;
    }
    const beatit::AnalysisResult result =
        stream.analyze_window(start_s, duration_s, total_duration_s, provider);

    const std::vector<unsigned long long>& beat_frames =
        result.coreml_beat_projected_sample_frames.empty()
            ? result.coreml_beat_sample_frames
            : result.coreml_beat_projected_sample_frames;
    if (beat_frames.size() < 16 || !(result.estimated_bpm > 0.0f)) {
        std::cout << "Drift probe: file=" << audio_path.string()
                  << " bpm=" << result.estimated_bpm
                  << " beats=" << beat_frames.size()
                  << " status=insufficient\n";
        return 0;
    }

    const std::vector<double> offsets_ms =
        compute_strong_peak_offsets_ms(beat_frames, mono, sample_rate, result.estimated_bpm);
    if (offsets_ms.size() < 16) {
        std::cout << "Drift probe: file=" << audio_path.string()
                  << " bpm=" << result.estimated_bpm
                  << " beats=" << beat_frames.size()
                  << " offsets=" << offsets_ms.size()
                  << " status=insufficient_offsets\n";
        return 0;
    }

    const std::size_t edge = std::min<std::size_t>(64, offsets_ms.size() / 4);
    if (edge < 8) {
        std::cout << "Drift probe: file=" << audio_path.string()
                  << " bpm=" << result.estimated_bpm
                  << " beats=" << beat_frames.size()
                  << " offsets=" << offsets_ms.size()
                  << " status=insufficient_edge_window\n";
        return 0;
    }

    const std::size_t middle_start = std::min<std::size_t>(
        offsets_ms.size() - edge, (offsets_ms.size() - edge) / 2);
    const std::size_t between_start = std::min<std::size_t>(offsets_ms.size() - edge, middle_start / 2);
    std::vector<double> first(offsets_ms.begin(), offsets_ms.begin() + static_cast<long>(edge));
    std::vector<double> between(offsets_ms.begin() + static_cast<long>(between_start),
                                offsets_ms.begin() + static_cast<long>(between_start + edge));
    std::vector<double> middle(offsets_ms.begin() + static_cast<long>(middle_start),
                               offsets_ms.begin() + static_cast<long>(middle_start + edge));
    std::vector<double> last(offsets_ms.end() - static_cast<long>(edge), offsets_ms.end());
    const double start_median_ms = median(first);
    const double between_median_ms = median(between);
    const double middle_median_ms = median(middle);
    const double end_median_ms = median(last);

    std::vector<double> between_abs = between;
    for (double& v : between_abs) {
        v = std::fabs(v);
    }
    std::vector<double> middle_abs = middle;
    for (double& v : middle_abs) {
        v = std::fabs(v);
    }
    const double between_median_abs_ms = median(between_abs);
    const double middle_median_abs_ms = median(middle_abs);

    const double beat_period_ms = 60000.0 / std::max(1e-6f, result.estimated_bpm);
    const double start_end_delta_ms = end_median_ms - start_median_ms;
    const double start_end_delta_beats = std::fabs(start_end_delta_ms) / beat_period_ms;
    const double slope_ms_per_beat = robust_linear_slope(offsets_ms, beat_period_ms);

    const std::vector<double> unwrapped_offsets_ms =
        unwrap_periodic_offsets_ms(offsets_ms, beat_period_ms);
    std::vector<double> first_unwrapped(unwrapped_offsets_ms.begin(),
                                        unwrapped_offsets_ms.begin() + static_cast<long>(edge));
    std::vector<double> last_unwrapped(unwrapped_offsets_ms.end() - static_cast<long>(edge),
                                       unwrapped_offsets_ms.end());
    const double unwrapped_start_median_ms = median(first_unwrapped);
    const double unwrapped_end_median_ms = median(last_unwrapped);
    const double unwrapped_start_end_delta_ms =
        unwrapped_end_median_ms - unwrapped_start_median_ms;
    const double unwrapped_start_end_delta_beats =
        std::fabs(unwrapped_start_end_delta_ms) / beat_period_ms;
    const double unwrapped_slope_ms_per_beat =
        robust_linear_slope(unwrapped_offsets_ms, beat_period_ms);

    const double early_interval_s = median_interval_seconds(beat_frames, sample_rate, 64, false);
    const double late_interval_s = median_interval_seconds(beat_frames, sample_rate, 64, true);
    const double early_bpm = early_interval_s > 0.0 ? (60.0 / early_interval_s) : 0.0;
    const double late_bpm = late_interval_s > 0.0 ? (60.0 / late_interval_s) : 0.0;
    const double edge_bpm_delta = std::fabs(early_bpm - late_bpm);

    const double phase_hot_threshold_ms = std::max(35.0, beat_period_ms * 0.22);
    const bool edges_calm =
        std::fabs(start_median_ms) <= (phase_hot_threshold_ms * 0.5) &&
        std::fabs(end_median_ms) <= (phase_hot_threshold_ms * 0.5);
    const bool between_hot = between_median_abs_ms > phase_hot_threshold_ms;
    const bool middle_hot = middle_median_abs_ms > phase_hot_threshold_ms;
    const bool one_beat_linear_signature =
        unwrapped_start_end_delta_beats > 0.70 && unwrapped_start_end_delta_beats < 1.30;
    const bool wrapped_middle_signature = edges_calm && (between_hot || middle_hot);

    std::cout << "Drift probe: file=" << audio_path.string()
              << " bpm=" << result.estimated_bpm
              << " beats=" << beat_frames.size()
              << " start_med=" << start_median_ms << "ms"
              << " between_med=" << between_median_ms << "ms"
              << " middle_med=" << middle_median_ms << "ms"
              << " end_med=" << end_median_ms << "ms"
              << " delta=" << start_end_delta_ms << "ms"
              << " delta_beats=" << start_end_delta_beats
              << " slope=" << slope_ms_per_beat << "ms/beat"
              << " unwrapped_delta=" << unwrapped_start_end_delta_ms << "ms"
              << " unwrapped_delta_beats=" << unwrapped_start_end_delta_beats
              << " unwrapped_slope=" << unwrapped_slope_ms_per_beat << "ms/beat"
              << " between_abs=" << between_median_abs_ms << "ms"
              << " middle_abs=" << middle_median_abs_ms << "ms"
              << " edge_bpm_delta=" << edge_bpm_delta
              << " one_beat_linear_signature=" << (one_beat_linear_signature ? 1 : 0)
              << " wrapped_middle_signature=" << (wrapped_middle_signature ? 1 : 0)
              << "\n";

    if (const char* fail = std::getenv("BEATIT_DRIFT_FAIL_ON_SIGNATURE")) {
        if (fail[0] != '\0' && fail[0] != '0' &&
            (one_beat_linear_signature || wrapped_middle_signature)) {
            return 1;
        }
    }
    return 0;
}
