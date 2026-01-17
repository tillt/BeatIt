#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "beatit/refiner.h"
#include "beatit/stream.h"

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

bool stream_audio_to_beatit(const std::string& path,
                            const beatit::CoreMLConfig& config,
                            beatit::AnalysisResult* result,
                            double* duration_seconds,
                            double* input_sample_rate,
                            double* last_active_seconds,
                            std::string* error) {
    if (!result) {
        if (error) {
            *error = "Result pointer is null.";
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

        beatit::BeatitStream stream(sample_rate, config, true);
        std::vector<float> mono;
        std::size_t total_frames = 0;
        std::size_t last_active_frame = 0;
        bool has_active = false;
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
            mono.assign(frames, 0.0f);
            for (AVAudioChannelCount ch = 0; ch < channels; ++ch) {
                const float* channel_data = output_buffer.floatChannelData[ch];
                for (AVAudioFrameCount i = 0; i < frames; ++i) {
                    mono[i] += channel_data[i];
                }
            }
            const float scale = 1.0f / static_cast<float>(channels);
            double sum_sq = 0.0;
            for (float& sample : mono) {
                sample *= scale;
                sum_sq += static_cast<double>(sample) * static_cast<double>(sample);
            }

            stream.push(mono);
            const double rms = frames > 0 ? std::sqrt(sum_sq / static_cast<double>(frames)) : 0.0;
            if (rms >= 0.001) {
                last_active_frame = total_frames + frames - 1;
                has_active = true;
            }
            total_frames += frames;
        }

        if (total_frames == 0) {
            if (error) {
                std::string message = "No audio frames were decoded.";
                if (read_error) {
                    message += " ";
                    message += read_error.localizedDescription.UTF8String;
                }
                *error = message;
            }
            return false;
        }

        if (duration_seconds) {
            *duration_seconds = static_cast<double>(total_frames) / sample_rate;
        }
        if (input_sample_rate) {
            *input_sample_rate = sample_rate;
        }
        if (last_active_seconds) {
            if (has_active) {
                *last_active_seconds = static_cast<double>(last_active_frame) / sample_rate;
            } else {
                *last_active_seconds = 0.0;
            }
        }
        *result = stream.finalize();
    }

    return true;
}

bool verify_result(const beatit::AnalysisResult& result, std::string* error) {
    if (result.coreml_beat_activation.empty() || result.coreml_beat_feature_frames.empty()) {
        if (error) {
            *error = "No CoreML beat output.";
        }
        return false;
    }
    if (result.coreml_beat_sample_frames.size() != result.coreml_beat_feature_frames.size()) {
        if (error) {
            *error = "Beat samples and frames count mismatch.";
        }
        return false;
    }
    if (!(result.estimated_bpm > 0.0)) {
        if (error) {
            *error = "Estimated BPM is non-positive.";
        }
        return false;
    }
    for (std::size_t i = 1; i < result.coreml_beat_sample_frames.size(); ++i) {
        if (result.coreml_beat_sample_frames[i] <= result.coreml_beat_sample_frames[i - 1]) {
            if (error) {
                *error = "Beat samples are not strictly increasing.";
            }
            return false;
        }
    }
    return true;
}

struct DeviationStats {
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

DeviationStats compute_deviation_stats(const std::vector<unsigned long long>& beat_samples,
                                       double sample_rate,
                                       double bpm) {
    DeviationStats stats;
    if (beat_samples.size() < 2 || sample_rate <= 0.0 || bpm <= 0.0) {
        return stats;
    }

    const double interval = 60.0 / bpm;
    std::vector<double> deviations;
    deviations.reserve(beat_samples.size() - 1);

    for (std::size_t i = 1; i < beat_samples.size(); ++i) {
        const double prev = static_cast<double>(beat_samples[i - 1]) / sample_rate;
        const double next = static_cast<double>(beat_samples[i]) / sample_rate;
        const double delta = next - prev;
        const double error = (delta - interval) * 1000.0;
        deviations.push_back(error);
    }

    if (deviations.empty()) {
        return stats;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    stats.min_ms = deviations.front();
    stats.max_ms = deviations.front();
    for (double value : deviations) {
        sum += value;
        sum_sq += value * value;
        stats.min_ms = std::min(stats.min_ms, value);
        stats.max_ms = std::max(stats.max_ms, value);
    }
    stats.mean_ms = sum / static_cast<double>(deviations.size());
    const double mean_sq = sum_sq / static_cast<double>(deviations.size());
    const double variance = std::max(0.0, mean_sq - stats.mean_ms * stats.mean_ms);
    stats.stddev_ms = std::sqrt(variance);
    return stats;
}

struct BpmStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
};

BpmStats compute_interval_stats(const std::vector<unsigned long long>& beat_samples,
                                double sample_rate) {
    BpmStats stats;
    if (beat_samples.size() < 2 || sample_rate <= 0.0) {
        return stats;
    }

    std::vector<double> intervals;
    intervals.reserve(beat_samples.size() - 1);
    for (std::size_t i = 1; i < beat_samples.size(); ++i) {
        const unsigned long long prev = beat_samples[i - 1];
        const unsigned long long next = beat_samples[i];
        if (next > prev) {
            intervals.push_back(static_cast<double>(next - prev) / sample_rate);
        }
    }

    if (intervals.empty()) {
        return stats;
    }

    double sum = 0.0;
    double min_val = intervals.front();
    double max_val = intervals.front();
    for (double value : intervals) {
        sum += value;
        min_val = std::min(min_val, value);
        max_val = std::max(max_val, value);
    }

    const double mean = sum / static_cast<double>(intervals.size());
    double variance = 0.0;
    for (double value : intervals) {
        const double diff = value - mean;
        variance += diff * diff;
    }
    stats.mean = mean;
    stats.stddev = std::sqrt(variance / static_cast<double>(intervals.size()));
    stats.min = min_val;
    stats.max = max_val;
    return stats;
}

BpmStats compute_bpm_stats(const std::vector<unsigned long long>& beat_samples, double sample_rate) {
    BpmStats stats;
    if (beat_samples.size() < 2 || sample_rate <= 0.0) {
        return stats;
    }

    std::vector<double> bpms;
    bpms.reserve(beat_samples.size() - 1);
    for (std::size_t i = 1; i < beat_samples.size(); ++i) {
        const unsigned long long prev = beat_samples[i - 1];
        const unsigned long long next = beat_samples[i];
        if (next > prev) {
            const double interval = static_cast<double>(next - prev) / sample_rate;
            if (interval > 0.0) {
                bpms.push_back(60.0 / interval);
            }
        }
    }

    if (bpms.empty()) {
        return stats;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    stats.min = bpms.front();
    stats.max = bpms.front();
    for (double value : bpms) {
        sum += value;
        sum_sq += value * value;
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
    }
    stats.mean = sum / static_cast<double>(bpms.size());
    const double mean_sq = sum_sq / static_cast<double>(bpms.size());
    const double variance = std::max(0.0, mean_sq - stats.mean * stats.mean);
    stats.stddev = std::sqrt(variance);
    return stats;
}

}  // namespace

int main() {
    std::string model_error;
    std::string model_path = compile_model_if_needed(BEATIT_TEST_MODEL_PATH, &model_error);
    if (model_path.empty()) {
        std::cerr << "Failed to prepare CoreML model: " << model_error << "\n";
        return 77;
    }

    beatit::CoreMLConfig config;
    config.model_path = model_path;
    config.verbose = true;
    config.activation_threshold = 0.45f;
    if (const char* force_cpu = std::getenv("BEATIT_TEST_CPU_ONLY")) {
        if (force_cpu[0] != '\0' && force_cpu[0] != '0') {
            config.compute_units = beatit::CoreMLConfig::ComputeUnits::CPUOnly;
        }
    }

    beatit::AnalysisResult result;
    std::string error;
    double duration_seconds = 0.0;
    double input_sample_rate = 0.0;
    double last_active_seconds = 0.0;
    const char* audio_override = std::getenv("BEATIT_TEST_AUDIO_OVERRIDE");
    const std::string audio_path =
        (audio_override && audio_override[0] != '\0') ? audio_override : BEATIT_TEST_AUDIO_PATH;
    if (!stream_audio_to_beatit(audio_path,
                                config,
                                &result,
                                &duration_seconds,
                                &input_sample_rate,
                                &last_active_seconds,
                                &error)) {
        std::cerr << "Failed to decode/stream audio: " << error << "\n";
        return 77;
    }
    if (!verify_result(result, &error)) {
        std::cerr << "Streaming CoreML MP3 test failed: " << error << "\n";
        return 1;
    }

    const double expected_beats = duration_seconds * (result.estimated_bpm / 60.0);
    const double expected_beats_active = last_active_seconds * (result.estimated_bpm / 60.0);
    double last_beat_time = 0.0;
    if (!result.coreml_beat_sample_frames.empty()) {
        last_beat_time =
            static_cast<double>(result.coreml_beat_sample_frames.back()) / input_sample_rate;
    }
    const double expected_last_beat_time =
        std::floor(expected_beats) * (60.0 / result.estimated_bpm);
    const DeviationStats deviation =
        compute_deviation_stats(result.coreml_beat_sample_frames, input_sample_rate, result.estimated_bpm);
    const BpmStats bpm_stats =
        compute_bpm_stats(result.coreml_beat_sample_frames, input_sample_rate);
    const BpmStats raw_interval_stats =
        compute_interval_stats(result.coreml_beat_sample_frames, input_sample_rate);

    beatit::ConstantBeatRefinerConfig refiner_config;
    beatit::ConstantBeatResult refined =
        beatit::refine_constant_beats(result.coreml_beat_feature_frames,
                                      result.coreml_beat_activation.size(),
                                      config,
                                      input_sample_rate,
                                      0,
                                      refiner_config,
                                      &result.coreml_beat_activation,
                                      &result.coreml_downbeat_activation,
                                      result.coreml_phase_energy.empty()
                                          ? nullptr
                                          : &result.coreml_phase_energy);
    if (refined.beat_sample_frames.empty() || refined.bpm <= 0.0f) {
        std::cerr << "Constant refiner failed to produce beats.\n";
        return 1;
    }

    const BpmStats refined_interval_stats =
        compute_interval_stats(refined.beat_sample_frames, input_sample_rate);
    const double refined_stddev_ms = refined_interval_stats.stddev * 1000.0;
    const double raw_stddev_ms = raw_interval_stats.stddev * 1000.0;
    if (refined_stddev_ms > 8.0) {
        std::cerr << "Constant refiner jitter too high: " << refined_stddev_ms << "ms\n";
        return 1;
    }
    if (raw_stddev_ms > 0.0 && refined_stddev_ms > raw_stddev_ms + 1.0) {
        std::cerr << "Constant refiner did not improve jitter: raw="
                  << raw_stddev_ms << "ms refined=" << refined_stddev_ms << "ms\n";
        return 1;
    }
    if (last_active_seconds > 0.0 && refined.beat_sample_frames.size() > 1) {
        const double interval_seconds = 60.0 / refined.bpm;
        const double last_refined =
            static_cast<double>(refined.beat_sample_frames.back()) / input_sample_rate;
        const double slack = std::max(interval_seconds * 3.0, 6.0);
        if (std::fabs(last_active_seconds - last_refined) > slack) {
            std::cerr << "Constant refiner last beat not anchored near active audio. "
                      << "last_active=" << last_active_seconds
                      << " last_beat=" << last_refined
                      << " slack=" << slack << "\n";
            return 1;
        }
    }
    std::cout << "Streaming CoreML MP3 test passed. BPM=" << result.estimated_bpm
              << " beats=" << result.coreml_beat_feature_frames.size()
              << " duration=" << duration_seconds
              << " expected_beats=" << expected_beats
              << " active_duration=" << last_active_seconds
              << " expected_beats_active=" << expected_beats_active
              << " last_beat_time=" << last_beat_time
              << " expected_last_beat_time=" << expected_last_beat_time
              << " dev_ms(mean=" << deviation.mean_ms
              << " stddev=" << deviation.stddev_ms
              << " min=" << deviation.min_ms
              << " max=" << deviation.max_ms << ")"
              << " bpm_stats(mean=" << bpm_stats.mean
              << " stddev=" << bpm_stats.stddev
              << " min=" << bpm_stats.min
              << " max=" << bpm_stats.max << ")"
              << " refiner_bpm=" << refined.bpm
              << " refiner_beats=" << refined.beat_feature_frames.size()
              << " refiner_stddev_ms=" << refined_stddev_ms
              << "\n";
    return 0;
}
