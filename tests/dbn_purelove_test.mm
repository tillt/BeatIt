#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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
            for (float& sample : mono) {
                sample *= scale;
            }

            stream.push(mono);
        }

        if (read_error) {
            if (error) {
                *error = "Failed to decode audio data.";
            }
            return false;
        }

        *result = stream.finalize();
    }

    return true;
}

bool expect_le(std::size_t value, std::size_t expected, const char* label) {
    if (value > expected) {
        std::cerr << "DBN test failed: " << label << " expected <= " << expected
                  << " got " << value << "\n";
        return false;
    }
    return true;
}

bool expect_non_positive(double value, const char* label) {
    if (value > 0.0) {
        std::cerr << "DBN test failed: " << label << " expected <= 0 got " << value << "\n";
        return false;
    }
    return true;
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
    config.activation_threshold = 0.4f;
    config.window_hop_frames = 2500;
    config.use_dbn = true;
    config.dbn_use_all_candidates = true;
    config.dbn_activation_floor = 0.01f;
    config.dbn_interval_tolerance = 0.3f;
    config.dbn_max_candidates = 4096;
    config.min_bpm = 55.0f;
    config.max_bpm = 400.0f;
    config.prefer_double_time = false;

    if (const char* force_cpu = std::getenv("BEATIT_TEST_CPU_ONLY")) {
        if (force_cpu[0] != '\0' && force_cpu[0] != '0') {
            config.compute_units = beatit::CoreMLConfig::ComputeUnits::CPUOnly;
        }
    }

    beatit::AnalysisResult result;
    std::string error;
    if (!stream_audio_to_beatit(BEATIT_TEST_AUDIO_PATH, config, &result, &error)) {
        std::cerr << "DBN purelove test failed: " << error << "\n";
        return 1;
    }
    if (result.coreml_beat_activation.empty()) {
        std::cerr << "SKIP: CoreML output unavailable (model load failed or unsupported).\n";
        return 77;
    }

    if (!expect_le(result.coreml_beat_feature_frames.size(), 1, "beat count")) {
        return 1;
    }
    if (!expect_non_positive(result.estimated_bpm, "estimated bpm")) {
        return 1;
    }

    std::cout << "DBN purelove regression test passed.\n";
    return 0;
}
