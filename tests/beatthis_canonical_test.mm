#import <AVFoundation/AVFoundation.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "beatit/coreml_preset.h"
#include "beatit/stream.h"

namespace {

struct CanonicalCase {
    const char* name;
    const char* filename;
    double bpm;
    double downbeat_seconds;
};

bool stream_audio_to_beatit(const std::string& path,
                            const beatit::CoreMLConfig& config,
                            beatit::AnalysisResult* result,
                            double* input_sample_rate,
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
        mono.reserve(chunk_frames);

        __block BOOL done_reading = NO;
        __block NSError* read_error = nil;

        while (!done_reading) {
            input_buffer.frameLength = 0;
            output_buffer.frameLength = 0;

            AVAudioConverterInputBlock input_block =
                ^AVAudioBuffer* _Nullable(AVAudioPacketCount, AVAudioConverterInputStatus* out_status) {
                    if (read_error) {
                        *out_status = AVAudioConverterInputStatus_NoDataNow;
                        return nil;
                    }
                    NSError* file_error = nil;
                    if (![file readIntoBuffer:input_buffer error:&file_error] || input_buffer.frameLength == 0) {
                        *out_status = AVAudioConverterInputStatus_EndOfStream;
                        done_reading = YES;
                        return nil;
                    }
                    if (file_error) {
                        read_error = file_error;
                        *out_status = AVAudioConverterInputStatus_NoDataNow;
                        return nil;
                    }
                    *out_status = AVAudioConverterInputStatus_HaveData;
                    return input_buffer;
                };

            NSError* convert_error = nil;
            AVAudioConverterOutputStatus status =
                [converter convertToBuffer:output_buffer error:&convert_error withInputFromBlock:input_block];

            if (status == AVAudioConverterOutputStatus_Error || convert_error) {
                if (error) {
                    *error = "AVAudioConverter failed.";
                    if (convert_error) {
                        *error += " ";
                        *error += convert_error.localizedDescription.UTF8String;
                    }
                }
                return false;
            }
            if (status == AVAudioConverterOutputStatus_EndOfStream) {
                break;
            }
            if (status != AVAudioConverterOutputStatus_HaveData || output_buffer.frameLength == 0) {
                continue;
            }

            mono.assign(output_buffer.frameLength, 0.0f);
            for (AVAudioChannelCount channel = 0; channel < channels; ++channel) {
                const float* channel_data = output_buffer.floatChannelData[channel];
                for (AVAudioFrameCount i = 0; i < output_buffer.frameLength; ++i) {
                    mono[i] += channel_data[i];
                }
            }
            const float scale = channels > 0 ? 1.0f / static_cast<float>(channels) : 1.0f;
            for (float& value : mono) {
                value *= scale;
            }

            stream.push(mono.data(), mono.size());
        }

        *result = stream.finalize();
        if (input_sample_rate) {
            *input_sample_rate = sample_rate;
        }
    }

    return true;
}

double downbeat_seconds_from_result(const beatit::AnalysisResult& result,
                                    const beatit::CoreMLConfig& config) {
    if (!result.coreml_downbeat_feature_frames.empty() && config.sample_rate > 0 && config.hop_size > 0) {
        const double frames = static_cast<double>(result.coreml_downbeat_feature_frames.front());
        const double seconds = (frames * config.hop_size) / config.sample_rate;
        if (config.prepend_silence_seconds > 0.0) {
            return std::max(0.0, seconds - config.prepend_silence_seconds);
        }
        return seconds;
    }
    return -1.0;
}

}  // namespace

int main() {
#if !defined(BEATIT_USE_TORCH)
    std::cerr << "SKIP: Torch backend not enabled.\n";
    return 77;
#endif

    beatit::CoreMLConfig config;
    if (auto preset = beatit::make_coreml_preset("beatthis")) {
        preset->apply(config);
    }
    config.prepend_silence_seconds = 0.0;
    if (const char* verbose = std::getenv("BEATIT_VERBOSE"); verbose && verbose[0] != '\0') {
        config.verbose = true;
    }

    std::filesystem::path test_root;
#if defined(BEATIT_TEST_DATA_DIR)
    test_root = std::filesystem::path(BEATIT_TEST_DATA_DIR);
#else
    test_root = std::filesystem::current_path();
#endif

    const char* env_model_path = std::getenv("BEATIT_TORCH_MODEL_PATH");
    const std::string fallback_model = (test_root / "models" / "beatthis.pt").string();
    const std::string model_path =
        (env_model_path && env_model_path[0] != '\0') ? env_model_path : fallback_model;
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "SKIP: Torch model missing (set BEATIT_TORCH_MODEL_PATH).\n";
        return 77;
    }

    config.backend = beatit::CoreMLConfig::Backend::Torch;
    config.torch_model_path = model_path;
    config.use_dbn = true;
    config.dbn_use_downbeat = true;

    const std::vector<CanonicalCase> cases = {
        {"manucho", "manucho.wav", 110.0, 0.040},
        {"moderat", "moderat.wav", 124.0, 0.060},
        {"samerano", "samerano.wav", 122.0, 0.200},
        {"purelove", "purelove.wav", 104.0, 0.320},
    };

    const std::filesystem::path training_dir = test_root / "training";

    for (const auto& entry : cases) {
        beatit::AnalysisResult result;
        std::string error;
        double input_sample_rate = 0.0;
        const std::filesystem::path audio_path = training_dir / entry.filename;
        if (!std::filesystem::exists(audio_path)) {
            std::cerr << "SKIP: missing " << audio_path.string() << "\n";
            continue;
        }
        if (!stream_audio_to_beatit(audio_path.string(), config, &result, &input_sample_rate, &error)) {
            std::cerr << "Failed to analyze " << entry.name << ": " << error << "\n";
            continue;
        }

        const double bpm_delta = result.estimated_bpm - entry.bpm;
        const double downbeat_s = downbeat_seconds_from_result(result, config);
        const double downbeat_delta = downbeat_s - entry.downbeat_seconds;
        const double downbeat_frame =
            (input_sample_rate > 0.0 && downbeat_s >= 0.0) ? downbeat_s * input_sample_rate : -1.0;

        std::cout << entry.name
                  << " bpm=" << result.estimated_bpm
                  << " bpm_target=" << entry.bpm
                  << " bpm_delta=" << bpm_delta
                  << " downbeat_s=" << downbeat_s
                  << " downbeat_target_s=" << entry.downbeat_seconds
                  << " downbeat_delta_s=" << downbeat_delta
                  << " downbeat_frame=" << downbeat_frame
                  << "\n";
    }

    return 0;
}
