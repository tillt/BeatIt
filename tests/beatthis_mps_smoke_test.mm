#import <AVFoundation/AVFoundation.h>

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "beatit/coreml_preset.h"
#include "beatit/stream.h"

namespace {

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
    }

    return true;
}

}  // namespace

int main() {
#if !defined(BEATIT_USE_TORCH)
    std::cerr << "SKIP: Torch backend not enabled.\n";
    return 77;
#endif
    if (!torch::mps::is_available()) {
        std::cerr << "SKIP: MPS not available in libtorch.\n";
        return 77;
    }

    beatit::CoreMLConfig base_config;
    if (auto preset = beatit::make_coreml_preset("beatthis")) {
        preset->apply(base_config);
    }

    std::filesystem::path test_root;
#if defined(BEATIT_TEST_DATA_DIR)
    test_root = std::filesystem::path(BEATIT_TEST_DATA_DIR);
#else
    test_root = std::filesystem::current_path();
#endif

    const char* env_model_path = std::getenv("BEATIT_TORCH_MODEL_PATH");
    const std::string cpu_model_path =
        (env_model_path && env_model_path[0] != '\0')
            ? env_model_path
            : (test_root / "models" / "beatthis.pt").string();
    const std::string mps_model_path = (test_root / "models" / "beatthis_mps.pt").string();
    if (!std::filesystem::exists(cpu_model_path)) {
        std::cerr << "SKIP: Torch CPU model missing.\n";
        return 77;
    }
    if (!std::filesystem::exists(mps_model_path)) {
        std::cerr << "SKIP: Torch MPS model missing (expected "
                  << mps_model_path << ").\n";
        return 77;
    }

    const std::filesystem::path audio_path = test_root / "training" / "manucho.wav";
    if (!std::filesystem::exists(audio_path)) {
        std::cerr << "SKIP: missing " << audio_path.string() << "\n";
        return 77;
    }

    auto run_case = [&](const std::string& device,
                        beatit::AnalysisResult* result,
                        double* runtime_seconds,
                        std::string* error) -> bool {
        beatit::CoreMLConfig config = base_config;
        config.backend = beatit::CoreMLConfig::Backend::Torch;
        config.torch_model_path = (device == "mps") ? mps_model_path : cpu_model_path;
        config.torch_device = device;
        config.verbose = true;
        config.profile = true;
        config.use_dbn = true;
        config.dbn_use_downbeat = true;
        if (device == "mps") {
            config.mel_backend = beatit::CoreMLConfig::MelBackend::Torch;
        }

        const auto start = std::chrono::steady_clock::now();
        const bool ok = stream_audio_to_beatit(audio_path.string(), config, result, error);
        const auto end = std::chrono::steady_clock::now();
        if (runtime_seconds) {
            *runtime_seconds =
                std::chrono::duration<double>(end - start).count();
        }
        return ok;
    };

    beatit::AnalysisResult cpu_result;
    beatit::AnalysisResult mps_result;
    double cpu_seconds = 0.0;
    double mps_seconds = 0.0;
    std::string error;

    if (!run_case("cpu", &cpu_result, &cpu_seconds, &error)) {
        std::cerr << "BeatThis CPU test failed: " << error << "\n";
        return 1;
    }

    error.clear();
    if (!run_case("mps", &mps_result, &mps_seconds, &error)) {
        std::cerr << "BeatThis MPS test failed: " << error << "\n";
        return 1;
    }

    if (cpu_result.coreml_beat_sample_frames.empty() || cpu_result.estimated_bpm <= 0.0f) {
        std::cerr << "BeatThis CPU produced no beats.\n";
        return 1;
    }
    if (mps_result.coreml_beat_sample_frames.empty() || mps_result.estimated_bpm <= 0.0f) {
        std::cerr << "BeatThis MPS produced no beats.\n";
        return 1;
    }

    const double bpm_delta = mps_result.estimated_bpm - cpu_result.estimated_bpm;
    const long long beat_delta = static_cast<long long>(mps_result.coreml_beat_sample_frames.size()) -
                                 static_cast<long long>(cpu_result.coreml_beat_sample_frames.size());
    std::cout << "BeatThis CPU: bpm=" << cpu_result.estimated_bpm
              << " beats=" << cpu_result.coreml_beat_sample_frames.size()
              << " time=" << cpu_seconds << "s\n";
    std::cout << "BeatThis MPS: bpm=" << mps_result.estimated_bpm
              << " beats=" << mps_result.coreml_beat_sample_frames.size()
              << " time=" << mps_seconds << "s\n";
    std::cout << "BeatThis delta: bpm=" << bpm_delta
              << " beats=" << beat_delta
              << " time=" << (mps_seconds - cpu_seconds) << "s\n";
    return 0;
}
