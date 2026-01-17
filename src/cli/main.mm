//
//  main.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <AudioToolbox/AudioToolbox.h>
#import <Foundation/Foundation.h>

#include "beatit/analysis.h"
#include "beatit/coreml.h"
#include "beatit/refiner.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct AudioData {
    double sample_rate = 0.0;
    std::vector<float> samples;
    std::uint64_t reported_frames = 0;
    bool has_reported_frames = false;
};

struct CliOptions {
    std::string input_path;
    double target_sample_rate = 0.0;
    bool info = false;
    bool show_help = false;
    std::string coreml_model_path;
    std::string coreml_input_name;
    std::string coreml_beat_name;
    std::string coreml_downbeat_name;
    std::size_t coreml_sample_rate = 44100;
    std::size_t coreml_frame_size = 2048;
    std::size_t coreml_hop_size = 441;
    std::size_t coreml_mel_bins = 81;
    float coreml_threshold = 0.5f;
    bool coreml_beattrack = false;
    float coreml_tempo_window = 20.0f;
    bool coreml_prefer_double_time = true;
    bool coreml_constant_refine = false;
    bool coreml_refine_csv = false;
    bool coreml_info = false;
    bool coreml_refine_downbeat_anchor = false;
    bool coreml_cpu_only = false;
    bool coreml_refine_halfbeat = false;
};

void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " --input <path> [options]\n\n"
        << "Options:\n"
        << "  -i, --input <path>        Audio file (MP3/MP4/WAV/AIFF/CAF)\n"
        << "  --sample-rate <hz>        Resample to target sample rate\n"
        << "  --model <path>            CoreML model path (.mlmodelc)\n"
        << "  --ml-input <name>         CoreML input feature name\n"
        << "  --ml-beat <name>          CoreML beat output name\n"
        << "  --ml-downbeat <name>      CoreML downbeat output name\n"
        << "  --ml-sr <hz>              CoreML feature sample rate\n"
        << "  --ml-frame <frames>       CoreML feature frame size\n"
        << "  --ml-hop <frames>         CoreML feature hop size\n"
        << "  --ml-mels <bins>          CoreML mel bin count\n"
        << "  --ml-threshold <value>    CoreML activation threshold\n"
        << "  --ml-beattrack            Use BeatTrack CoreML defaults\n"
        << "  --ml-tempo-window <pct>   Percent window around classic BPM\n"
        << "  --ml-prefer-double        Prefer double-time BPM if stronger\n"
        << "  --ml-refine-constant      Post-process beats into a constant grid\n"
        << "  --ml-refine-csv           Print CSV for constant beat events\n"
        << "  --ml-refine-downbeat      Use model downbeats to anchor bar phase\n"
        << "  --ml-refine-halfbeat      Enable half-beat phase correction\n"
        << "  --ml-cpu-only             Force CoreML CPU-only execution\n"
        << "  --ml-info                 Print CoreML model metadata\n"
        << "  --info                    Print decoded audio stats\n"
        << "  -h, --help                Show this help\n";
}

bool parse_size_t(const std::string& value, std::size_t* output) {
    try {
        std::size_t idx = 0;
        std::size_t parsed = std::stoul(value, &idx);
        if (idx != value.size()) {
            return false;
        }
        *output = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_float(const std::string& value, float* output) {
    try {
        std::size_t idx = 0;
        float parsed = std::stof(value, &idx);
        if (idx != value.size()) {
            return false;
        }
        *output = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string& value, double* output) {
    try {
        std::size_t idx = 0;
        double parsed = std::stod(value, &idx);
        if (idx != value.size()) {
            return false;
        }
        *output = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

std::string os_status_string(OSStatus status) {
    std::ostringstream stream;
    stream << status << " (0x" << std::hex << static_cast<std::uint32_t>(status) << ")";
    return stream.str();
}

bool parse_args(int argc, char** argv, CliOptions* options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            options->show_help = true;
            return false;
        }

        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return {};
            }
            return argv[++i];
        };

        if (arg == "-i" || arg == "--input") {
            options->input_path = require_value(arg.c_str());
        } else if (arg == "--sample-rate") {
            std::string value = require_value(arg.c_str());
            if (!parse_double(value, &options->target_sample_rate)) {
                std::cerr << "Invalid sample rate: " << value << "\n";
                return false;
            }
        } else if (arg == "--model") {
            options->coreml_model_path = require_value(arg.c_str());
        } else if (arg == "--ml-input") {
            options->coreml_input_name = require_value(arg.c_str());
        } else if (arg == "--ml-beat") {
            options->coreml_beat_name = require_value(arg.c_str());
        } else if (arg == "--ml-downbeat") {
            options->coreml_downbeat_name = require_value(arg.c_str());
        } else if (arg == "--ml-sr") {
            std::string value = require_value(arg.c_str());
            std::size_t parsed = 0;
            if (!parse_size_t(value, &parsed)) {
                std::cerr << "Invalid ml sample rate: " << value << "\n";
                return false;
            }
            options->coreml_sample_rate = parsed;
        } else if (arg == "--ml-frame") {
            std::string value = require_value(arg.c_str());
            if (!parse_size_t(value, &options->coreml_frame_size)) {
                std::cerr << "Invalid ml frame size: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-hop") {
            std::string value = require_value(arg.c_str());
            if (!parse_size_t(value, &options->coreml_hop_size)) {
                std::cerr << "Invalid ml hop size: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-mels") {
            std::string value = require_value(arg.c_str());
            if (!parse_size_t(value, &options->coreml_mel_bins)) {
                std::cerr << "Invalid ml mel bins: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-threshold") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_threshold)) {
                std::cerr << "Invalid ml threshold: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-beattrack") {
            options->coreml_beattrack = true;
        } else if (arg == "--ml-tempo-window") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_tempo_window)) {
                std::cerr << "Invalid ml tempo window: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-prefer-double") {
            options->coreml_prefer_double_time = true;
        } else if (arg == "--ml-refine-constant") {
            options->coreml_constant_refine = true;
        } else if (arg == "--ml-refine-csv") {
            options->coreml_refine_csv = true;
        } else if (arg == "--ml-refine-downbeat") {
            options->coreml_refine_downbeat_anchor = true;
        } else if (arg == "--ml-refine-halfbeat") {
            options->coreml_refine_halfbeat = true;
        } else if (arg == "--ml-cpu-only") {
            options->coreml_cpu_only = true;
        } else if (arg == "--ml-info") {
            options->coreml_info = true;
        } else if (arg == "--info") {
            options->info = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }

    if (options->input_path.empty()) {
        std::cerr << "Input path is required.\n";
        print_usage(argv[0]);
        return false;
    }

    return true;
}

bool load_audio_with_extaudio(const std::string& path, double target_sample_rate, AudioData* output) {
    @autoreleasepool {
        NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:ns_path];

        ExtAudioFileRef file = nullptr;
        OSStatus status = ExtAudioFileOpenURL((__bridge CFURLRef)url, &file);
        if (status != noErr || !file) {
            std::cerr << "Failed to open audio file with ExtAudioFile: "
                      << os_status_string(status) << "\n";
            return false;
        }

        AudioStreamBasicDescription file_format{};
        UInt32 format_size = sizeof(file_format);
        status = ExtAudioFileGetProperty(file,
                                         kExtAudioFileProperty_FileDataFormat,
                                         &format_size,
                                         &file_format);
        if (status != noErr) {
            ExtAudioFileDispose(file);
            std::cerr << "Failed to read audio format: " << os_status_string(status) << "\n";
            return false;
        }

        const double sample_rate = target_sample_rate > 0.0 ? target_sample_rate : file_format.mSampleRate;
        const std::uint32_t channels = file_format.mChannelsPerFrame > 0 ? file_format.mChannelsPerFrame : 1;

        SInt64 file_length_frames = 0;
        UInt32 length_size = sizeof(file_length_frames);
        status = ExtAudioFileGetProperty(file,
                                         kExtAudioFileProperty_FileLengthFrames,
                                         &length_size,
                                         &file_length_frames);
        if (status == noErr && file_length_frames > 0) {
            output->reported_frames = static_cast<std::uint64_t>(file_length_frames);
            output->has_reported_frames = true;
        } else {
            output->has_reported_frames = false;
        }

        AudioStreamBasicDescription client_format{};
        client_format.mSampleRate = sample_rate;
        client_format.mFormatID = kAudioFormatLinearPCM;
        client_format.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
        client_format.mBitsPerChannel = 16;
        client_format.mChannelsPerFrame = channels;
        client_format.mFramesPerPacket = 1;
        client_format.mBytesPerFrame = static_cast<UInt32>(channels * sizeof(std::int16_t));
        client_format.mBytesPerPacket = client_format.mBytesPerFrame;

        status = ExtAudioFileSetProperty(file,
                                         kExtAudioFileProperty_ClientDataFormat,
                                         sizeof(client_format),
                                         &client_format);
        if (status != noErr) {
            ExtAudioFileDispose(file);
            std::cerr << "Failed to set client data format: "
                      << os_status_string(status) << "\n";
            std::cerr << "File format sample rate: " << file_format.mSampleRate
                      << ", channels: " << file_format.mChannelsPerFrame << "\n";
            return false;
        }

        const UInt32 frames_per_chunk = 32768;
        std::vector<std::int16_t> buffer(static_cast<std::size_t>(frames_per_chunk) * channels);
        AudioBufferList buffer_list{};
        buffer_list.mNumberBuffers = 1;
        buffer_list.mBuffers[0].mNumberChannels = channels;
        buffer_list.mBuffers[0].mDataByteSize = static_cast<UInt32>(buffer.size() * sizeof(std::int16_t));
        buffer_list.mBuffers[0].mData = buffer.data();

        output->samples.clear();
        output->sample_rate = sample_rate;
        std::size_t write_pos = 0;
        if (output->has_reported_frames) {
            output->samples.resize(static_cast<std::size_t>(output->reported_frames));
        } else {
            output->samples.reserve(static_cast<std::size_t>(frames_per_chunk) * 4);
        }

        while (true) {
            UInt32 frames = frames_per_chunk;
            status = ExtAudioFileRead(file, &frames, &buffer_list);
            if (status != noErr) {
                ExtAudioFileDispose(file);
                std::cerr << "ExtAudioFile read failed: " << os_status_string(status) << "\n";
                return false;
            }
            if (frames == 0) {
                break;
            }

            const float sum_scale = 1.0f / static_cast<float>(channels);
            const std::int16_t* data = static_cast<const std::int16_t*>(buffer_list.mBuffers[0].mData);
            for (UInt32 i = 0; i < frames; ++i) {
                float sum = 0.0f;
                for (std::uint32_t ch = 0; ch < channels; ++ch) {
                    const std::size_t idx = static_cast<std::size_t>(i) * channels + ch;
                    sum += static_cast<float>(data[idx]) / 32768.0f;
                }
                const float value = sum * sum_scale;
                if (output->has_reported_frames) {
                    if (write_pos < output->samples.size()) {
                        output->samples[write_pos++] = value;
                    }
                } else {
                    output->samples.push_back(value);
                    ++write_pos;
                }
            }
        }

        if (output->has_reported_frames && write_pos < output->samples.size()) {
            output->samples.resize(write_pos);
        }

        ExtAudioFileDispose(file);
    }

    return true;
}

void print_audio_info(const AudioData& audio, const std::string& source) {
    if (audio.sample_rate <= 0.0) {
        std::cerr << "Decoded audio (" << source << "): invalid sample rate\n";
        return;
    }
    const double seconds = static_cast<double>(audio.samples.size()) / audio.sample_rate;
    std::cerr << "Decoded audio (" << source << "): " << audio.samples.size()
              << " samples, " << audio.sample_rate << " Hz, "
              << seconds << " sec";
    if (audio.has_reported_frames) {
        const double reported_seconds = static_cast<double>(audio.reported_frames) / audio.sample_rate;
        const double ratio = audio.reported_frames > 0
                                 ? static_cast<double>(audio.samples.size()) /
                                       static_cast<double>(audio.reported_frames)
                                 : 0.0;
        std::cerr << " (reported: " << audio.reported_frames
                  << " frames, " << reported_seconds << " sec, "
                  << "decoded " << static_cast<int>(ratio * 100.0) << "%)";
    }
    std::cerr << "\n";
}

bool load_audio_file(const std::string& path,
                     double target_sample_rate,
                     bool info,
                     AudioData* output) {
    if (!load_audio_with_extaudio(path, target_sample_rate, output)) {
        return false;
    }

    if (info) {
        print_audio_info(*output, "ExtAudioFile");
    }

    return true;
}

} // namespace

int main(int argc, char** argv) {
    CliOptions options;
    if (!parse_args(argc, argv, &options)) {
        return options.show_help ? 0 : 1;
    }

    AudioData audio;
    const auto decode_start = std::chrono::steady_clock::now();
    if (!load_audio_file(options.input_path, options.target_sample_rate, options.info, &audio)) {
        return 1;
    }
    const auto decode_end = std::chrono::steady_clock::now();
    beatit::CoreMLConfig ml_config;
    if (!options.coreml_model_path.empty()) {
        ml_config.model_path = options.coreml_model_path;
    }
    if (!options.coreml_input_name.empty()) {
        ml_config.input_name = options.coreml_input_name;
    }
    if (!options.coreml_beat_name.empty()) {
        ml_config.beat_output_name = options.coreml_beat_name;
    }
    if (!options.coreml_downbeat_name.empty()) {
        ml_config.downbeat_output_name = options.coreml_downbeat_name;
    }
    ml_config.sample_rate = options.coreml_sample_rate;
    ml_config.frame_size = options.coreml_frame_size;
    ml_config.hop_size = options.coreml_hop_size;
    ml_config.mel_bins = options.coreml_mel_bins;
    ml_config.activation_threshold = options.coreml_threshold;
    ml_config.verbose = options.info;
    ml_config.tempo_window_percent = options.coreml_tempo_window;
    ml_config.prefer_double_time = options.coreml_prefer_double_time;
    if (options.coreml_cpu_only) {
        ml_config.compute_units = beatit::CoreMLConfig::ComputeUnits::CPUOnly;
    }

    if (options.coreml_beattrack) {
        ml_config.sample_rate = 44100;
        ml_config.frame_size = 2048;
        ml_config.hop_size = 441;
        ml_config.mel_bins = 81;
        ml_config.use_log_mel = false;
        ml_config.input_layout = beatit::CoreMLConfig::InputLayout::ChannelsFramesMels;
    }

    if (options.coreml_info) {
        const beatit::CoreMLMetadata metadata = beatit::load_coreml_metadata(ml_config);
        std::cout << "CoreML metadata:\n";
        if (!metadata.author.empty()) {
            std::cout << "  Author: " << metadata.author << "\n";
        }
        if (!metadata.short_description.empty()) {
            std::cout << "  Description: " << metadata.short_description << "\n";
        }
        if (!metadata.license.empty()) {
            std::cout << "  License: " << metadata.license << "\n";
        }
        if (!metadata.version.empty()) {
            std::cout << "  Version: " << metadata.version << "\n";
        }
        if (!metadata.user_defined.empty()) {
            std::cout << "  User metadata:\n";
            for (const auto& entry : metadata.user_defined) {
                std::cout << "    " << entry.first << ": " << entry.second << "\n";
            }
        }
    }

    const auto analyze_start = std::chrono::steady_clock::now();
    beatit::AnalysisResult result = beatit::analyze(audio.samples, audio.sample_rate, ml_config);
    const auto analyze_end = std::chrono::steady_clock::now();

    if (options.info) {
        const auto decode_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count();
        const auto analyze_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(analyze_end - analyze_start).count();
        std::cerr << "Timing: decode=" << decode_ms << "ms, analyze=" << analyze_ms << "ms\n";
    }

    if (result.coreml_beat_activation.empty() && result.coreml_downbeat_activation.empty()) {
        std::cout << "CoreML: no output (model missing or incompatible).\n";
        return 0;
    }

    auto print_bpm_stats = [&](const std::vector<unsigned long long>& beats, const char* label) {
        if (beats.size() < 2) {
            return;
        }
        std::vector<double> bpms;
        bpms.reserve(beats.size() - 1);
        for (std::size_t i = 1; i < beats.size(); ++i) {
            const unsigned long long prev = beats[i - 1];
            const unsigned long long next = beats[i];
            if (next > prev) {
                const double interval = static_cast<double>(next - prev) / audio.sample_rate;
                if (interval > 0.0) {
                    bpms.push_back(60.0 / interval);
                }
            }
        }

        if (bpms.empty()) {
            return;
        }

        double sum = 0.0;
        double min_val = bpms.front();
        double max_val = bpms.front();
        for (double value : bpms) {
            sum += value;
            min_val = std::min(min_val, value);
            max_val = std::max(max_val, value);
        }

        const double mean = sum / static_cast<double>(bpms.size());
        double variance = 0.0;
        for (double value : bpms) {
            const double diff = value - mean;
            variance += diff * diff;
        }
        const double stddev = std::sqrt(variance / static_cast<double>(bpms.size()));

        std::cout << label << " BPM stats: "
                  << "mean=" << mean
                  << " stddev=" << stddev
                  << " min=" << min_val
                  << " max=" << max_val
                  << "\n";
    };

    std::cout << "Estimated BPM: " << result.estimated_bpm << "\n";
    std::cout << "Beats (first 64):\n";
    const std::size_t max_beats = std::min<std::size_t>(64, result.coreml_beat_feature_frames.size());
    for (std::size_t i = 0; i < max_beats; ++i) {
        const bool is_downbeat =
            std::find(result.coreml_downbeat_feature_frames.begin(),
                      result.coreml_downbeat_feature_frames.end(),
                      result.coreml_beat_feature_frames[i]) != result.coreml_downbeat_feature_frames.end();
        std::cout << (is_downbeat ? "* " : "  ")
                  << "feature_frame=" << result.coreml_beat_feature_frames[i]
                  << " sample_frame=" << result.coreml_beat_sample_frames[i]
                  << " strength=" << result.coreml_beat_strengths[i]
                  << "\n";
    }

    if (result.coreml_beat_sample_frames.size() > 1) {
        print_bpm_stats(result.coreml_beat_sample_frames, "CoreML");
    }

    if (options.coreml_constant_refine) {
        beatit::ConstantBeatRefinerConfig refiner_config;
        refiner_config.use_downbeat_anchor = options.coreml_refine_downbeat_anchor;
        refiner_config.use_half_beat_correction = options.coreml_refine_halfbeat;
        beatit::ConstantBeatResult refined =
            beatit::refine_constant_beats(result.coreml_beat_feature_frames,
                                          result.coreml_beat_activation.size(),
                                          ml_config,
                                          audio.sample_rate,
                                          0,
                                          refiner_config,
                                          &result.coreml_beat_activation,
                                          &result.coreml_downbeat_activation,
                                          result.coreml_phase_energy.empty()
                                              ? nullptr
                                              : &result.coreml_phase_energy);
        if (refined.beat_feature_frames.empty()) {
            std::cout << "Constant refiner: no output (insufficient beats).\n";
        } else {
            std::cout << "Constant BPM: " << refined.bpm << "\n";
            std::cout << "Constant beats (first 64):\n";
            const std::size_t max_refined =
                std::min<std::size_t>(64, refined.beat_feature_frames.size());
            for (std::size_t i = 0; i < max_refined; ++i) {
                const bool is_downbeat =
                    std::find(refined.downbeat_feature_frames.begin(),
                              refined.downbeat_feature_frames.end(),
                              refined.beat_feature_frames[i]) != refined.downbeat_feature_frames.end();
                std::cout << (is_downbeat ? "* " : "  ")
                          << "feature_frame=" << refined.beat_feature_frames[i]
                          << " sample_frame=" << refined.beat_sample_frames[i]
                          << " strength=" << refined.beat_strengths[i]
                          << "\n";
            }
            if (refined.beat_sample_frames.size() > 1) {
                print_bpm_stats(refined.beat_sample_frames, "Constant");
            }

            if (options.coreml_refine_csv) {
                write_beat_events_csv(std::cout,
                                      refined.beat_events,
                                      true,
                                      refined.downbeat_phase,
                                      refined.phase_shift_frames);
            }
        }
    }

    return 0;
}
