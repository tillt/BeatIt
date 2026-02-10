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
#include "beatit/coreml_preset.h"
#include "beatit/refiner.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
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
    std::string ml_backend = "coreml";
    std::string beatthis_python;
    std::string beatthis_script;
    std::string beatthis_checkpoint;
    bool beatthis_use_dbn = false;
    float beatthis_fps = 100.0f;
    std::string torch_model_path;
    std::string torch_device;
    float torch_fps = 100.0f;
    std::string coreml_model_path;
    std::string coreml_input_name;
    std::string coreml_beat_name;
    std::string coreml_downbeat_name;
    std::size_t coreml_sample_rate = 44100;
    std::size_t coreml_frame_size = 2048;
    std::size_t coreml_hop_size = 441;
    std::size_t coreml_mel_bins = 81;
    std::size_t coreml_window_hop_frames = 0;
    float coreml_min_bpm = 55.0f;
    float coreml_max_bpm = 215.0f;
    float coreml_threshold = 0.5f;
    float coreml_gap_tolerance = 0.05f;
    float coreml_offbeat_tolerance = 0.10f;
    std::string coreml_preset = "beatthis";
    float coreml_tempo_window = 20.0f;
    bool coreml_prefer_double_time = true;
    bool coreml_use_dbn = false;
    float coreml_dbn_bpm_step = 1.0f;
    float coreml_dbn_interval_tolerance = 0.05f;
    float coreml_dbn_activation_floor = 0.01f;
    std::size_t coreml_dbn_beats_per_bar = 4;
    bool coreml_dbn_use_downbeat = true;
    float coreml_dbn_downbeat_weight = 1.0f;
    std::string coreml_dbn_mode = "calmdad";
    float coreml_dbn_tempo_change_penalty = 0.05f;
    float coreml_dbn_tempo_prior_weight = 0.0f;
    std::size_t coreml_dbn_max_candidates = 4096;
    float coreml_dbn_transition_reward = 0.7f;
    bool coreml_dbn_use_all_candidates = true;
    bool coreml_dbn_trace = false;
    bool coreml_disable_dbn = false;
    double coreml_dbn_window_seconds = 0.0;
    bool coreml_dbn_window_best = false;
    bool coreml_dbn_grid_align_downbeat_peak = false;
    bool coreml_dbn_grid_strong_start = false;
    bool coreml_dbn_project_grid = false;
    bool coreml_dbn_window_intro_mid_outro = false;
    bool coreml_dbn_window_consensus = false;
    bool coreml_dbn_window_stitch = false;
    double coreml_output_latency_seconds = 0.0;
    double coreml_debug_activations_start_s = -1.0;
    double coreml_debug_activations_end_s = -1.0;
    std::size_t coreml_debug_activations_max = 0;
    bool coreml_constant_refine = false;
    bool coreml_refine_csv = false;
    bool coreml_info = false;
    bool coreml_refine_downbeat_anchor = false;
    bool coreml_cpu_only = false;
    bool coreml_refine_halfbeat = false;
    float coreml_refine_lowfreq = -1.0f;
    bool coreml_verbose = false;
};

void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " --input <path> [options]\n\n"
        << "Options:\n"
        << "  -i, --input <path>        Audio file (MP3/MP4/WAV/AIFF/CAF)\n"
        << "  --sample-rate <hz>        Resample to target sample rate\n"
        << "  --ml-backend <name>       ML backend (coreml, torch, beatthis[external])\n"
        << "  --beatthis-python <path>  External BeatThis python interpreter\n"
        << "  --beatthis-script <path>  External BeatThis inference script\n"
        << "  --beatthis-checkpoint <p> External BeatThis checkpoint path\n"
        << "  --beatthis-dbn            External BeatThis enable DBN postprocess\n"
        << "  --beatthis-fps <hz>       External BeatThis activation fps (default 100)\n"
        << "  --torch-model <path>      TorchScript model path\n"
        << "  --torch-device <name>     Torch device (cpu, mps)\n"
        << "  --torch-fps <hz>          Torch output fps (default 100)\n"
        << "  --model <path>            CoreML model path (.mlmodelc)\n"
        << "  --ml-input <name>         CoreML input feature name\n"
        << "  --ml-beat <name>          CoreML beat output name\n"
        << "  --ml-downbeat <name>      CoreML downbeat output name\n"
        << "  --ml-sr <hz>              CoreML feature sample rate\n"
        << "  --ml-frame <frames>       CoreML feature frame size\n"
        << "  --ml-hop <frames>         CoreML feature hop size\n"
        << "  --ml-mels <bins>          CoreML mel bin count\n"
        << "  --ml-window-hop <frames>  CoreML window hop (fixed-frame mode)\n"
        << "  --ml-min-bpm <bpm>        CoreML min BPM\n"
        << "  --ml-max-bpm <bpm>        CoreML max BPM\n"
        << "  --ml-threshold <value>    CoreML activation threshold\n"
        << "  --ml-gap <ratio>          CoreML gap tolerance (0-1)\n"
        << "  --ml-offbeat <ratio>      CoreML offbeat tolerance (0-1)\n"
        << "  --ml-preset <name>        CoreML preset (beattrack, beatthis)\n"
        << "  --ml-beattrack            Alias for --ml-preset beattrack\n"
        << "  --ml-beatthis             Alias for --ml-preset beatthis\n"
        << "  --ml-tempo-window <pct>   Percent window around classic BPM\n"
        << "  --ml-prefer-double        Prefer double-time BPM if stronger\n"
        << "  --ml-no-prefer-double     Disable double-time BPM preference\n"
        << "  --ml-dbn                  Use DBN-style beat decoder\n"
        << "  --ml-no-dbn               Disable DBN even if preset enables it\n"
        << "  --ml-dbn-step <bpm>       DBN BPM step size\n"
        << "  --ml-dbn-tol <ratio>      DBN interval tolerance (0-1)\n"
        << "  --ml-dbn-floor <value>    DBN activation floor\n"
        << "  --ml-dbn-bpb <beats>      DBN beats-per-bar (default 4)\n"
        << "  --ml-dbn-no-downbeat      DBN ignore downbeat activations\n"
        << "  --ml-dbn-downbeat <w>     DBN downbeat weight\n"
        << "  --ml-dbn-mode <name>      DBN mode (beatit, calmdad)\n"
        << "  --ml-dbn-tempo-pen <w>    DBN tempo change penalty per BPM\n"
        << "  --ml-dbn-tempo-prior <w>  DBN tempo prior weight\n"
        << "  --ml-dbn-max-cand <n>     DBN max candidate frames\n"
        << "  --ml-dbn-reward <w>       DBN transition reward\n"
        << "  --ml-dbn-all-cand         DBN use every frame as candidate\n"
        << "  --ml-dbn-trace            DBN detailed trace logging\n"
        << "  --ml-dbn-window <sec>     DBN run only on best <sec> window\n"
        << "  --ml-dbn-window-best      Select best window by activation energy\n"
        << "  --ml-dbn-grid-align-downbeat Align grid start to earliest downbeat peak\n"
        << "  --ml-dbn-grid-strong-start Use strongest early beat peak as grid anchor\n"
        << "  --ml-dbn-project-grid     Project DBN window beats across full track\n"
        << "  --ml-dbn-no-project-grid  Disable grid projection\n"
        << "  --ml-dbn-imo              Use intro/mid/outro window selection\n"
        << "  --ml-dbn-consensus        Use multi-window consensus BPM\n"
        << "  --ml-dbn-stitch           Stitch intro/mid/outro candidates into one DBN\n"
        << "  --ml-output-latency <sec> Subtract output latency from detected events\n"
        << "  --ml-activations-window <start> <end>\n"
        << "                           Dump beat/downbeat activations between seconds\n"
        << "  --ml-activations-max <n>  Cap activation dump rows (0 = no cap)\n"
        << "  --ml-refine-constant      Post-process beats into a constant grid\n"
        << "  --ml-refine-csv           Print CSV for constant beat events\n"
        << "  --ml-refine-downbeat      Use model downbeats to anchor bar phase\n"
        << "  --ml-refine-halfbeat      Enable half-beat phase correction\n"
        << "  --ml-refine-lowfreq <w>   Low-frequency weight for bar phase\n"
        << "  --ml-verbose              Enable verbose ML diagnostics\n"
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
        } else if (arg == "--ml-backend") {
            options->ml_backend = require_value(arg.c_str());
        } else if (arg == "--beatthis-python") {
            options->beatthis_python = require_value(arg.c_str());
        } else if (arg == "--beatthis-script") {
            options->beatthis_script = require_value(arg.c_str());
        } else if (arg == "--beatthis-checkpoint") {
            options->beatthis_checkpoint = require_value(arg.c_str());
        } else if (arg == "--beatthis-dbn") {
            options->beatthis_use_dbn = true;
        } else if (arg == "--beatthis-fps") {
            float parsed = 0.0f;
            const std::string value = require_value(arg.c_str());
            if (!parse_float(value, &parsed) || parsed <= 0.0f) {
                std::cerr << "Invalid --beatthis-fps: " << value << "\n";
                return false;
            }
            options->beatthis_fps = parsed;
        } else if (arg == "--torch-model") {
            options->torch_model_path = require_value(arg.c_str());
        } else if (arg == "--torch-device") {
            options->torch_device = require_value(arg.c_str());
        } else if (arg == "--torch-fps") {
            float parsed = 0.0f;
            const std::string value = require_value(arg.c_str());
            if (!parse_float(value, &parsed) || parsed <= 0.0f) {
                std::cerr << "Invalid --torch-fps: " << value << "\n";
                return false;
            }
            options->torch_fps = parsed;
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
        } else if (arg == "--ml-window-hop") {
            std::string value = require_value(arg.c_str());
            if (!parse_size_t(value, &options->coreml_window_hop_frames)) {
                std::cerr << "Invalid ml window hop: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-min-bpm") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_min_bpm)) {
                std::cerr << "Invalid ml min bpm: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-max-bpm") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_max_bpm)) {
                std::cerr << "Invalid ml max bpm: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-threshold") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_threshold)) {
                std::cerr << "Invalid ml threshold: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-gap") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_gap_tolerance)) {
                std::cerr << "Invalid ml gap tolerance: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-offbeat") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_offbeat_tolerance)) {
                std::cerr << "Invalid ml offbeat tolerance: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-preset") {
            options->coreml_preset = require_value(arg.c_str());
        } else if (arg == "--ml-beattrack") {
            options->coreml_preset = "beattrack";
        } else if (arg == "--ml-beatthis") {
            options->coreml_preset = "beatthis";
        } else if (arg == "--ml-tempo-window") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_tempo_window)) {
                std::cerr << "Invalid ml tempo window: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-prefer-double") {
            options->coreml_prefer_double_time = true;
        } else if (arg == "--ml-no-prefer-double") {
            options->coreml_prefer_double_time = false;
        } else if (arg == "--ml-dbn") {
            options->coreml_use_dbn = true;
        } else if (arg == "--ml-no-dbn") {
            options->coreml_disable_dbn = true;
            options->coreml_use_dbn = false;
            options->beatthis_use_dbn = false;
        } else if (arg == "--ml-dbn-step") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_dbn_bpm_step)) {
                std::cerr << "Invalid ml dbn step: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-tol") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_dbn_interval_tolerance)) {
                std::cerr << "Invalid ml dbn tolerance: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-floor") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_dbn_activation_floor)) {
                std::cerr << "Invalid ml dbn activation floor: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-bpb") {
            std::string value = require_value(arg.c_str());
            if (!parse_size_t(value, &options->coreml_dbn_beats_per_bar)) {
                std::cerr << "Invalid ml dbn beats per bar: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-no-downbeat") {
            options->coreml_dbn_use_downbeat = false;
        } else if (arg == "--ml-dbn-downbeat") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_dbn_downbeat_weight)) {
                std::cerr << "Invalid ml dbn downbeat weight: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-mode") {
            options->coreml_dbn_mode = require_value(arg.c_str());
        } else if (arg == "--ml-dbn-tempo-pen") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_dbn_tempo_change_penalty)) {
                std::cerr << "Invalid ml dbn tempo penalty: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-tempo-prior") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_dbn_tempo_prior_weight)) {
                std::cerr << "Invalid ml dbn tempo prior: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-max-cand") {
            std::string value = require_value(arg.c_str());
            if (!parse_size_t(value, &options->coreml_dbn_max_candidates)) {
                std::cerr << "Invalid ml dbn max candidates: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-reward") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_dbn_transition_reward)) {
                std::cerr << "Invalid ml dbn transition reward: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-all-cand") {
            options->coreml_dbn_use_all_candidates = true;
        } else if (arg == "--ml-dbn-trace") {
            options->coreml_dbn_trace = true;
        } else if (arg == "--ml-dbn-window") {
            const std::string value = require_value(arg.c_str());
            if (!parse_double(value, &options->coreml_dbn_window_seconds)) {
                std::cerr << "Invalid ml dbn window: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-dbn-window-best") {
            options->coreml_dbn_window_best = true;
        } else if (arg == "--ml-dbn-grid-align-downbeat") {
            options->coreml_dbn_grid_align_downbeat_peak = true;
        } else if (arg == "--ml-dbn-grid-strong-start") {
            options->coreml_dbn_grid_strong_start = true;
        } else if (arg == "--ml-dbn-project-grid") {
            options->coreml_dbn_project_grid = true;
        } else if (arg == "--ml-dbn-no-project-grid") {
            options->coreml_dbn_project_grid = false;
        } else if (arg == "--ml-dbn-imo") {
            options->coreml_dbn_window_intro_mid_outro = true;
        } else if (arg == "--ml-dbn-consensus") {
            options->coreml_dbn_window_consensus = true;
        } else if (arg == "--ml-dbn-stitch") {
            options->coreml_dbn_window_stitch = true;
        } else if (arg == "--ml-output-latency") {
            std::string value = require_value(arg.c_str());
            if (!parse_double(value, &options->coreml_output_latency_seconds)) {
                std::cerr << "Invalid ml output latency: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-activations-window") {
            std::string start = require_value(arg.c_str());
            std::string end = require_value(arg.c_str());
            try {
                options->coreml_debug_activations_start_s = std::stod(start);
                options->coreml_debug_activations_end_s = std::stod(end);
            } catch (...) {
                std::cerr << "Invalid ml activations window: " << start
                          << " " << end << "\n";
                return false;
            }
        } else if (arg == "--ml-activations-max") {
            std::string value = require_value(arg.c_str());
            if (!parse_size_t(value, &options->coreml_debug_activations_max)) {
                std::cerr << "Invalid ml activations max: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-refine-constant") {
            options->coreml_constant_refine = true;
        } else if (arg == "--ml-refine-csv") {
            options->coreml_refine_csv = true;
        } else if (arg == "--ml-refine-downbeat") {
            options->coreml_refine_downbeat_anchor = true;
        } else if (arg == "--ml-refine-halfbeat") {
            options->coreml_refine_halfbeat = true;
        } else if (arg == "--ml-refine-lowfreq") {
            std::string value = require_value(arg.c_str());
            if (!parse_float(value, &options->coreml_refine_lowfreq)) {
                std::cerr << "Invalid ml refine lowfreq weight: " << value << "\n";
                return false;
            }
        } else if (arg == "--ml-verbose") {
            options->coreml_verbose = true;
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
    if (options.ml_backend == "beatthis") {
        ml_config.backend = beatit::CoreMLConfig::Backend::BeatThisExternal;
    } else if (options.ml_backend == "torch") {
        ml_config.backend = beatit::CoreMLConfig::Backend::Torch;
    } else if (options.ml_backend != "coreml") {
        std::cerr << "Unknown --ml-backend: " << options.ml_backend << "\n";
        return 1;
    }
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
    ml_config.min_bpm = options.coreml_min_bpm;
    ml_config.max_bpm = options.coreml_max_bpm;
    ml_config.activation_threshold = options.coreml_threshold;
    ml_config.gap_tolerance = options.coreml_gap_tolerance;
    ml_config.offbeat_tolerance = options.coreml_offbeat_tolerance;
    ml_config.use_dbn = options.coreml_use_dbn;
    ml_config.dbn_bpm_step = options.coreml_dbn_bpm_step;
    ml_config.dbn_interval_tolerance = options.coreml_dbn_interval_tolerance;
    ml_config.dbn_activation_floor = options.coreml_dbn_activation_floor;
    ml_config.dbn_beats_per_bar = options.coreml_dbn_beats_per_bar;
    ml_config.dbn_use_downbeat = options.coreml_dbn_use_downbeat;
    ml_config.dbn_downbeat_weight = options.coreml_dbn_downbeat_weight;
    if (options.coreml_dbn_mode == "beatit") {
        ml_config.dbn_mode = beatit::CoreMLConfig::DBNMode::Beatit;
    } else if (options.coreml_dbn_mode == "calmdad") {
        ml_config.dbn_mode = beatit::CoreMLConfig::DBNMode::Calmdad;
    } else {
        std::cerr << "Invalid ml dbn mode: " << options.coreml_dbn_mode << "\n";
        return 1;
    }
    ml_config.dbn_tempo_change_penalty = options.coreml_dbn_tempo_change_penalty;
    ml_config.dbn_tempo_prior_weight = options.coreml_dbn_tempo_prior_weight;
    ml_config.dbn_max_candidates = options.coreml_dbn_max_candidates;
    ml_config.dbn_transition_reward = options.coreml_dbn_transition_reward;
    ml_config.dbn_use_all_candidates = options.coreml_dbn_use_all_candidates;
    ml_config.dbn_trace = options.coreml_dbn_trace;
    ml_config.dbn_window_seconds = options.coreml_dbn_window_seconds;
    ml_config.dbn_window_best = options.coreml_dbn_window_best;
    ml_config.dbn_grid_align_downbeat_peak = options.coreml_dbn_grid_align_downbeat_peak;
    ml_config.dbn_grid_start_strong_peak = options.coreml_dbn_grid_strong_start;
    ml_config.dbn_project_grid = options.coreml_dbn_project_grid;
    ml_config.dbn_window_intro_mid_outro = options.coreml_dbn_window_intro_mid_outro;
    ml_config.dbn_window_consensus = options.coreml_dbn_window_consensus;
    ml_config.dbn_window_stitch = options.coreml_dbn_window_stitch;
    ml_config.output_latency_seconds = options.coreml_output_latency_seconds;
    ml_config.debug_activations_start_s = options.coreml_debug_activations_start_s;
    ml_config.debug_activations_end_s = options.coreml_debug_activations_end_s;
    ml_config.debug_activations_max = options.coreml_debug_activations_max;
    ml_config.verbose = options.coreml_verbose || options.info;
    ml_config.tempo_window_percent = options.coreml_tempo_window;
    ml_config.prefer_double_time = options.coreml_prefer_double_time;
    if (options.coreml_cpu_only) {
        ml_config.compute_units = beatit::CoreMLConfig::ComputeUnits::CPUOnly;
    }

    if (!options.coreml_preset.empty()) {
        std::unique_ptr<beatit::CoreMLPreset> preset =
            beatit::make_coreml_preset(options.coreml_preset);
        if (!preset) {
            std::cerr << "Unknown CoreML preset: " << options.coreml_preset << "\n";
            return 1;
        }
        if (ml_config.backend != beatit::CoreMLConfig::Backend::BeatThisExternal) {
            preset->apply(ml_config);
        }
    }
    if (options.coreml_disable_dbn) {
        ml_config.use_dbn = false;
        ml_config.dbn_use_downbeat = false;
    }
    if (options.coreml_window_hop_frames > 0) {
        ml_config.window_hop_frames = options.coreml_window_hop_frames;
    }

    if (options.coreml_info && ml_config.backend == beatit::CoreMLConfig::Backend::CoreML) {
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
    if (ml_config.backend == beatit::CoreMLConfig::Backend::BeatThisExternal) {
        if (!options.beatthis_python.empty()) {
            ml_config.beatthis_python = options.beatthis_python;
        }
        ml_config.beatthis_script = options.beatthis_script;
        ml_config.beatthis_checkpoint = options.beatthis_checkpoint;
        ml_config.beatthis_use_dbn = options.beatthis_use_dbn;
        ml_config.beatthis_fps = options.beatthis_fps;

        if (ml_config.beatthis_script.empty()) {
            const std::filesystem::path fallback =
                std::filesystem::current_path() / "training" / "beatthis_infer.py";
            if (std::filesystem::exists(fallback)) {
                ml_config.beatthis_script = fallback.string();
            }
        }
        if (ml_config.beatthis_checkpoint.empty()) {
            const std::filesystem::path fallback =
                std::filesystem::current_path() / "third_party" / "beat_this" / "beat_this-final0.ckpt";
            if (std::filesystem::exists(fallback)) {
                ml_config.beatthis_checkpoint = fallback.string();
            }
        }

        if (ml_config.beatthis_script.empty() || ml_config.beatthis_checkpoint.empty()) {
            std::cerr << "BeatThis backend requires --beatthis-script and --beatthis-checkpoint.\n";
            return 1;
        }
    }
    if (ml_config.backend == beatit::CoreMLConfig::Backend::Torch) {
        ml_config.torch_model_path = options.torch_model_path;
        if (!options.torch_device.empty()) {
            ml_config.torch_device = options.torch_device;
        }
        ml_config.torch_fps = options.torch_fps;
        if (ml_config.torch_model_path.empty()) {
            std::cerr << "Torch backend requires --torch-model.\n";
            return 1;
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
        if (ml_config.backend == beatit::CoreMLConfig::Backend::BeatThisExternal) {
            std::cout << "BeatThis: no output (backend missing or incompatible).\n";
        } else if (ml_config.backend == beatit::CoreMLConfig::Backend::Torch) {
            std::cout << "Torch: no output (backend missing or incompatible).\n";
        } else {
            std::cout << "CoreML: no output (model missing or incompatible).\n";
        }
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

    const char* backend_label = "CoreML";
    if (ml_config.backend == beatit::CoreMLConfig::Backend::BeatThisExternal) {
        backend_label = "BeatThis";
    } else if (ml_config.backend == beatit::CoreMLConfig::Backend::Torch) {
        backend_label = "Torch";
    }

    const double anchor_peaks =
        beatit::estimate_bpm_from_activation(result.coreml_beat_activation,
                                             ml_config,
                                             audio.sample_rate);
    const double anchor_autocorr =
        beatit::estimate_bpm_from_activation_autocorr(result.coreml_beat_activation,
                                                      ml_config,
                                                      audio.sample_rate);
    const double anchor_comb =
        beatit::estimate_bpm_from_activation_comb(result.coreml_beat_activation,
                                                  ml_config,
                                                  audio.sample_rate);
    const double anchor_beats =
        beatit::estimate_bpm_from_beats(result.coreml_beat_sample_frames, audio.sample_rate);
    const auto choose_anchor = [&](double peaks,
                                   double autocorr,
                                   double comb,
                                   double beats) {
        const double tol = 0.02;
        auto near = [&](double a, double b) {
            if (a <= 0.0 || b <= 0.0) {
                return false;
            }
            return (std::abs(a - b) / std::max(a, 1e-6)) <= tol;
        };
        if (near(peaks, comb)) {
            return 0.5 * (peaks + comb);
        }
        if (near(peaks, autocorr)) {
            return 0.5 * (peaks + autocorr);
        }
        if (near(comb, autocorr)) {
            return 0.5 * (comb + autocorr);
        }
        if (peaks > 0.0) {
            return peaks;
        }
        if (comb > 0.0) {
            return comb;
        }
        if (autocorr > 0.0) {
            return autocorr;
        }
        return beats;
    };
    const double anchor_chosen =
        choose_anchor(anchor_peaks, anchor_autocorr, anchor_comb, anchor_beats);
    std::cout << "Tempo anchor: peaks=" << anchor_peaks
              << " autocorr=" << anchor_autocorr
              << " comb=" << anchor_comb
              << " beats=" << anchor_beats
              << " chosen=" << anchor_chosen << "\n";
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

    std::cout << backend_label << " raw: "
              << "beats=" << result.coreml_beat_feature_frames.size()
              << " downbeats=" << result.coreml_downbeat_feature_frames.size()
              << " activations=" << result.coreml_beat_activation.size()
              << " threshold=" << ml_config.activation_threshold
              << " window_hop=" << ml_config.window_hop_frames
              << " fixed_frames=" << ml_config.fixed_frames
              << "\n";

    if (!result.coreml_downbeat_feature_frames.empty()) {
        const auto& beat_frames = result.coreml_beat_feature_frames;
        const auto& sample_frames = result.coreml_beat_sample_frames;
        const auto& downbeat_frames = result.coreml_downbeat_feature_frames;

        const unsigned long long downbeat_feature_frame = downbeat_frames.front();
        double fps = 0.0;
        if (ml_config.backend == beatit::CoreMLConfig::Backend::Torch
            || ml_config.backend == beatit::CoreMLConfig::Backend::BeatThisExternal) {
            fps = ml_config.torch_fps;
        } else if (ml_config.hop_size > 0 && ml_config.sample_rate > 0) {
            fps = static_cast<double>(ml_config.sample_rate) / ml_config.hop_size;
        }

        double downbeat_s = 0.0;
        if (fps > 0.0) {
            downbeat_s = static_cast<double>(downbeat_feature_frame) / fps;
        }

        std::cout << "First downbeat feature_frame: " << downbeat_feature_frame
                  << " (s " << downbeat_s << ")";

        if (!beat_frames.empty()) {
            const auto first_downbeat = std::find_first_of(
                beat_frames.begin(),
                beat_frames.end(),
                downbeat_frames.begin(),
                downbeat_frames.end());

            if (first_downbeat != beat_frames.end()) {
                const std::size_t idx =
                    static_cast<std::size_t>(std::distance(beat_frames.begin(), first_downbeat));
                const unsigned long long sample_frame = sample_frames[idx];
                std::cout << " (sample_frame " << sample_frame << ")";
            }
        }

        std::cout << "\n";

        if (ml_config.verbose) {
            std::unordered_map<unsigned long long, unsigned long long> feature_to_sample;
            feature_to_sample.reserve(beat_frames.size());
            for (std::size_t i = 0; i < beat_frames.size(); ++i) {
                feature_to_sample.emplace(beat_frames[i], sample_frames[i]);
            }

            std::cout << "Beats (all sample frames):";
            for (const auto beat_frame : beat_frames) {
                const auto it = feature_to_sample.find(beat_frame);
                if (it == feature_to_sample.end()) {
                    std::cout << " " << beat_frame;
                    continue;
                }
                const auto sample_frame = it->second;
                double seconds = 0.0;
                if (audio.sample_rate > 0.0) {
                    seconds = static_cast<double>(sample_frame) / audio.sample_rate;
                }
                std::cout << " " << sample_frame << "(" << seconds << ")";
            }
            std::cout << "\n";

            std::cout << "Downbeats (all sample frames):";
            for (const auto downbeat_frame : downbeat_frames) {
                const auto it = feature_to_sample.find(downbeat_frame);
                if (it == feature_to_sample.end()) {
                    std::cout << " " << downbeat_frame;
                    continue;
                }
                const auto sample_frame = it->second;
                double seconds = 0.0;
                if (audio.sample_rate > 0.0) {
                    seconds = static_cast<double>(sample_frame) / audio.sample_rate;
                }
                std::cout << " " << sample_frame << "(" << seconds << ")";
            }
            std::cout << "\n";
        }
    }

    if (!result.coreml_beat_activation.empty()) {
        std::size_t above_threshold = 0;
        float max_activation = result.coreml_beat_activation.front();
        double sum_activation = 0.0;
        std::size_t first_peak_idx = result.coreml_beat_activation.size();
        std::size_t first_peak_sample = 0;
        std::size_t first_floor_idx = result.coreml_beat_activation.size();
        std::size_t first_floor_sample = 0;
        const float floor_threshold = 0.01f;
        float max_activation_first2s = 0.0f;
        for (float value : result.coreml_beat_activation) {
            if (value >= ml_config.activation_threshold) {
                ++above_threshold;
            }
            max_activation = std::max(max_activation, value);
            sum_activation += value;
        }

        const double hop_scale = ml_config.sample_rate > 0.0
            ? (audio.sample_rate / static_cast<double>(ml_config.sample_rate))
            : 1.0;

        if (ml_config.verbose) {
            std::vector<std::size_t> peaks;
            if (result.coreml_beat_activation.size() >= 3) {
                for (std::size_t i = 1; i + 1 < result.coreml_beat_activation.size(); ++i) {
                    const float prev = result.coreml_beat_activation[i - 1];
                    const float curr = result.coreml_beat_activation[i];
                    const float next = result.coreml_beat_activation[i + 1];
                    if (curr >= ml_config.activation_threshold && curr >= prev && curr >= next) {
                        peaks.push_back(i);
                    }
                    if (curr >= floor_threshold && first_floor_idx == result.coreml_beat_activation.size()) {
                        first_floor_idx = i;
                    }
                }
            }
            if (!peaks.empty()) {
                first_peak_idx = peaks.front();
                const double sample_pos =
                    static_cast<double>(first_peak_idx * ml_config.hop_size) * hop_scale;
                first_peak_sample = static_cast<unsigned long long>(std::llround(sample_pos));
            }
            if (first_floor_idx < result.coreml_beat_activation.size()) {
                const double sample_pos =
                    static_cast<double>(first_floor_idx * ml_config.hop_size) * hop_scale;
                first_floor_sample = static_cast<unsigned long long>(std::llround(sample_pos));
            }

            const std::size_t max_samples =
                static_cast<std::size_t>(std::max(0.0, audio.sample_rate * 2.0));
            for (std::size_t i = 0; i < result.coreml_beat_activation.size(); ++i) {
                const double sample_pos =
                    static_cast<double>(i * ml_config.hop_size) * hop_scale;
                if (sample_pos > static_cast<double>(max_samples)) {
                    continue;
                }
                max_activation_first2s =
                    std::max(max_activation_first2s, result.coreml_beat_activation[i]);
            }
        }

        const double mean_activation = sum_activation / result.coreml_beat_activation.size();
        std::cout << backend_label << " activation stats: "
                  << "mean=" << mean_activation
                  << " max=" << max_activation
                  << " above_threshold=" << above_threshold
                  << "\n";
        if (ml_config.verbose && first_peak_idx < result.coreml_beat_activation.size()) {
            double first_peak_s = 0.0;
            if (audio.sample_rate > 0.0) {
                first_peak_s = static_cast<double>(first_peak_sample) / audio.sample_rate;
            }
            std::cout << backend_label << " first beat peak: "
                      << "activation_idx=" << first_peak_idx
                      << " sample_frame=" << first_peak_sample
                      << " (s " << first_peak_s << ")\n";
        }
        if (ml_config.verbose && first_floor_idx < result.coreml_beat_activation.size()) {
            double first_floor_s = 0.0;
            if (audio.sample_rate > 0.0) {
                first_floor_s = static_cast<double>(first_floor_sample) / audio.sample_rate;
            }
            std::cout << backend_label << " first beat floor: "
                      << "activation_idx=" << first_floor_idx
                      << " sample_frame=" << first_floor_sample
                      << " (s " << first_floor_s << ")\n";
            std::cout << backend_label << " max beat activation (first 2s): "
                      << max_activation_first2s << "\n";
        }
        if (ml_config.verbose && !result.coreml_beat_feature_frames.empty()) {
            const std::size_t dump_count =
                std::min<std::size_t>(10, result.coreml_beat_feature_frames.size());
            std::cout << backend_label << " beat frame->sample map (first "
                      << dump_count << "):";
            for (std::size_t i = 0; i < dump_count; ++i) {
                const auto feature_frame = result.coreml_beat_feature_frames[i];
                std::size_t sample_frame = 0;
                double seconds = 0.0;
                if (feature_frame < result.coreml_beat_sample_frames.size()) {
                    sample_frame = result.coreml_beat_sample_frames[feature_frame];
                    if (audio.sample_rate > 0.0) {
                        seconds = static_cast<double>(sample_frame) / audio.sample_rate;
                    }
                }
                std::cout << " [" << i
                          << " ff=" << feature_frame
                          << " sf=" << sample_frame
                          << " s=" << seconds << "]";
            }
            std::cout << "\n";
        }
        if (ml_config.verbose) {
            const std::size_t dump_count =
                std::min<std::size_t>(10, result.coreml_beat_activation.size());
            std::cout << backend_label << " activation->sample map (first "
                      << dump_count << "):";
            for (std::size_t i = 0; i < dump_count; ++i) {
                const double sample_pos =
                    static_cast<double>(i * ml_config.hop_size) * hop_scale;
                const auto sample_frame =
                    static_cast<unsigned long long>(std::llround(sample_pos));
                double seconds = 0.0;
                if (audio.sample_rate > 0.0) {
                    seconds = static_cast<double>(sample_frame) / audio.sample_rate;
                }
                std::cout << " [" << i
                          << " sf=" << sample_frame
                          << " s=" << seconds << "]";
            }
            std::cout << "\n";
        }
    }

    if (result.coreml_beat_sample_frames.size() > 1) {
        print_bpm_stats(result.coreml_beat_sample_frames, backend_label);
    }

    if (options.coreml_constant_refine) {
        beatit::ConstantBeatRefinerConfig refiner_config;
        refiner_config.use_downbeat_anchor = options.coreml_refine_downbeat_anchor;
        refiner_config.use_half_beat_correction = options.coreml_refine_halfbeat;
        if (options.coreml_refine_lowfreq >= 0.0f) {
            refiner_config.low_freq_weight = options.coreml_refine_lowfreq;
        }
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
                                      refined.phase_shift_frames,
                                      refined.used_model_downbeat,
                                      refined.downbeat_coverage,
                                      refined.active_start_frame,
                                      refined.active_end_frame,
                                      refined.found_count,
                                      refined.total_count,
                                      refined.max_missing_run,
                                      refined.avg_missing_run);
            }
        }
    }

    return 0;
}
