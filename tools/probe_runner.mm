//
//  probe_runner.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "tests/coreml_test_config.h"
#include "tests/window_alignment_test_utils.h"
#include "beatit/stream.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int usage() {
    std::cerr << "usage: beatit_tool_probe_runner <audio-file> <start-s> <duration-s> [dense|sparse]\n";
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    using namespace beatit::tests::window_alignment;

    if (argc < 4 || argc > 5) {
        return usage();
    }

    const std::filesystem::path audio_path = argv[1];
    const double probe_start = std::stod(argv[2]);
    const double probe_duration = std::stod(argv[3]);
    const bool sparse_mode = argc < 5 || std::string(argv[4]) != "dense";

    std::string source_model_path = beatit::tests::resolve_beatthis_coreml_model_path();
    std::string model_error;
    std::string model_path = compile_model_if_needed(source_model_path, &model_error);
    if (model_path.empty()) {
        std::cerr << model_error << "\n";
        return 2;
    }

    beatit::BeatitConfig config;
    beatit::tests::apply_beatthis_coreml_test_config(config);
    config.model_path = model_path;
    config.use_dbn = true;
    config.max_analysis_seconds = 60.0;
    config.dbn_window_start_seconds = 0.0;
    config.sparse_probe_mode = sparse_mode;
    config.compute_units = beatit::BeatitConfig::ComputeUnits::CPUOnly;
    config.log_verbosity = beatit::LogVerbosity::Debug;
    config.dbn_trace = true;
    config.profile = true;

    std::vector<float> mono;
    double sample_rate = 0.0;
    std::string decode_error;
    if (!decode_audio_mono(audio_path.string(), &mono, &sample_rate, &decode_error)) {
        std::cerr << decode_error << "\n";
        return 3;
    }

    const double total_duration_s = static_cast<double>(mono.size()) / sample_rate;
    auto provider = [&](double start_seconds, double duration_seconds, std::vector<float>* out_samples)
        -> std::size_t {
        out_samples->clear();
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
    const beatit::AnalysisResult result =
        stream.analyze_window(probe_start, probe_duration, total_duration_s, provider);

    auto print_head = [](const char* label, const std::vector<unsigned long long>& frames) {
        std::cout << label << ":";
        for (std::size_t i = 0; i < std::min<std::size_t>(frames.size(), 12); ++i) {
            std::cout << " " << frames[i];
        }
        std::cout << "\n";
    };

    std::cout << "bpm=" << result.estimated_bpm << " sparse=" << (sparse_mode ? 1 : 0) << "\n";
    print_head("beat_feature", result.coreml_beat_feature_frames);
    print_head("beat_projected_feature", result.coreml_beat_projected_feature_frames);
    print_head("downbeat_feature", result.coreml_downbeat_feature_frames);
    print_head("downbeat_projected_feature", result.coreml_downbeat_projected_feature_frames);
    print_head("beat_projected_sample", result.coreml_beat_projected_sample_frames);

    std::cout << "events:";
    for (std::size_t i = 0; i < std::min<std::size_t>(result.coreml_beat_events.size(), 12); ++i) {
        const auto& event = result.coreml_beat_events[i];
        std::cout << " [" << event.frame << ":" << static_cast<unsigned long long>(event.style) << "]";
    }
    std::cout << "\n";
    return 0;
}
