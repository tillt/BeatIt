//
//  cache_benchmark.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "tests/coreml_test_config.h"
#include "tests/window_alignment_test_utils.h"

#include "beatit/analysis.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

int usage() {
    std::cerr << "usage: beatit_tool_cache_benchmark <audio-file>\n";
    return 1;
}

struct TimedRun {
    double milliseconds = 0.0;
    double bpm = 0.0;
    std::size_t beats = 0;
    std::size_t downbeats = 0;
};

TimedRun run_once(const std::vector<float>& mono,
                  double sample_rate,
                  const beatit::BeatitConfig& config) {
    const auto start = std::chrono::steady_clock::now();
    const beatit::AnalysisResult result = beatit::analyze(mono, sample_rate, config);
    const auto end = std::chrono::steady_clock::now();

    TimedRun run;
    run.milliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    run.bpm = result.estimated_bpm;
    run.beats = beatit::output_beat_sample_frames(result).size();
    run.downbeats = beatit::output_downbeat_feature_frames(result).size();
    return run;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return usage();
    }

    const std::filesystem::path audio_path = argv[1];

    std::string model_path = beatit::tests::resolve_beatthis_coreml_model_path();
    if (model_path.empty()) {
        std::cerr << "Could not locate BeatThis CoreML model package.\n";
        return 2;
    }

    beatit::BeatitConfig config;
    beatit::tests::apply_beatthis_coreml_test_config(config);
    config.model_path = model_path;
    config.use_dbn = true;
    config.sparse_probe_mode = true;
    config.max_analysis_seconds = 60.0;
    config.compute_units = beatit::BeatitConfig::ComputeUnits::CPUOnly;

    std::vector<float> mono;
    double sample_rate = 0.0;
    std::string decode_error;
    if (!beatit::tests::window_alignment::decode_audio_mono(
            audio_path.string(), &mono, &sample_rate, &decode_error)) {
        std::cerr << decode_error << "\n";
        return 3;
    }

    const TimedRun first = run_once(mono, sample_rate, config);
    const TimedRun second = run_once(mono, sample_rate, config);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "model=" << model_path << "\n";
    std::cout << "file=" << audio_path.string() << "\n";
    std::cout << "first_ms=" << first.milliseconds
              << " second_ms=" << second.milliseconds
              << " speedup_x=" << (second.milliseconds > 0.0
                                        ? first.milliseconds / second.milliseconds
                                        : 0.0)
              << "\n";
    std::cout << std::setprecision(4);
    std::cout << "first_bpm=" << first.bpm
              << " second_bpm=" << second.bpm
              << " beats=" << second.beats
              << " downbeats=" << second.downbeats
              << "\n";
    return 0;
}
