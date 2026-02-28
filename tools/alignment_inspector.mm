//
//  alignment_inspector.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-28.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "tests/coreml_test_config.h"
#include "tests/window_alignment_test_utils.h"
#include "beatit/analysis.h"
#include "beatit/stream.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int usage() {
    std::cerr << "usage: beatit_tool_alignment_inspector <audio-file> [window-beats]\n";
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    using namespace beatit::tests::window_alignment;

    if (argc < 2 || argc > 3) {
        return usage();
    }

    const std::filesystem::path audio_path = argv[1];
    const std::size_t window_beats = argc >= 3 ? static_cast<std::size_t>(std::stoul(argv[2])) : 32;

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
    config.sparse_probe_mode = true;
    config.compute_units = beatit::BeatitConfig::ComputeUnits::CPUOnly;

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
    double start_s = 0.0;
    double duration_s = 0.0;
    if (!stream.request_analysis_window(&start_s, &duration_s)) {
        std::cerr << "request_analysis_window failed\n";
        return 4;
    }
    const beatit::AnalysisResult result = stream.analyze_window(start_s, duration_s, total_duration_s, provider);

    std::vector<unsigned long long> beat_frames;
    beat_frames.reserve(result.coreml_beat_events.size());
    for (const auto& event : result.coreml_beat_events) {
        beat_frames.push_back(event.frame);
    }

    const auto offsets_ms = compute_strong_peak_offsets_ms(beat_frames, mono, sample_rate, result.estimated_bpm);
    std::cout << "bpm=" << result.estimated_bpm << " beats=" << beat_frames.size() << "\n";
    std::cout << "first16=" << format_slice(offsets_ms, 16, false) << "\n";
    std::cout << "last16=" << format_slice(offsets_ms, 16, true) << "\n";

    const std::vector<double> fractions = {0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.5, 0.75, 0.9};
    for (double fraction : fractions) {
        const std::size_t max_start = offsets_ms.size() > window_beats ? (offsets_ms.size() - window_beats) : 0;
        const std::size_t center_index = static_cast<std::size_t>(
            std::llround(fraction * static_cast<double>(offsets_ms.size() - 1)));
        const std::size_t half_window = window_beats / 2;
        const std::size_t start_index = std::min(
            max_start,
            center_index > half_window ? (center_index - half_window) : std::size_t{0});

        std::vector<double> window(offsets_ms.begin() + static_cast<long>(start_index),
                                   offsets_ms.begin() + static_cast<long>(start_index + window_beats));
        const double signed_median_ms = median(window);
        for (double& value : window) {
            value = std::fabs(value);
        }
        const double abs_median_ms = median(window);

        std::cout << "fraction=" << fraction
                  << " start_index=" << start_index
                  << " median_ms=" << signed_median_ms
                  << " median_abs_ms=" << abs_median_ms << "\n";
    }
    return 0;
}
