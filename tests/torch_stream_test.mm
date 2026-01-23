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

std::vector<float> make_click_track(double sample_rate, double bpm, double seconds) {
    const std::size_t total_samples =
        static_cast<std::size_t>(std::ceil(sample_rate * seconds));
    const std::size_t beat_period =
        static_cast<std::size_t>(std::round(sample_rate * 60.0 / bpm));
    const std::size_t pulse_width = static_cast<std::size_t>(sample_rate * 0.005);
    std::vector<float> samples(total_samples, 0.0f);

    for (std::size_t i = 0; i < total_samples; ++i) {
        const std::size_t phase = i % beat_period;
        if (phase < pulse_width) {
            const float taper = 1.0f - static_cast<float>(phase) / static_cast<float>(pulse_width);
            samples[i] = 0.9f * taper;
        }
    }

    return samples;
}

std::string resolve_model_path() {
    if (const char* env = std::getenv("BEATIT_TORCH_MODEL_PATH")) {
        if (env[0] != '\0') {
            return env;
        }
    }

#ifdef BEATIT_TEST_TORCH_MODEL_PATH
    return BEATIT_TEST_TORCH_MODEL_PATH;
#else
    return {};
#endif
}

}  // namespace

int main() {
#if !defined(BEATIT_USE_TORCH)
    std::cout << "SKIP: Torch backend disabled at build time.\n";
    return 77;
#else
    std::string model_path = resolve_model_path();
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        std::cout << "SKIP: Torch model missing (set BEATIT_TORCH_MODEL_PATH).\n";
        return 77;
    }

    beatit::CoreMLConfig config;
    if (auto preset = beatit::make_coreml_preset("beatthis")) {
        preset->apply(config);
    }
    config.backend = beatit::CoreMLConfig::Backend::Torch;
    config.torch_model_path = model_path;
    config.torch_device = "cpu";
    config.verbose = true;

    const double sample_rate = 44100.0;
    const double bpm = 120.0;
    const double seconds = 32.0;

    beatit::BeatitStream stream(sample_rate, config, true);
    std::vector<float> samples = make_click_track(sample_rate, bpm, seconds);

    const std::size_t chunk = 4096;
    for (std::size_t offset = 0; offset < samples.size(); offset += chunk) {
        const std::size_t remaining = samples.size() - offset;
        const std::size_t count = remaining < chunk ? remaining : chunk;
        stream.push(samples.data() + offset, count);
    }

    beatit::AnalysisResult result = stream.finalize();
    if (result.coreml_beat_activation.empty()) {
        std::cerr << "Torch stream test failed: no activation output.\n";
        return 1;
    }

    std::cout << "Torch stream test passed. Beats=" << result.coreml_beat_feature_frames.size()
              << " BPM=" << result.estimated_bpm << "\n";
    return 0;
#endif
}
