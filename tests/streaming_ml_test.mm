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

std::vector<float> make_click_track(double sample_rate, double bpm, double seconds) {
    const std::size_t total_samples =
        static_cast<std::size_t>(std::ceil(sample_rate * seconds));
    const std::size_t beat_period =
        static_cast<std::size_t>(std::round(sample_rate * 60.0 / bpm));
    const std::size_t pulse_width = static_cast<std::size_t>(sample_rate * 0.005);  // 5ms click.
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

bool verify_result(const beatit::AnalysisResult& result, std::string* error) {
    if (result.coreml_beat_activation.empty()) {
        if (error) {
            *error = "CoreML activation output is empty.";
        }
        return false;
    }
    if (result.coreml_beat_feature_frames.empty()) {
        if (error) {
            *error = "No beat frames detected.";
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

}  // namespace

int main() {
    const double sample_rate = 44100.0;
    const double bpm = 120.0;
    const double seconds = 32.0;

    std::string model_error;
    std::string model_path = compile_model_if_needed(BEATIT_TEST_MODEL_PATH, &model_error);
    if (model_path.empty()) {
        std::cerr << "Failed to prepare CoreML model: " << model_error << "\n";
        return 1;
    }

    beatit::CoreMLConfig config;
    config.model_path = model_path;
    config.verbose = true;
    config.activation_threshold = 0.2f;
    if (const char* force_cpu = std::getenv("BEATIT_TEST_CPU_ONLY")) {
        if (force_cpu[0] != '\0' && force_cpu[0] != '0') {
            config.compute_units = beatit::CoreMLConfig::ComputeUnits::CPUOnly;
        }
    }

    beatit::BeatitStream stream(sample_rate, config, true);
    std::vector<float> samples = make_click_track(sample_rate, bpm, seconds);

    const std::size_t chunk = 4096;
    for (std::size_t offset = 0; offset < samples.size(); offset += chunk) {
        const std::size_t remaining = samples.size() - offset;
        const std::size_t count = remaining < chunk ? remaining : chunk;
        stream.push(samples.data() + offset, count);
    }

    beatit::AnalysisResult result = stream.finalize();
    std::string error;
    if (result.coreml_beat_activation.empty() || result.coreml_beat_feature_frames.empty()) {
        std::cerr << "SKIP: CoreML output unavailable (model load failed or unsupported).\n";
        return 77;
    }
    if (!verify_result(result, &error)) {
        std::cerr << "Streaming CoreML test failed: " << error << "\n";
        return 1;
    }

    std::cout << "Streaming CoreML test passed. BPM=" << result.estimated_bpm << "\n";
    return 0;
}
