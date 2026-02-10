#import <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "beatit/coreml.h"

namespace {

std::vector<float> make_activation(std::size_t frames,
                                   std::size_t interval,
                                   float peak_value) {
    std::vector<float> activation(frames, 0.0f);
    for (std::size_t i = 0; i < frames; i += interval) {
        activation[i] = peak_value;
    }
    return activation;
}

bool expect_eq(std::size_t value, std::size_t expected, const char* label) {
    if (value != expected) {
        std::cerr << "DBN test failed: " << label << " expected " << expected
                  << " got " << value << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    beatit::CoreMLConfig config;
    config.sample_rate = 100;
    config.hop_size = 1;
    config.min_bpm = 118.0f;
    config.max_bpm = 122.0f;
    config.activation_threshold = 0.2f;
    config.use_dbn = true;
    config.dbn_bpm_step = 1.0f;
    config.dbn_interval_tolerance = 0.05f;
    config.dbn_activation_floor = 0.05f;
    config.dbn_beats_per_bar = 4;
    config.dbn_use_downbeat = true;
    config.prefer_double_time = false;

    const std::size_t frames = 500;
    const double bpm = 120.0;
    const std::size_t interval = static_cast<std::size_t>(std::llround((60.0 * 100.0) / bpm));

    std::vector<float> beat_activation = make_activation(frames, interval, 1.0f);
    std::vector<float> downbeat_activation(frames, 0.0f);
    for (std::size_t i = 0; i < frames; i += interval * 4) {
        downbeat_activation[i] = 1.0f;
    }

    const std::size_t expected_beats = (frames + interval - 1) / interval;
    const std::size_t expected_downbeats = (expected_beats + 3) / 4;

    beatit::CoreMLResult result =
        beatit::postprocess_coreml_activations(beat_activation,
                                               downbeat_activation,
                                               nullptr,
                                               config,
                                               config.sample_rate,
                                               0.0f,
                                               0);

    if (!expect_eq(result.beat_feature_frames.size(), expected_beats, "beat count")) {
        return 1;
    }
    if (!expect_eq(result.downbeat_feature_frames.size(), expected_downbeats, "downbeat count")) {
        return 1;
    }

    std::vector<float> sparse_activation(frames, 0.0f);
    sparse_activation[interval] = 1.0f;
    beatit::CoreMLResult sparse =
        beatit::postprocess_coreml_activations(sparse_activation,
                                               {},
                                               nullptr,
                                               config,
                                               config.sample_rate,
                                               0.0f,
                                               0);

    if (!expect_eq(sparse.beat_feature_frames.size(), 1, "sparse beat count")) {
        return 1;
    }

    std::cout << "DBN decoder test passed.\n";
    return 0;
}
