//
//  analysis.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/analysis.h"
#include "beatit/coreml.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace beatit {
namespace {

float estimate_bpm_from_beats(const std::vector<unsigned long long>& beat_samples,
                              double sample_rate) {
    if (beat_samples.size() < 2 || sample_rate <= 0.0) {
        return 0.0f;
    }

    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 1; i < beat_samples.size(); ++i) {
        const unsigned long long prev = beat_samples[i - 1];
        const unsigned long long next = beat_samples[i];
        if (next > prev) {
            const double interval = static_cast<double>(next - prev) / sample_rate;
            if (interval > 0.0) {
                sum += 60.0 / interval;
                ++count;
            }
        }
    }

    if (count == 0) {
        return 0.0f;
    }

    return static_cast<float>(sum / static_cast<double>(count));
}

std::vector<float> resample_linear(const std::vector<float>& input,
                                   double input_rate,
                                   std::size_t target_rate) {
    if (input_rate <= 0.0 || target_rate == 0 || input.empty()) {
        return {};
    }
    if (static_cast<std::size_t>(std::lround(input_rate)) == target_rate) {
        return input;
    }

    const double ratio = static_cast<double>(target_rate) / input_rate;
    const std::size_t output_size = static_cast<std::size_t>(std::lround(input.size() * ratio));
    std::vector<float> output(output_size, 0.0f);

    for (std::size_t i = 0; i < output_size; ++i) {
        const double position = static_cast<double>(i) / ratio;
        const std::size_t index = static_cast<std::size_t>(position);
        const double frac = position - static_cast<double>(index);
        if (index + 1 < input.size()) {
            const float a = input[index];
            const float b = input[index + 1];
            output[i] = static_cast<float>((1.0 - frac) * a + frac * b);
        } else if (index < input.size()) {
            output[i] = input[index];
        }
    }

    return output;
}

std::vector<float> compute_phase_energy(const std::vector<float>& samples,
                                        double sample_rate,
                                        const CoreMLConfig& config) {
    if (samples.empty() || sample_rate <= 0.0 || config.sample_rate == 0 || config.hop_size == 0) {
        return {};
    }

    std::vector<float> resampled = resample_linear(samples, sample_rate, config.sample_rate);
    if (resampled.empty()) {
        return {};
    }

    const double cutoff_hz = 150.0;
    const double dt = 1.0 / static_cast<double>(config.sample_rate);
    const double rc = 1.0 / (2.0 * 3.141592653589793 * cutoff_hz);
    const double alpha = dt / (rc + dt);

    double state = 0.0;
    double sum_sq = 0.0;
    std::size_t count = 0;
    std::vector<float> energy;
    energy.reserve(resampled.size() / config.hop_size + 1);

    for (float sample : resampled) {
        state += alpha * (static_cast<double>(sample) - state);
        sum_sq += state * state;
        count++;
        if (count >= config.hop_size) {
            const double rms = std::sqrt(sum_sq / static_cast<double>(count));
            energy.push_back(static_cast<float>(rms));
            sum_sq = 0.0;
            count = 0;
        }
    }

    return energy;
}

std::size_t estimate_last_active_frame(const std::vector<float>& samples,
                                       double sample_rate,
                                       const CoreMLConfig& config) {
    if (samples.empty() || sample_rate <= 0.0 || config.hop_size == 0) {
        return 0;
    }

    const float rms_threshold = 0.001f;
    const std::size_t window = 1024;
    std::size_t last_active_sample = 0;
    bool found = false;
    for (std::size_t start = 0; start < samples.size(); start += window) {
        const std::size_t end = std::min(samples.size(), start + window);
        double sum_sq = 0.0;
        for (std::size_t i = start; i < end; ++i) {
            const double value = samples[i];
            sum_sq += value * value;
        }
        const double rms = sum_sq > 0.0
            ? std::sqrt(sum_sq / static_cast<double>(end - start))
            : 0.0;
        if (rms >= rms_threshold) {
            last_active_sample = end - 1;
            found = true;
        }
    }
    if (!found) {
        return 0;
    }

    const double ratio = static_cast<double>(config.sample_rate) / sample_rate;
    const double sample_pos = static_cast<double>(last_active_sample) * ratio;
    const std::size_t frame =
        static_cast<std::size_t>(std::llround(sample_pos / static_cast<double>(config.hop_size)));
    return frame;
}

} // namespace

AnalysisResult analyze(const std::vector<float>& samples,
                       double sample_rate,
                       const CoreMLConfig& config) {
    AnalysisResult result;
    if (samples.empty() || sample_rate <= 0.0) {
        return result;
    }

    CoreMLConfig base_config = config;
    base_config.tempo_window_percent = 0.0f;
    base_config.prefer_double_time = false;
    base_config.synthetic_fill = false;

    const std::size_t last_active_frame =
        estimate_last_active_frame(samples, sample_rate, config);

    CoreMLResult base = analyze_with_coreml(samples, sample_rate, base_config, 0.0f);
    const float reference_bpm = estimate_bpm_from_beats(base.beat_sample_frames, sample_rate);

    CoreMLResult final_result = postprocess_coreml_activations(base.beat_activation,
                                                              base.downbeat_activation,
                                                              config,
                                                              sample_rate,
                                                              reference_bpm,
                                                              last_active_frame);

    result.coreml_beat_activation = std::move(final_result.beat_activation);
    result.coreml_downbeat_activation = std::move(final_result.downbeat_activation);
    result.coreml_phase_energy = compute_phase_energy(samples, sample_rate, config);
    result.coreml_beat_feature_frames = std::move(final_result.beat_feature_frames);
    result.coreml_beat_sample_frames = std::move(final_result.beat_sample_frames);
    result.coreml_beat_strengths = std::move(final_result.beat_strengths);
    result.coreml_downbeat_feature_frames = std::move(final_result.downbeat_feature_frames);
    result.estimated_bpm = estimate_bpm_from_beats(result.coreml_beat_sample_frames, sample_rate);

    return result;
}

} // namespace beatit
