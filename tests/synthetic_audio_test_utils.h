//
//  synthetic_audio_test_utils.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace beatit::tests::synthetic_audio {

inline void write_pcm16_wav(const std::filesystem::path& path,
                            const std::vector<float>& samples,
                            int sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return;
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * channels * (bits_per_sample / 8);
    const uint16_t block_align = channels * (bits_per_sample / 8);
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riff_size), sizeof(riff_size));
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    const uint32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), sizeof(fmt_size));
    const uint16_t audio_format = 1;
    out.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
    out.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    const uint32_t sr_u32 = static_cast<uint32_t>(sample_rate);
    out.write(reinterpret_cast<const char*>(&sr_u32), sizeof(sr_u32));
    out.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    out.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
    out.write(reinterpret_cast<const char*>(&bits_per_sample), sizeof(bits_per_sample));
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

    for (float sample : samples) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0f));
        out.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
    }
}

inline std::vector<float> make_click_track(double sample_rate,
                                           double bpm,
                                           double seconds,
                                           double pulse_ms,
                                           float amp,
                                           double active_start_s,
                                           double active_end_s) {
    const std::size_t total_samples =
        static_cast<std::size_t>(std::ceil(sample_rate * seconds));
    const std::size_t beat_period =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::round(sample_rate * 60.0 / bpm)));
    const std::size_t pulse_width =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::round(sample_rate * pulse_ms * 0.001)));
    std::vector<float> samples(total_samples, 0.0f);

    const std::size_t active_start =
        static_cast<std::size_t>(std::max(0.0, active_start_s) * sample_rate);
    const std::size_t active_end =
        std::min<std::size_t>(total_samples,
                              static_cast<std::size_t>(std::max(0.0, active_end_s) * sample_rate));

    for (std::size_t i = active_start; i < active_end; ++i) {
        const std::size_t phase = (i - active_start) % beat_period;
        if (phase < pulse_width) {
            const float taper = 1.0f - static_cast<float>(phase) /
                                           static_cast<float>(pulse_width);
            samples[i] = amp * taper;
        }
    }

    return samples;
}

inline std::vector<float> make_tremolo_sine(double sample_rate,
                                            double carrier_hz,
                                            double bpm_mod,
                                            double seconds,
                                            float amp) {
    const std::size_t total_samples =
        static_cast<std::size_t>(std::ceil(sample_rate * seconds));
    std::vector<float> samples(total_samples, 0.0f);
    const double w_carrier = 2.0 * M_PI * carrier_hz;
    const double w_mod = 2.0 * M_PI * (bpm_mod / 60.0);
    for (std::size_t i = 0; i < total_samples; ++i) {
        const double t = static_cast<double>(i) / sample_rate;
        const double env = 0.5 * (1.0 + std::sin(w_mod * t));
        samples[i] = amp * static_cast<float>(std::sin(w_carrier * t) * env);
    }
    return samples;
}

inline std::vector<float> make_nonrhythmic_ambient(double sample_rate,
                                                   double seconds,
                                                   uint64_t seed) {
    const std::size_t total_samples =
        static_cast<std::size_t>(std::ceil(sample_rate * seconds));
    std::vector<float> samples(total_samples, 0.0f);
    if (total_samples == 0 || sample_rate <= 0.0) {
        return samples;
    }

    struct Oscillator {
        double base_hz = 0.0;
        double drift_hz = 0.0;
        double drift_rate_hz = 0.0;
        double phase = 0.0;
        double drift_phase = 0.0;
    };

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);

    std::vector<Oscillator> oscillators;
    oscillators.push_back({87.0, 7.0, 0.009, phase_dist(rng), phase_dist(rng)});
    oscillators.push_back({131.0, 11.0, 0.013, phase_dist(rng), phase_dist(rng)});
    oscillators.push_back({197.0, 17.0, 0.017, phase_dist(rng), phase_dist(rng)});

    const double env_phase_a = phase_dist(rng);
    const double env_phase_b = phase_dist(rng);
    float brown_noise = 0.0f;
    constexpr double kTwoPi = 2.0 * M_PI;

    for (std::size_t i = 0; i < total_samples; ++i) {
        const double t = static_cast<double>(i) / sample_rate;

        const double env_a = 0.5 + 0.5 * std::sin(kTwoPi * 0.011 * t + env_phase_a);
        const double env_b = 0.5 + 0.5 * std::sin(kTwoPi * 0.017 * t + env_phase_b);
        const double envelope = 0.05 + 0.2 * env_a + 0.15 * env_b;

        double tonal = 0.0;
        for (auto& osc : oscillators) {
            const double drift =
                osc.drift_hz * std::sin(kTwoPi * osc.drift_rate_hz * t + osc.drift_phase);
            const double inst_hz = std::max(20.0, osc.base_hz + drift);
            osc.phase += kTwoPi * inst_hz / sample_rate;
            if (osc.phase > kTwoPi) {
                osc.phase -= kTwoPi;
            }
            tonal += std::sin(osc.phase);
        }
        tonal /= static_cast<double>(oscillators.size());

        const float white = noise_dist(rng) * 0.01f;
        brown_noise = 0.998f * brown_noise + 0.002f * white;
        float sample = static_cast<float>(envelope * (0.2 * tonal)) + brown_noise;

        const std::size_t fade_len = static_cast<std::size_t>(std::round(sample_rate * 2.0));
        if (i < fade_len) {
            sample *= static_cast<float>(i) / static_cast<float>(std::max<std::size_t>(1, fade_len));
        } else if (i + fade_len > total_samples) {
            const std::size_t remain = total_samples - i;
            sample *= static_cast<float>(remain) /
                      static_cast<float>(std::max<std::size_t>(1, fade_len));
        }
        samples[i] = sample;
    }

    for (float& sample : samples) {
        sample = std::max(-1.0f, std::min(1.0f, sample));
    }
    return samples;
}

inline void add_in_place(std::vector<float>* dst, const std::vector<float>& src, float gain) {
    if (!dst) {
        return;
    }
    if (dst->size() < src.size()) {
        dst->resize(src.size(), 0.0f);
    }
    for (std::size_t i = 0; i < src.size(); ++i) {
        (*dst)[i] += src[i] * gain;
    }
}

inline void clamp_audio(std::vector<float>* samples) {
    if (!samples) {
        return;
    }
    for (float& sample : *samples) {
        sample = std::max(-1.0f, std::min(1.0f, sample));
    }
}

} // namespace beatit::tests::synthetic_audio
