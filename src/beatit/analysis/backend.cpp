//
//  backend.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "internal.h"
#include "beatit/audio/dsp.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace beatit {
namespace {

std::string shell_escape(const std::string& value) {
    std::string escaped = "'";
    for (char ch : value) {
        if (ch == '\'') {
            escaped += "'\\''";
        } else {
            escaped += ch;
        }
    }
    escaped += "'";
    return escaped;
}

bool write_wav_mono_16(const std::string& path,
                       const std::vector<float>& samples,
                       double sample_rate,
                       std::string* error) {
    if (samples.empty() || sample_rate <= 0.0) {
        if (error) {
            *error = "Empty samples or invalid sample rate.";
        }
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        if (error) {
            *error = "Failed to open WAV output.";
        }
        return false;
    }

    const std::uint16_t channels = 1;
    const std::uint16_t bits_per_sample = 16;
    const std::uint32_t sample_rate_u = static_cast<std::uint32_t>(std::lround(sample_rate));
    const std::uint32_t byte_rate = sample_rate_u * channels * (bits_per_sample / 8);
    const std::uint16_t block_align = channels * (bits_per_sample / 8);
    const std::uint32_t data_size =
        static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    const std::uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riff_size), sizeof(riff_size));
    out.write("WAVE", 4);

    out.write("fmt ", 4);
    const std::uint32_t fmt_size = 16;
    const std::uint16_t audio_format = 1;
    out.write(reinterpret_cast<const char*>(&fmt_size), sizeof(fmt_size));
    out.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
    out.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    out.write(reinterpret_cast<const char*>(&sample_rate_u), sizeof(sample_rate_u));
    out.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    out.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
    out.write(reinterpret_cast<const char*>(&bits_per_sample), sizeof(bits_per_sample));

    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

    for (float sample : samples) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const std::int16_t value = static_cast<std::int16_t>(std::lround(clamped * 32767.0f));
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    if (!out.good()) {
        if (error) {
            *error = "Failed to write WAV data.";
        }
        return false;
    }

    return true;
}

bool parse_beatthis_output(const std::string& output,
                           std::vector<double>* beats,
                           std::vector<double>* downbeats,
                           std::string* error) {
    if (!beats || !downbeats) {
        if (error) {
            *error = "Invalid output buffers.";
        }
        return false;
    }

    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream line_stream(line);
        std::string label;
        line_stream >> label;
        if (label != "beats" && label != "downbeats") {
            continue;
        }
        std::size_t count = 0;
        line_stream >> count;
        std::vector<double>& target = (label == "beats") ? *beats : *downbeats;
        target.clear();
        target.reserve(count);
        double value = 0.0;
        while (line_stream >> value) {
            target.push_back(value);
        }
    }

    return true;
}

bool run_beatthis_external(const std::vector<float>& samples,
                           double sample_rate,
                           const CoreMLConfig& config,
                           std::vector<double>* beats,
                           std::vector<double>* downbeats,
                           std::string* error) {
    if (!beats || !downbeats) {
        if (error) {
            *error = "Invalid output buffers.";
        }
        return false;
    }
    if (config.beatthis_script.empty() || config.beatthis_checkpoint.empty()) {
        if (error) {
            *error = "BeatThis script or checkpoint path missing.";
        }
        return false;
    }

    const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
    const std::string tmp_name =
        "beatthis_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
        ".wav";
    const std::filesystem::path tmp_path = tmp_dir / tmp_name;

    std::string wav_error;
    if (!write_wav_mono_16(tmp_path.string(), samples, sample_rate, &wav_error)) {
        if (error) {
            *error = wav_error;
        }
        return false;
    }

    std::ostringstream command;
    command << shell_escape(config.beatthis_python)
            << " "
            << shell_escape(config.beatthis_script)
            << " --input "
            << shell_escape(tmp_path.string())
            << " --checkpoint "
            << shell_escape(config.beatthis_checkpoint)
            << " --device cpu";
    if (config.beatthis_use_dbn) {
        command << " --dbn";
    }

    FILE* pipe = popen(command.str().c_str(), "r");
    if (!pipe) {
        if (error) {
            *error = "Failed to launch BeatThis subprocess.";
        }
        return false;
    }

    std::string output;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }
    const int status = pclose(pipe);
    std::error_code remove_error;
    std::filesystem::remove(tmp_path, remove_error);

    if (status != 0) {
        if (error) {
            *error = "BeatThis subprocess failed.";
        }
        return false;
    }

    return parse_beatthis_output(output, beats, downbeats, error);
}

} // namespace

std::vector<float> compute_phase_energy(const std::vector<float>& samples,
                                        double sample_rate,
                                        const CoreMLConfig& config) {
    if (samples.empty() || sample_rate <= 0.0 || config.sample_rate == 0 || config.hop_size == 0) {
        return {};
    }

    std::vector<float> resampled =
        detail::resample_linear_mono(samples, sample_rate, config.sample_rate);
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

AnalysisResult analyze_with_beatthis(const std::vector<float>& samples,
                                     double sample_rate,
                                     const CoreMLConfig& config) {
    AnalysisResult result;
    std::vector<double> beat_times;
    std::vector<double> downbeat_times;
    std::string error;

    if (!run_beatthis_external(samples,
                               sample_rate,
                               config,
                               &beat_times,
                               &downbeat_times,
                               &error)) {
        BEATIT_LOG_ERROR("BeatThis failed: " << error);
        return result;
    }

    const double duration = static_cast<double>(samples.size()) / sample_rate;
    const float fallback_fps = config.hop_size > 0
        ? static_cast<float>(config.sample_rate) / static_cast<float>(config.hop_size)
        : 100.0f;
    const float fps = config.beatthis_fps > 0.0f ? config.beatthis_fps : fallback_fps;
    const std::size_t total_frames =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(duration * fps)));

    result.coreml_beat_activation.assign(total_frames, 0.0f);
    result.coreml_downbeat_activation.assign(total_frames, 0.0f);

    for (double time : beat_times) {
        const std::size_t frame = static_cast<std::size_t>(std::llround(time * fps));
        if (frame < total_frames) {
            result.coreml_beat_activation[frame] = 1.0f;
            result.coreml_beat_feature_frames.push_back(static_cast<unsigned long long>(frame));
            const auto sample_frame =
                static_cast<unsigned long long>(std::llround(time * sample_rate));
            result.coreml_beat_sample_frames.push_back(sample_frame);
            result.coreml_beat_strengths.push_back(1.0f);
        }
    }

    for (double time : downbeat_times) {
        const std::size_t frame = static_cast<std::size_t>(std::llround(time * fps));
        if (frame < total_frames) {
            result.coreml_downbeat_activation[frame] = 1.0f;
            result.coreml_downbeat_feature_frames.push_back(static_cast<unsigned long long>(frame));
        }
    }

    result.estimated_bpm =
        estimate_bpm_from_beats(result.coreml_beat_sample_frames, sample_rate);
    return result;
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

float choose_candidate_bpm(float peaks,
                           float autocorr,
                           float comb,
                           float beats) {
    const float tol = 0.02f;
    auto near = [&](float a, float b) {
        if (a <= 0.0f || b <= 0.0f) {
            return false;
        }
        return (std::abs(a - b) / std::max(a, 1e-6f)) <= tol;
    };
    if (near(peaks, comb)) {
        return 0.5f * (peaks + comb);
    }
    if (near(peaks, autocorr)) {
        return 0.5f * (peaks + autocorr);
    }
    if (near(comb, autocorr)) {
        return 0.5f * (comb + autocorr);
    }
    if (peaks > 0.0f) {
        return peaks;
    }
    if (comb > 0.0f) {
        return comb;
    }
    if (autocorr > 0.0f) {
        return autocorr;
    }
    return beats;
}

void assign_coreml_result(AnalysisResult* result,
                          CoreMLResult&& coreml_result,
                          const std::vector<float>& phase_energy,
                          double sample_rate,
                          const CoreMLConfig& config) {
    if (!result) {
        return;
    }

    result->coreml_beat_activation = std::move(coreml_result.beat_activation);
    result->coreml_downbeat_activation = std::move(coreml_result.downbeat_activation);
    result->coreml_phase_energy = phase_energy;
    result->coreml_beat_feature_frames = std::move(coreml_result.beat_feature_frames);
    result->coreml_beat_sample_frames = std::move(coreml_result.beat_sample_frames);
    result->coreml_beat_projected_feature_frames =
        std::move(coreml_result.beat_projected_feature_frames);
    result->coreml_beat_projected_sample_frames =
        std::move(coreml_result.beat_projected_sample_frames);
    result->coreml_beat_strengths = std::move(coreml_result.beat_strengths);
    result->coreml_downbeat_feature_frames = std::move(coreml_result.downbeat_feature_frames);
    result->coreml_downbeat_projected_feature_frames =
        std::move(coreml_result.downbeat_projected_feature_frames);
    result->estimated_bpm = estimate_bpm_from_beats(output_beat_sample_frames(*result), sample_rate);
    rebuild_output_beat_events(result, sample_rate, config);
}

} // namespace beatit
