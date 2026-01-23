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

#if defined(BEATIT_USE_TORCH)
#include <torch/script.h>
#endif

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
    const std::uint32_t data_size = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
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
        "beatthis_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".wav";
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

    if (!parse_beatthis_output(output, beats, downbeats, error)) {
        return false;
    }

    return true;
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
        if (config.verbose) {
            std::cerr << "BeatThis failed: " << error << "\n";
        }
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

    result.coreml_beat_feature_frames.clear();
    result.coreml_beat_sample_frames.clear();
    result.coreml_beat_strengths.clear();

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

    result.coreml_downbeat_feature_frames.clear();
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

#if defined(BEATIT_USE_TORCH)
CoreMLResult analyze_with_torch_activations(const std::vector<float>& samples,
                                            double sample_rate,
                                            const CoreMLConfig& config) {
    CoreMLResult result;
    if (config.torch_model_path.empty()) {
        if (config.verbose) {
            std::cerr << "Torch backend: missing model path.\n";
        }
        return result;
    }

    const std::filesystem::path model_path(config.torch_model_path);
    if (!std::filesystem::exists(model_path)) {
        if (config.verbose) {
            std::cerr << "Torch backend: model not found: " << config.torch_model_path << "\n";
        }
        return result;
    }

    torch::Device device(torch::kCPU);
    if (config.torch_device == "mps") {
        device = torch::Device(torch::kMPS);
    }

    torch::jit::script::Module module;
    try {
        if (config.verbose) {
            std::cerr << "Torch backend: loading model=" << config.torch_model_path
                      << " device=" << config.torch_device << "\n";
        }
        module = torch::jit::load(config.torch_model_path, torch::kCPU);
        module.to(torch::kFloat32);
        if (device.type() != torch::kCPU) {
            try {
                module.to(device);
            } catch (const c10::Error& err) {
                if (config.verbose) {
                    std::string message = err.what();
                    const std::size_t newline = message.find('\n');
                    if (newline != std::string::npos) {
                        message = message.substr(0, newline);
                    }
                    std::cerr << "Torch backend: device move failed, falling back to cpu: "
                              << message << "\n";
                }
                device = torch::kCPU;
            }
        }
    } catch (const c10::Error& err) {
        if (config.verbose) {
            std::cerr << "Torch backend: failed to load model: " << err.what() << "\n";
        }
        return result;
    }

    std::size_t frames = 0;
    std::vector<float> features = compute_mel_features(samples, sample_rate, config, &frames);
    if (features.empty() || frames == 0) {
        if (config.verbose) {
            std::cerr << "Torch backend: mel feature extraction failed.\n";
        }
        return result;
    }

    const std::size_t window_frames = config.fixed_frames > 0 ? config.fixed_frames : frames;
    const std::size_t hop_frames =
        config.window_hop_frames > 0 ? config.window_hop_frames : window_frames;
    const std::size_t border =
        std::min(config.window_border_frames, window_frames / 2);

    if (config.verbose) {
        std::cerr << "Torch backend: mel_frames=" << frames
                  << " window_frames=" << window_frames
                  << " hop_frames=" << hop_frames
                  << " border_frames=" << border
                  << " mel_bins=" << config.mel_bins
                  << " sample_rate=" << config.sample_rate
                  << " hop_size=" << config.hop_size
                  << "\n";
    }

    result.beat_activation.assign(frames, -1.0f);
    result.downbeat_activation.assign(frames, -1.0f);

    std::vector<char> filled(frames, 0);
    std::vector<char> downbeat_filled(frames, 0);

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    std::vector<float> window_buffer(window_frames * config.mel_bins, 0.0f);

    for (std::size_t start = 0; start < frames; start += hop_frames) {
        const std::size_t available = frames - start;
        if (available < window_frames && !config.pad_final_window) {
            break;
        }

        std::fill(window_buffer.begin(), window_buffer.end(), 0.0f);
        const std::size_t copy_frames = std::min(window_frames, available);
        for (std::size_t f = 0; f < copy_frames; ++f) {
            const std::size_t src = (start + f) * config.mel_bins;
            const std::size_t dst = f * config.mel_bins;
            std::copy(features.begin() + src,
                      features.begin() + src + config.mel_bins,
                      window_buffer.begin() + dst);
        }

        torch::Tensor input =
            torch::from_blob(window_buffer.data(),
                             {1, static_cast<long long>(window_frames),
                              static_cast<long long>(config.mel_bins)},
                             torch::kFloat32)
                .to(options)
                .clone();

        torch::IValue output;
        std::vector<torch::IValue> inputs;
        inputs.reserve(1);
        inputs.emplace_back(input);
        try {
            if (config.verbose) {
                std::cerr << "Torch backend: forward start frame=" << start
                          << " copy_frames=" << copy_frames << "\n";
            }
            output = module.forward(inputs);
        } catch (const c10::Error& err) {
            if (config.verbose) {
                std::cerr << "Torch backend: forward failed at frame=" << start
                          << " err=" << err.what() << "\n";
            }
            return CoreMLResult{};
        } catch (const std::exception& err) {
            if (config.verbose) {
                std::cerr << "Torch backend: forward exception at frame=" << start
                          << " err=" << err.what() << "\n";
            }
            return CoreMLResult{};
        } catch (...) {
            if (config.verbose) {
                std::cerr << "Torch backend: forward unknown exception at frame=" << start << "\n";
            }
            return CoreMLResult{};
        }

        torch::Tensor beat_tensor;
        torch::Tensor downbeat_tensor;
        if (output.isTuple()) {
            const auto tuple = output.toTuple();
            const auto& elements = tuple->elements();
            if (elements.size() >= 2 && elements[0].isTensor() && elements[1].isTensor()) {
                beat_tensor = elements[0].toTensor();
                downbeat_tensor = elements[1].toTensor();
            }
        } else if (output.isGenericDict()) {
            auto dict = output.toGenericDict();
            if (dict.contains("beat")) {
                beat_tensor = dict.at("beat").toTensor();
            }
            if (dict.contains("downbeat")) {
                downbeat_tensor = dict.at("downbeat").toTensor();
            }
        }

        if (!beat_tensor.defined()) {
            if (config.verbose) {
                std::cerr << "Torch backend: unexpected output signature.\n";
            }
            return CoreMLResult{};
        }

        beat_tensor = torch::sigmoid(beat_tensor).to(torch::kCPU).flatten();
        if (downbeat_tensor.defined()) {
            downbeat_tensor = torch::sigmoid(downbeat_tensor).to(torch::kCPU).flatten();
        }

        const std::size_t window_size = static_cast<std::size_t>(beat_tensor.numel());
        const std::size_t write_start = start + border;
        const std::size_t write_end =
            std::min(frames, start + window_size - border);
        if (write_end <= write_start) {
            continue;
        }

        const auto beat_accessor = beat_tensor.accessor<float, 1>();
        for (std::size_t i = write_start; i < write_end; ++i) {
            if (filled[i]) {
                continue;
            }
            const std::size_t local = i - start;
            if (local < window_size) {
                result.beat_activation[i] = beat_accessor[static_cast<long long>(local)];
                filled[i] = 1;
            }
        }

        if (downbeat_tensor.defined() && downbeat_tensor.numel() == beat_tensor.numel()) {
            const auto downbeat_accessor = downbeat_tensor.accessor<float, 1>();
            for (std::size_t i = write_start; i < write_end; ++i) {
                if (downbeat_filled[i]) {
                    continue;
                }
                const std::size_t local = i - start;
                if (local < window_size) {
                    result.downbeat_activation[i] =
                        downbeat_accessor[static_cast<long long>(local)];
                    downbeat_filled[i] = 1;
                }
            }
        }

        if (start + window_frames >= frames) {
            break;
        }
    }

    for (std::size_t i = 0; i < frames; ++i) {
        if (result.beat_activation[i] < 0.0f) {
            result.beat_activation[i] = 0.0f;
        }
        if (result.downbeat_activation[i] < 0.0f) {
            result.downbeat_activation[i] = 0.0f;
        }
    }

    return result;
}
#else
CoreMLResult analyze_with_torch_activations(const std::vector<float>&,
                                            double,
                                            const CoreMLConfig& config) {
    if (config.verbose) {
        std::cerr << "Torch backend not enabled in this build.\n";
    }
    return {};
}
#endif

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

    if (config.backend == CoreMLConfig::Backend::BeatThisExternal) {
        return analyze_with_beatthis(samples, sample_rate, config);
    }
    if (config.backend == CoreMLConfig::Backend::Torch) {
        CoreMLConfig base_config = config;
        base_config.tempo_window_percent = 0.0f;
        base_config.prefer_double_time = false;
        base_config.synthetic_fill = false;

        const std::size_t last_active_frame =
            estimate_last_active_frame(samples, sample_rate, config);

        CoreMLResult raw = analyze_with_torch_activations(samples, sample_rate, base_config);
        CoreMLResult base = postprocess_coreml_activations(raw.beat_activation,
                                                          raw.downbeat_activation,
                                                          base_config,
                                                          sample_rate,
                                                          0.0f,
                                                          last_active_frame);
        const float reference_bpm = estimate_bpm_from_beats(base.beat_sample_frames, sample_rate);
        CoreMLResult final_result = postprocess_coreml_activations(raw.beat_activation,
                                                                  raw.downbeat_activation,
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
        result.estimated_bpm =
            estimate_bpm_from_beats(result.coreml_beat_sample_frames, sample_rate);

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
