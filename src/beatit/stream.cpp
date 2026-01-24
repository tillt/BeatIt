//
//  stream.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/stream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <iostream>
#include <vector>

#if defined(BEATIT_USE_TORCH)
#include <c10/core/InferenceMode.h>
#include <torch/script.h>
#include <torch/mps.h>
#include "beatit/torch_mel.h"
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

} // namespace

#if defined(BEATIT_USE_TORCH)
struct BeatitStream::TorchState {
    torch::jit::script::Module module;
    torch::Device device = torch::kCPU;
};
#endif

BeatitStream::~BeatitStream() = default;

void BeatitStream::LinearResampler::push(const float* input,
                                         std::size_t count,
                                         std::vector<float>* output) {
    if (!output || count == 0) {
        return;
    }

    buffer.insert(buffer.end(), input, input + count);
    if (ratio <= 0.0) {
        return;
    }

    const double step = 1.0 / ratio;
    while (src_index + 1.0 < static_cast<double>(buffer.size())) {
        const std::size_t index = static_cast<std::size_t>(src_index);
        const double frac = src_index - static_cast<double>(index);
        const float a = buffer[index];
        const float b = buffer[index + 1];
        output->push_back(static_cast<float>((1.0 - frac) * a + frac * b));
        src_index += step;
    }

    const std::size_t drop = static_cast<std::size_t>(src_index);
    if (drop > 0) {
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<long>(drop));
        src_index -= static_cast<double>(drop);
    }
}

#if defined(BEATIT_USE_TORCH)
bool BeatitStream::infer_torch_window(const std::vector<float>& window,
                                      std::vector<float>* beat,
                                      std::vector<float>* downbeat) {
    if (!beat || !downbeat) {
        return false;
    }
    if (coreml_config_.torch_model_path.empty()) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: missing model path.\n";
        }
        return false;
    }

    if (!torch_state_) {
        torch_state_ = std::make_unique<TorchState>();
        torch_state_->device = torch::kCPU;
        if (coreml_config_.torch_device == "mps") {
            void* metal_graph = dlopen("/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/MetalPerformanceShadersGraph",
                                       RTLD_LAZY);
            void* metal_mps = dlopen("/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders",
                                     RTLD_LAZY);
            void* metal = dlopen("/System/Library/Frameworks/Metal.framework/Metal",
                                 RTLD_LAZY);
            if (!metal_graph || !metal_mps || !metal) {
                if (coreml_config_.verbose) {
                    std::cerr << "Torch backend: Metal frameworks missing, MPS unavailable.\n";
                }
                torch_state_->device = torch::kCPU;
            } else {
                dlclose(metal_graph);
                dlclose(metal_mps);
                dlclose(metal);
                torch_state_->device = torch::kMPS;
            }
        }
        try {
            torch_state_->module = torch::jit::load(coreml_config_.torch_model_path, torch::kCPU);
            torch_state_->module.to(torch::kFloat32);
            if (torch_state_->device.type() != torch::kCPU) {
                try {
                    torch_state_->module.to(torch_state_->device);
                } catch (const c10::Error& err) {
                    if (coreml_config_.verbose) {
                        std::string message = err.what();
                        const std::size_t newline = message.find('\n');
                        if (newline != std::string::npos) {
                            message = message.substr(0, newline);
                        }
                        std::cerr << "Torch backend: device move failed, falling back to cpu: "
                                  << message << "\n";
                    }
                    torch_state_->device = torch::kCPU;
                }
            }
            if (coreml_config_.verbose) {
                std::cerr << "Torch backend: resolved device="
                          << torch_state_->device.str() << "\n";
                std::cerr << "Torch backend: mel backend="
                          << (coreml_config_.mel_backend == CoreMLConfig::MelBackend::Torch ? "torch" : "cpu")
                          << "\n";
            }
        } catch (const c10::Error& err) {
            if (coreml_config_.verbose) {
                std::cerr << "Torch backend: failed to load model: " << err.what() << "\n";
            }
            return false;
        }
    }

    std::size_t frames = 0;
    std::vector<float> features;
    if (coreml_config_.mel_backend == CoreMLConfig::MelBackend::Torch) {
        std::string mel_error;
        const auto mel_start = std::chrono::steady_clock::now();
        features = compute_mel_features_torch(window,
                                              coreml_config_.sample_rate,
                                              coreml_config_,
                                              torch_state_->device,
                                              &frames,
                                              &mel_error);
        const auto mel_end = std::chrono::steady_clock::now();
        perf_.mel_ms +=
            std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
    } else {
        const auto mel_start = std::chrono::steady_clock::now();
        features = compute_mel_features(window, coreml_config_.sample_rate, coreml_config_, &frames);
        const auto mel_end = std::chrono::steady_clock::now();
        perf_.mel_ms +=
            std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
    }
    if (features.empty() || frames == 0) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: mel feature extraction failed.\n";
        }
        return false;
    }

    const std::size_t expected_frames = coreml_config_.fixed_frames;
    if (expected_frames > 0 && frames < expected_frames) {
        features.resize(expected_frames * coreml_config_.mel_bins, 0.0f);
        frames = expected_frames;
    }

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch_state_->device);
    torch::Tensor input =
        torch::from_blob(features.data(),
                         {1, static_cast<long long>(frames),
                          static_cast<long long>(coreml_config_.mel_bins)},
                         torch::kFloat32)
            .to(options)
            .clone();

    torch::IValue output;
    std::vector<torch::IValue> inputs;
    inputs.reserve(1);
    inputs.emplace_back(input);
    try {
        c10::InferenceMode inference_guard(true);
        const auto forward_start = std::chrono::steady_clock::now();
        output = torch_state_->module.forward(inputs);
        const auto forward_end = std::chrono::steady_clock::now();
        perf_.torch_forward_ms +=
            std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
    } catch (const c10::Error& err) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward failed: " << err.what() << "\n";
        }
        return false;
    } catch (const std::exception& err) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward exception: " << err.what() << "\n";
        }
        return false;
    } catch (...) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward unknown exception\n";
        }
        return false;
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
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: unexpected output signature.\n";
        }
        return false;
    }

    beat_tensor = torch::sigmoid(beat_tensor).to(torch::kCPU).flatten();
    if (downbeat_tensor.defined()) {
        downbeat_tensor = torch::sigmoid(downbeat_tensor).to(torch::kCPU).flatten();
    }

    const std::size_t total = static_cast<std::size_t>(beat_tensor.numel());
    beat->assign(total, 0.0f);
    downbeat->assign(total, 0.0f);
    const auto beat_accessor = beat_tensor.accessor<float, 1>();
    for (std::size_t i = 0; i < total; ++i) {
        (*beat)[i] = beat_accessor[static_cast<long long>(i)];
    }
    if (downbeat_tensor.defined() && downbeat_tensor.numel() == beat_tensor.numel()) {
        const auto downbeat_accessor = downbeat_tensor.accessor<float, 1>();
        for (std::size_t i = 0; i < total; ++i) {
            (*downbeat)[i] = downbeat_accessor[static_cast<long long>(i)];
        }
    }

    return true;
}

bool BeatitStream::infer_torch_windows(const std::vector<std::vector<float>>& windows,
                                       std::vector<std::vector<float>>* beats,
                                       std::vector<std::vector<float>>* downbeats) {
    if (!beats || !downbeats) {
        return false;
    }
    if (windows.empty()) {
        return true;
    }

    if (!torch_state_) {
        std::vector<float> dummy_beat;
        std::vector<float> dummy_downbeat;
        if (!infer_torch_window(windows.front(), &dummy_beat, &dummy_downbeat)) {
            return false;
        }
    }

    if (!torch_state_) {
        return false;
    }

    const std::size_t expected_frames = coreml_config_.fixed_frames;
    const std::size_t batch = windows.size();
    const std::size_t mel_bins = coreml_config_.mel_bins;

    std::vector<float> batch_buffer(batch * expected_frames * mel_bins, 0.0f);
    for (std::size_t b = 0; b < batch; ++b) {
        std::size_t frames = 0;
        std::vector<float> features;
        if (coreml_config_.mel_backend == CoreMLConfig::MelBackend::Torch) {
            std::string mel_error;
            const auto mel_start = std::chrono::steady_clock::now();
            features = compute_mel_features_torch(windows[b],
                                                  coreml_config_.sample_rate,
                                                  coreml_config_,
                                                  torch_state_->device,
                                                  &frames,
                                                  &mel_error);
            const auto mel_end = std::chrono::steady_clock::now();
            perf_.mel_ms +=
                std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
        } else {
            const auto mel_start = std::chrono::steady_clock::now();
            features = compute_mel_features(windows[b], coreml_config_.sample_rate, coreml_config_, &frames);
            const auto mel_end = std::chrono::steady_clock::now();
            perf_.mel_ms +=
                std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
        }
        if (features.empty() || frames == 0) {
            if (coreml_config_.verbose) {
                std::cerr << "Torch backend: mel feature extraction failed.\n";
            }
            return false;
        }
        if (expected_frames > 0 && frames < expected_frames) {
            features.resize(expected_frames * mel_bins, 0.0f);
            frames = expected_frames;
        }
        for (std::size_t f = 0; f < frames; ++f) {
            const std::size_t src = f * mel_bins;
            const std::size_t dst = (b * expected_frames + f) * mel_bins;
            std::copy(features.begin() + src,
                      features.begin() + src + mel_bins,
                      batch_buffer.begin() + dst);
        }
    }

    const auto options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch_state_->device);
    torch::Tensor input =
        torch::from_blob(batch_buffer.data(),
                         {static_cast<long long>(batch),
                          static_cast<long long>(expected_frames),
                          static_cast<long long>(mel_bins)},
                         torch::kFloat32)
            .to(options)
            .clone();

    torch::IValue output;
    std::vector<torch::IValue> inputs;
    inputs.reserve(1);
    inputs.emplace_back(input);
    try {
        c10::InferenceMode inference_guard(true);
        const auto forward_start = std::chrono::steady_clock::now();
        output = torch_state_->module.forward(inputs);
        const auto forward_end = std::chrono::steady_clock::now();
        perf_.torch_forward_ms +=
            std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
    } catch (const c10::Error& err) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward failed: " << err.what() << "\n";
        }
        return false;
    } catch (const std::exception& err) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward exception: " << err.what() << "\n";
        }
        return false;
    } catch (...) {
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: forward unknown exception\n";
        }
        return false;
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
        if (coreml_config_.verbose) {
            std::cerr << "Torch backend: unexpected output signature.\n";
        }
        return false;
    }

    beat_tensor = torch::sigmoid(beat_tensor).to(torch::kCPU);
    if (downbeat_tensor.defined()) {
        downbeat_tensor = torch::sigmoid(downbeat_tensor).to(torch::kCPU);
    }

    if (beat_tensor.dim() == 3 && beat_tensor.size(1) == 1) {
        beat_tensor = beat_tensor.squeeze(1);
    }
    if (downbeat_tensor.defined() && downbeat_tensor.dim() == 3 && downbeat_tensor.size(1) == 1) {
        downbeat_tensor = downbeat_tensor.squeeze(1);
    }

    if (beat_tensor.dim() == 1) {
        beat_tensor = beat_tensor.unsqueeze(0);
    }
    if (downbeat_tensor.defined() && downbeat_tensor.dim() == 1) {
        downbeat_tensor = downbeat_tensor.unsqueeze(0);
    }

    beats->assign(batch, {});
    downbeats->assign(batch, {});

    const auto beat_cpu = beat_tensor.contiguous();
    const auto beat_accessor = beat_cpu.accessor<float, 2>();
    for (std::size_t b = 0; b < batch; ++b) {
        (*beats)[b].assign(expected_frames, 0.0f);
        for (std::size_t i = 0; i < expected_frames; ++i) {
            (*beats)[b][i] = beat_accessor[static_cast<long long>(b)][static_cast<long long>(i)];
        }
    }

    if (downbeat_tensor.defined() && downbeat_tensor.numel() > 0) {
        const auto downbeat_cpu = downbeat_tensor.contiguous();
        const auto downbeat_accessor = downbeat_cpu.accessor<float, 2>();
        for (std::size_t b = 0; b < batch; ++b) {
            (*downbeats)[b].assign(expected_frames, 0.0f);
            for (std::size_t i = 0; i < expected_frames; ++i) {
                (*downbeats)[b][i] =
                    downbeat_accessor[static_cast<long long>(b)][static_cast<long long>(i)];
            }
        }
    }

    return true;
}
#endif

BeatitStream::BeatitStream(double sample_rate,
                           const CoreMLConfig& coreml_config,
                           bool enable_coreml)
    : sample_rate_(sample_rate),
      coreml_config_(coreml_config),
      coreml_enabled_(enable_coreml) {
    resampler_.ratio = coreml_config_.sample_rate / sample_rate_;
    if (coreml_config_.sample_rate > 0) {
        const double cutoff_hz = 150.0;
        const double dt = 1.0 / static_cast<double>(coreml_config_.sample_rate);
        const double rc = 1.0 / (2.0 * 3.141592653589793 * cutoff_hz);
        phase_energy_alpha_ = dt / (rc + dt);
    }
}

void BeatitStream::process_coreml_windows() {
    if (!coreml_enabled_ || coreml_config_.fixed_frames == 0) {
        return;
    }

    const std::size_t window_samples =
        coreml_config_.frame_size + (coreml_config_.fixed_frames - 1) * coreml_config_.hop_size;
    const std::size_t hop_samples = coreml_config_.window_hop_frames * coreml_config_.hop_size;

    while (resampled_buffer_.size() - resampled_offset_ >= window_samples) {
        std::vector<float> window(window_samples, 0.0f);
        const float* start_ptr = resampled_buffer_.data() + resampled_offset_;
        std::copy(start_ptr, start_ptr + window_samples, window.begin());

        CoreMLConfig local_config = coreml_config_;
        local_config.tempo_window_percent = 0.0f;
        local_config.prefer_double_time = false;

        const auto infer_start = std::chrono::steady_clock::now();
        CoreMLResult window_result = analyze_with_coreml(window,
                                                         local_config.sample_rate,
                                                         local_config,
                                                         0.0f);
        const auto infer_end = std::chrono::steady_clock::now();
        perf_.window_infer_ms +=
            std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        perf_.window_count += 1;
        if (window_result.beat_activation.size() > local_config.fixed_frames) {
            window_result.beat_activation.resize(local_config.fixed_frames);
        }
        if (window_result.downbeat_activation.size() > local_config.fixed_frames) {
            window_result.downbeat_activation.resize(local_config.fixed_frames);
        }

        const std::size_t needed = coreml_frame_offset_ + local_config.fixed_frames;
        if (coreml_beat_activation_.size() < needed) {
            coreml_beat_activation_.resize(needed, 0.0f);
            coreml_downbeat_activation_.resize(needed, 0.0f);
        }

        for (std::size_t i = 0; i < local_config.fixed_frames; ++i) {
            const std::size_t idx = coreml_frame_offset_ + i;
            if (i < window_result.beat_activation.size()) {
                coreml_beat_activation_[idx] = std::max(coreml_beat_activation_[idx],
                                                        window_result.beat_activation[i]);
            }
            if (i < window_result.downbeat_activation.size()) {
                coreml_downbeat_activation_[idx] = std::max(coreml_downbeat_activation_[idx],
                                                            window_result.downbeat_activation[i]);
            }
        }

        resampled_offset_ += hop_samples;
        coreml_frame_offset_ += coreml_config_.window_hop_frames;

        if (resampled_offset_ > window_samples) {
            resampled_buffer_.erase(resampled_buffer_.begin(),
                                    resampled_buffer_.begin() + static_cast<long>(resampled_offset_));
            resampled_offset_ = 0;
        }
    }
}

void BeatitStream::process_torch_windows() {
    if (!coreml_enabled_ || coreml_config_.fixed_frames == 0) {
        return;
    }
#if !defined(BEATIT_USE_TORCH)
    if (coreml_config_.verbose) {
        std::cerr << "Torch backend not enabled in this build.\n";
    }
    return;
#else
    if (coreml_config_.backend != CoreMLConfig::Backend::Torch) {
        return;
    }

    const std::size_t window_samples =
        coreml_config_.frame_size + (coreml_config_.fixed_frames - 1) * coreml_config_.hop_size;
    const std::size_t hop_samples = coreml_config_.window_hop_frames * coreml_config_.hop_size;
    const std::size_t border = std::min(coreml_config_.window_border_frames,
                                        coreml_config_.fixed_frames / 2);
    const std::size_t batch_size = std::max<std::size_t>(1, coreml_config_.torch_batch_size);

    while (resampled_buffer_.size() - resampled_offset_ >= window_samples) {
        const std::size_t available =
            (resampled_buffer_.size() - resampled_offset_ - window_samples) / hop_samples + 1;
        const std::size_t batch_count = std::min(batch_size, available);

        std::vector<std::vector<float>> windows;
        windows.reserve(batch_count);
        for (std::size_t w = 0; w < batch_count; ++w) {
            std::vector<float> window(window_samples, 0.0f);
            const float* start_ptr =
                resampled_buffer_.data() + resampled_offset_ + w * hop_samples;
            std::copy(start_ptr, start_ptr + window_samples, window.begin());
            windows.push_back(std::move(window));
        }

        std::vector<std::vector<float>> beat_activations;
        std::vector<std::vector<float>> downbeat_activations;

        const auto infer_start = std::chrono::steady_clock::now();
        bool ok = infer_torch_windows(windows, &beat_activations, &downbeat_activations);
        const auto infer_end = std::chrono::steady_clock::now();
        perf_.window_infer_ms +=
            std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        perf_.window_count += batch_count;
        if (!ok) {
            return;
        }

        for (std::size_t w = 0; w < batch_count; ++w) {
            auto& beat_activation = beat_activations[w];
            auto& downbeat_activation = downbeat_activations[w];

            if (beat_activation.size() > coreml_config_.fixed_frames) {
                beat_activation.resize(coreml_config_.fixed_frames);
            }
            if (downbeat_activation.size() > coreml_config_.fixed_frames) {
                downbeat_activation.resize(coreml_config_.fixed_frames);
            }

            const std::size_t window_offset =
                coreml_frame_offset_ + w * coreml_config_.window_hop_frames;
            const std::size_t needed = window_offset + coreml_config_.fixed_frames;
            if (coreml_beat_activation_.size() < needed) {
                coreml_beat_activation_.resize(needed, 0.0f);
                coreml_downbeat_activation_.resize(needed, 0.0f);
            }

            for (std::size_t i = 0; i < coreml_config_.fixed_frames; ++i) {
                if (i < border || i >= coreml_config_.fixed_frames - border) {
                    continue;
                }
                const std::size_t idx = window_offset + i;
                if (i < beat_activation.size()) {
                    coreml_beat_activation_[idx] =
                        std::max(coreml_beat_activation_[idx], beat_activation[i]);
                }
                if (i < downbeat_activation.size()) {
                    coreml_downbeat_activation_[idx] =
                        std::max(coreml_downbeat_activation_[idx], downbeat_activation[i]);
                }
            }
        }

        resampled_offset_ += hop_samples * batch_count;
        coreml_frame_offset_ += coreml_config_.window_hop_frames * batch_count;

        if (resampled_offset_ > window_samples) {
            resampled_buffer_.erase(resampled_buffer_.begin(),
                                    resampled_buffer_.begin() + static_cast<long>(resampled_offset_));
            resampled_offset_ = 0;
        }
    }
#endif
}

void BeatitStream::push(const float* samples, std::size_t count) {
    if (!samples || count == 0 || !coreml_enabled_) {
        return;
    }

    const float rms_threshold = 0.001f;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double value = samples[i];
        sum_sq += value * value;
    }
    const double rms = count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0;
    if (rms >= rms_threshold) {
        last_active_sample_ = total_input_samples_ + count - 1;
        has_active_sample_ = true;
    }
    total_input_samples_ += count;

    const std::size_t before_size = resampled_buffer_.size();
    const auto resample_start = std::chrono::steady_clock::now();
    resampler_.push(samples, count, &resampled_buffer_);
    const auto resample_end = std::chrono::steady_clock::now();
    perf_.resample_ms +=
        std::chrono::duration<double, std::milli>(resample_end - resample_start).count();
    const std::size_t after_size = resampled_buffer_.size();
    if (after_size > before_size && coreml_config_.hop_size > 0) {
        for (std::size_t i = before_size; i < after_size; ++i) {
            const double input = resampled_buffer_[i];
            phase_energy_state_ += phase_energy_alpha_ * (input - phase_energy_state_);
            phase_energy_sum_sq_ += phase_energy_state_ * phase_energy_state_;
            phase_energy_sample_count_++;
            if (phase_energy_sample_count_ >= coreml_config_.hop_size) {
                const double rms =
                    std::sqrt(phase_energy_sum_sq_ / static_cast<double>(phase_energy_sample_count_));
                coreml_phase_energy_.push_back(static_cast<float>(rms));
                phase_energy_sum_sq_ = 0.0;
                phase_energy_sample_count_ = 0;
            }
        }
    }
    const auto process_start = std::chrono::steady_clock::now();
    if (coreml_config_.backend == CoreMLConfig::Backend::Torch) {
        process_torch_windows();
    } else {
        process_coreml_windows();
    }
    const auto process_end = std::chrono::steady_clock::now();
    perf_.process_ms +=
        std::chrono::duration<double, std::milli>(process_end - process_start).count();
}

AnalysisResult BeatitStream::finalize() {
    AnalysisResult result;

    const auto finalize_start = std::chrono::steady_clock::now();
    if (!coreml_enabled_) {
        return result;
    }

    if (coreml_config_.fixed_frames > 0 && coreml_config_.pad_final_window) {
        const std::size_t window_samples =
            coreml_config_.frame_size + (coreml_config_.fixed_frames - 1) * coreml_config_.hop_size;
        const std::size_t available =
            resampled_buffer_.size() > resampled_offset_
                ? resampled_buffer_.size() - resampled_offset_
                : 0;
        if (available > 0 && window_samples > 0) {
            std::vector<float> window(window_samples, 0.0f);
            const float* start_ptr = resampled_buffer_.data() + resampled_offset_;
            std::copy(start_ptr, start_ptr + std::min(available, window_samples), window.begin());

            CoreMLConfig local_config = coreml_config_;
            local_config.tempo_window_percent = 0.0f;
            local_config.prefer_double_time = false;

            const auto infer_start = std::chrono::steady_clock::now();
            std::vector<float> beat_activation;
            std::vector<float> downbeat_activation;
            bool ok = false;
            if (coreml_config_.backend == CoreMLConfig::Backend::Torch) {
#if defined(BEATIT_USE_TORCH)
                ok = infer_torch_window(window, &beat_activation, &downbeat_activation);
#else
                ok = false;
#endif
            } else {
                CoreMLResult window_result = analyze_with_coreml(window,
                                                                 local_config.sample_rate,
                                                                 local_config,
                                                                 0.0f);
                beat_activation = std::move(window_result.beat_activation);
                downbeat_activation = std::move(window_result.downbeat_activation);
                ok = true;
            }
            const auto infer_end = std::chrono::steady_clock::now();
            perf_.finalize_infer_ms +=
                std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
            if (!ok) {
                return result;
            }
            if (beat_activation.size() > local_config.fixed_frames) {
                beat_activation.resize(local_config.fixed_frames);
            }
            if (downbeat_activation.size() > local_config.fixed_frames) {
                downbeat_activation.resize(local_config.fixed_frames);
            }

            const std::size_t needed = coreml_frame_offset_ + local_config.fixed_frames;
            if (coreml_beat_activation_.size() < needed) {
                coreml_beat_activation_.resize(needed, 0.0f);
                coreml_downbeat_activation_.resize(needed, 0.0f);
            }

            for (std::size_t i = 0; i < local_config.fixed_frames; ++i) {
                const std::size_t idx = coreml_frame_offset_ + i;
                if (i < beat_activation.size()) {
                    coreml_beat_activation_[idx] =
                        std::max(coreml_beat_activation_[idx], beat_activation[i]);
                }
                if (i < downbeat_activation.size()) {
                    coreml_downbeat_activation_[idx] =
                        std::max(coreml_downbeat_activation_[idx], downbeat_activation[i]);
                }
            }
        }
    }

    if (coreml_beat_activation_.empty()) {
        return result;
    }

    CoreMLConfig base_config = coreml_config_;
    base_config.tempo_window_percent = 0.0f;
    base_config.prefer_double_time = false;
    base_config.synthetic_fill = false;

    std::size_t last_active_frame = 0;
    if (has_active_sample_ && coreml_config_.hop_size > 0 && sample_rate_ > 0.0) {
        const double ratio = coreml_config_.sample_rate / sample_rate_;
        const double sample_pos = static_cast<double>(last_active_sample_) * ratio;
        last_active_frame =
            static_cast<std::size_t>(std::llround(sample_pos / coreml_config_.hop_size));
    }

    const auto postprocess_start = std::chrono::steady_clock::now();
    CoreMLResult base = postprocess_coreml_activations(coreml_beat_activation_,
                                                      coreml_downbeat_activation_,
                                                      base_config,
                                                      sample_rate_,
                                                      0.0f,
                                                      last_active_frame);
    const float reference_bpm = estimate_bpm_from_beats(base.beat_sample_frames, sample_rate_);

    CoreMLResult final_result = postprocess_coreml_activations(coreml_beat_activation_,
                                                              coreml_downbeat_activation_,
                                                              coreml_config_,
                                                              sample_rate_,
                                                              reference_bpm,
                                                              last_active_frame);
    const auto postprocess_end = std::chrono::steady_clock::now();
    perf_.postprocess_ms +=
        std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();

    result.coreml_beat_activation = std::move(final_result.beat_activation);
    result.coreml_downbeat_activation = std::move(final_result.downbeat_activation);
    result.coreml_beat_feature_frames = std::move(final_result.beat_feature_frames);
    result.coreml_beat_sample_frames = std::move(final_result.beat_sample_frames);
    result.coreml_beat_strengths = std::move(final_result.beat_strengths);
    result.coreml_downbeat_feature_frames = std::move(final_result.downbeat_feature_frames);
    result.coreml_phase_energy = std::move(coreml_phase_energy_);
    result.estimated_bpm = estimate_bpm_from_beats(result.coreml_beat_sample_frames, sample_rate_);
    const auto marker_start = std::chrono::steady_clock::now();
    result.coreml_beat_events =
        build_shakespear_markers(result.coreml_beat_feature_frames,
                                 result.coreml_beat_sample_frames,
                                 result.coreml_downbeat_feature_frames,
                                 &result.coreml_beat_activation,
                                 result.estimated_bpm,
                                 sample_rate_,
                                 coreml_config_);
    const auto marker_end = std::chrono::steady_clock::now();
    perf_.marker_ms +=
        std::chrono::duration<double, std::milli>(marker_end - marker_start).count();

    const auto finalize_end = std::chrono::steady_clock::now();
    perf_.finalize_ms =
        std::chrono::duration<double, std::milli>(finalize_end - finalize_start).count();

    if (coreml_config_.profile) {
        std::cerr << "Timing(stream): resample=" << perf_.resample_ms
                  << "ms process=" << perf_.process_ms
                  << "ms mel=" << perf_.mel_ms
                  << "ms torch=" << perf_.torch_forward_ms
                  << "ms window_infer=" << perf_.window_infer_ms
                  << "ms windows=" << perf_.window_count
                  << " finalize_infer=" << perf_.finalize_infer_ms
                  << "ms postprocess=" << perf_.postprocess_ms
                  << "ms markers=" << perf_.marker_ms
                  << "ms total_finalize=" << perf_.finalize_ms
                  << "ms\n";
    }

    return result;
}

} // namespace beatit
