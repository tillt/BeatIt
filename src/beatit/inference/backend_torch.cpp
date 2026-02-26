//
//  backend_torch.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/inference/backend_torch.h"

#include "beatit/audio/mel_torch.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#include <c10/core/InferenceMode.h>
#include <dlfcn.h>
#include <torch/mps.h>
#include <torch/script.h>

namespace beatit {
namespace detail {
namespace {

bool forward_torch_model(torch::jit::script::Module* module,
                         const std::vector<torch::IValue>& inputs,
                         const CoreMLConfig& config,
                         InferenceTiming* timing,
                         torch::IValue* output) {
    if (!module || !output) {
        return false;
    }
    try {
        c10::InferenceMode inference_guard(true);
        const auto forward_start = std::chrono::steady_clock::now();
        *output = module->forward(inputs);
        const auto forward_end = std::chrono::steady_clock::now();
        if (timing) {
            timing->torch_forward_ms +=
                std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
        }
    } catch (const c10::Error& err) {
        BEATIT_LOG_ERROR("Torch backend: forward failed: " << err.what());
        return false;
    } catch (const std::exception& err) {
        BEATIT_LOG_ERROR("Torch backend: forward exception: " << err.what());
        return false;
    } catch (...) {
        BEATIT_LOG_ERROR("Torch backend: forward unknown exception");
        return false;
    }
    return true;
}

bool extract_torch_output_tensors(const torch::IValue& output,
                                  const CoreMLConfig& config,
                                  torch::Tensor* beat_tensor,
                                  torch::Tensor* downbeat_tensor) {
    if (!beat_tensor || !downbeat_tensor) {
        return false;
    }
    *beat_tensor = torch::Tensor();
    *downbeat_tensor = torch::Tensor();

    if (output.isTuple()) {
        const auto tuple = output.toTuple();
        const auto& elements = tuple->elements();
        if (elements.size() >= 2 && elements[0].isTensor() && elements[1].isTensor()) {
            *beat_tensor = elements[0].toTensor();
            *downbeat_tensor = elements[1].toTensor();
        }
    } else if (output.isGenericDict()) {
        auto dict = output.toGenericDict();
        if (dict.contains("beat")) {
            *beat_tensor = dict.at("beat").toTensor();
        }
        if (dict.contains("downbeat")) {
            *downbeat_tensor = dict.at("downbeat").toTensor();
        }
    }

    if (!beat_tensor->defined()) {
        BEATIT_LOG_ERROR("Torch backend: unexpected output signature.");
        return false;
    }
    return true;
}

class TorchInferenceBackend final : public InferenceBackend {
public:
    std::size_t max_batch_size(const CoreMLConfig& config) const override {
        return std::max<std::size_t>(1, config.torch_batch_size);
    }

    std::size_t border_frames(const CoreMLConfig& config) const override {
        return config.window_border_frames;
    }

    bool infer_window(const std::vector<float>& window,
                      const CoreMLConfig& config,
                      std::vector<float>* beat,
                      std::vector<float>* downbeat,
                      InferenceTiming* timing) override {
        if (!beat || !downbeat) {
            return false;
        }
        if (!ensure_state(config)) {
            return false;
        }

        std::size_t frames = 0;
        std::vector<float> features;
        if (config.mel_backend == CoreMLConfig::MelBackend::Torch) {
            std::string mel_error;
            const auto mel_start = std::chrono::steady_clock::now();
            features = compute_mel_features_torch(window,
                                                  config.sample_rate,
                                                  config,
                                                  torch_state_->device,
                                                  &frames,
                                                  &mel_error);
            const auto mel_end = std::chrono::steady_clock::now();
            if (timing) {
                timing->mel_ms +=
                    std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
            }
        } else {
            const auto mel_start = std::chrono::steady_clock::now();
            features = compute_mel_features(window, config.sample_rate, config, &frames);
            const auto mel_end = std::chrono::steady_clock::now();
            if (timing) {
                timing->mel_ms +=
                    std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
            }
        }
        if (features.empty() || frames == 0) {
            BEATIT_LOG_ERROR("Torch backend: mel feature extraction failed.");
            return false;
        }

        const std::size_t expected_frames = config.fixed_frames;
        if (expected_frames > 0 && frames < expected_frames) {
            features.resize(expected_frames * config.mel_bins, 0.0f);
            frames = expected_frames;
        }

        const auto options =
            torch::TensorOptions().dtype(torch::kFloat32).device(torch_state_->device);
        torch::Tensor input =
            torch::from_blob(features.data(),
                             {1, static_cast<long long>(frames),
                              static_cast<long long>(config.mel_bins)},
                             torch::kFloat32)
                .to(options)
                .clone();

        torch::IValue output;
        std::vector<torch::IValue> inputs;
        inputs.reserve(1);
        inputs.emplace_back(input);
        if (!forward_torch_model(&torch_state_->module, inputs, config, timing, &output)) {
            return false;
        }

        torch::Tensor beat_tensor;
        torch::Tensor downbeat_tensor;
        if (!extract_torch_output_tensors(output, config, &beat_tensor, &downbeat_tensor)) {
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

    bool infer_windows(const std::vector<std::vector<float>>& windows,
                       const CoreMLConfig& config,
                       std::vector<std::vector<float>>* beats,
                       std::vector<std::vector<float>>* downbeats,
                       InferenceTiming* timing) override {
        if (!beats || !downbeats) {
            return false;
        }
        if (windows.empty()) {
            return true;
        }
        if (!ensure_state(config)) {
            return false;
        }

        const std::size_t expected_frames = config.fixed_frames;
        const std::size_t batch = windows.size();
        const std::size_t mel_bins = config.mel_bins;

        std::vector<float> batch_buffer(batch * expected_frames * mel_bins, 0.0f);
        for (std::size_t b = 0; b < batch; ++b) {
            std::size_t frames = 0;
            std::vector<float> features;
            if (config.mel_backend == CoreMLConfig::MelBackend::Torch) {
                std::string mel_error;
                const auto mel_start = std::chrono::steady_clock::now();
                features = compute_mel_features_torch(windows[b],
                                                      config.sample_rate,
                                                      config,
                                                      torch_state_->device,
                                                      &frames,
                                                      &mel_error);
                const auto mel_end = std::chrono::steady_clock::now();
                if (timing) {
                    timing->mel_ms +=
                        std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
                }
            } else {
                const auto mel_start = std::chrono::steady_clock::now();
                features = compute_mel_features(windows[b], config.sample_rate, config, &frames);
                const auto mel_end = std::chrono::steady_clock::now();
                if (timing) {
                    timing->mel_ms +=
                        std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
                }
            }
            if (features.empty() || frames == 0) {
                BEATIT_LOG_ERROR("Torch backend: mel feature extraction failed.");
                return false;
            }
            if (expected_frames > 0) {
                if (frames < expected_frames) {
                    features.resize(expected_frames * mel_bins, 0.0f);
                    frames = expected_frames;
                } else if (frames > expected_frames) {
                    frames = expected_frames;
                }
            }
            for (std::size_t f = 0; f < frames; ++f) {
                const std::size_t src = f * mel_bins;
                const std::size_t dst = (b * expected_frames + f) * mel_bins;
                std::copy(features.begin() + static_cast<long>(src),
                          features.begin() + static_cast<long>(src + mel_bins),
                          batch_buffer.begin() + static_cast<long>(dst));
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
        if (!forward_torch_model(&torch_state_->module, inputs, config, timing, &output)) {
            return false;
        }

        torch::Tensor beat_tensor;
        torch::Tensor downbeat_tensor;
        if (!extract_torch_output_tensors(output, config, &beat_tensor, &downbeat_tensor)) {
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

        if (beat_tensor.dim() != 2 || static_cast<std::size_t>(beat_tensor.size(0)) < batch ||
            static_cast<std::size_t>(beat_tensor.size(1)) < expected_frames) {
            BEATIT_LOG_ERROR("Torch backend: unexpected beat tensor shape.");
            return false;
        }

        beats->assign(batch, {});
        downbeats->assign(batch, {});

        const auto beat_cpu = beat_tensor.contiguous();
        const auto beat_accessor = beat_cpu.accessor<float, 2>();
        for (std::size_t b = 0; b < batch; ++b) {
            (*beats)[b].assign(expected_frames, 0.0f);
            for (std::size_t i = 0; i < expected_frames; ++i) {
                (*beats)[b][i] =
                    beat_accessor[static_cast<long long>(b)][static_cast<long long>(i)];
            }
        }

        if (downbeat_tensor.defined() && downbeat_tensor.numel() > 0) {
            if (downbeat_tensor.dim() != 2 ||
                static_cast<std::size_t>(downbeat_tensor.size(0)) < batch ||
                static_cast<std::size_t>(downbeat_tensor.size(1)) < expected_frames) {
                BEATIT_LOG_ERROR("Torch backend: unexpected downbeat tensor shape.");
                return false;
            }
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

private:
    struct TorchState {
        torch::jit::script::Module module;
        torch::Device device = torch::kCPU;
    };

    bool ensure_state(const CoreMLConfig& config) {
        if (torch_state_) {
            return true;
        }
        if (config.torch_model_path.empty()) {
            BEATIT_LOG_ERROR("Torch backend: missing model path.");
            return false;
        }

        torch_state_ = std::make_unique<TorchState>();
        torch_state_->device = torch::kCPU;
        if (config.torch_device == "mps") {
            void* metal_graph = dlopen("/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/MetalPerformanceShadersGraph",
                                       RTLD_LAZY);
            void* metal_mps = dlopen("/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders",
                                     RTLD_LAZY);
            void* metal = dlopen("/System/Library/Frameworks/Metal.framework/Metal",
                                 RTLD_LAZY);
            if (!metal_graph || !metal_mps || !metal) {
                BEATIT_LOG_WARN("Torch backend: Metal frameworks missing, MPS unavailable.");
                torch_state_->device = torch::kCPU;
            } else {
                dlclose(metal_graph);
                dlclose(metal_mps);
                dlclose(metal);
                torch_state_->device = torch::kMPS;
            }
        }
        try {
            torch_state_->module = torch::jit::load(config.torch_model_path, torch::kCPU);
            torch_state_->module.to(torch::kFloat32);
            if (torch_state_->device.type() != torch::kCPU) {
                try {
                    torch_state_->module.to(torch_state_->device);
                } catch (const c10::Error& err) {
                    std::string message = err.what();
                    const std::size_t newline = message.find('\n');
                    if (newline != std::string::npos) {
                        message = message.substr(0, newline);
                    }
                    BEATIT_LOG_WARN(
                        "Torch backend: device move failed, falling back to cpu: " << message);
                    torch_state_->device = torch::kCPU;
                }
            }
            BEATIT_LOG_DEBUG("Torch backend: resolved device=" << torch_state_->device.str());
            BEATIT_LOG_DEBUG("Torch backend: mel backend="
                             << (config.mel_backend == CoreMLConfig::MelBackend::Torch
                                     ? "torch"
                                     : "cpu"));
        } catch (const c10::Error& err) {
            BEATIT_LOG_ERROR("Torch backend: failed to load model: " << err.what());
            torch_state_.reset();
            return false;
        }
        return true;
    }

    std::unique_ptr<TorchState> torch_state_;
};

} // namespace

std::unique_ptr<InferenceBackend> make_torch_inference_backend() {
    return std::make_unique<TorchInferenceBackend>();
}

} // namespace detail
} // namespace beatit
