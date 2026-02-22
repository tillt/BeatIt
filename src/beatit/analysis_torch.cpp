//
//  analysis_torch.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#include "beatit/analysis_torch_backend.h"

#include "beatit/torch_mel.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <c10/core/InferenceMode.h>
#include <torch/script.h>

namespace beatit {

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
    std::vector<float> features;
    if (config.mel_backend == CoreMLConfig::MelBackend::Torch) {
        std::string mel_error;
        features = compute_mel_features_torch(samples,
                                              sample_rate,
                                              config,
                                              device,
                                              &frames,
                                              &mel_error);
    } else {
        features = compute_mel_features(samples, sample_rate, config, &frames);
    }
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
                  << " mel_backend="
                  << (config.mel_backend == CoreMLConfig::MelBackend::Torch ? "torch" : "cpu")
                  << "\n";
    }

    result.beat_activation.assign(frames, -1.0f);
    result.downbeat_activation.assign(frames, -1.0f);

    std::vector<char> filled(frames, 0);
    std::vector<char> downbeat_filled(frames, 0);

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const std::size_t batch_size = std::max<std::size_t>(1, config.torch_batch_size);

    struct BatchItem {
        std::size_t start = 0;
        std::size_t write_start = 0;
        std::size_t write_end = 0;
        std::size_t copy_frames = 0;
    };

    std::vector<BatchItem> batch_items;
    std::vector<float> batch_buffer;

    for (std::size_t start = 0; start < frames; ) {
        batch_items.clear();
        for (std::size_t slot = 0; slot < batch_size && start < frames; ++slot) {
            const std::size_t available = frames - start;
            if (available < window_frames && !config.pad_final_window) {
                break;
            }

            const std::size_t copy_frames = std::min(window_frames, available);
            const std::size_t write_start = start + border;
            const std::size_t write_end =
                std::min(frames, start + window_frames - border);
            if (write_end > write_start) {
                batch_items.push_back({start, write_start, write_end, copy_frames});
            }

            start += hop_frames;
        }

        if (batch_items.empty()) {
            break;
        }

        batch_buffer.assign(batch_items.size() * window_frames * config.mel_bins, 0.0f);
        for (std::size_t b = 0; b < batch_items.size(); ++b) {
            const auto& item = batch_items[b];
            for (std::size_t f = 0; f < item.copy_frames; ++f) {
                const std::size_t src = (item.start + f) * config.mel_bins;
                const std::size_t dst =
                    (b * window_frames + f) * config.mel_bins;
                std::copy(features.begin() + static_cast<long>(src),
                          features.begin() + static_cast<long>(src + config.mel_bins),
                          batch_buffer.begin() + static_cast<long>(dst));
            }
        }

        torch::Tensor input =
            torch::from_blob(batch_buffer.data(),
                             {static_cast<long long>(batch_items.size()),
                              static_cast<long long>(window_frames),
                              static_cast<long long>(config.mel_bins)},
                             torch::kFloat32)
                .to(options)
                .clone();

        torch::IValue output;
        std::vector<torch::IValue> inputs;
        inputs.reserve(1);
        inputs.emplace_back(input);
        try {
            c10::InferenceMode inference_guard(true);
            if (config.verbose) {
                std::cerr << "Torch backend: forward batch=" << batch_items.size()
                          << " start_frame=" << batch_items.front().start << "\n";
            }
            output = module.forward(inputs);
        } catch (const c10::Error& err) {
            if (config.verbose) {
                std::cerr << "Torch backend: forward failed at frame=" << batch_items.front().start
                          << " err=" << err.what() << "\n";
            }
            return CoreMLResult{};
        } catch (const std::exception& err) {
            if (config.verbose) {
                std::cerr << "Torch backend: forward exception at frame=" << batch_items.front().start
                          << " err=" << err.what() << "\n";
            }
            return CoreMLResult{};
        } catch (...) {
            if (config.verbose) {
                std::cerr << "Torch backend: forward unknown exception at frame="
                          << batch_items.front().start << "\n";
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

        const auto beat_cpu = beat_tensor.contiguous();
        const auto beat_accessor = beat_cpu.accessor<float, 2>();
        torch::Tensor downbeat_cpu;
        if (downbeat_tensor.defined()) {
            downbeat_cpu = downbeat_tensor.contiguous();
        }

        for (std::size_t b = 0; b < batch_items.size(); ++b) {
            const auto& item = batch_items[b];
            for (std::size_t i = item.write_start; i < item.write_end; ++i) {
                if (filled[i]) {
                    continue;
                }
                const std::size_t local = i - item.start;
                result.beat_activation[i] =
                    beat_accessor[static_cast<long long>(b)][static_cast<long long>(local)];
                filled[i] = 1;
            }

            if (downbeat_cpu.defined()) {
                const auto downbeat_accessor = downbeat_cpu.accessor<float, 2>();
                for (std::size_t i = item.write_start; i < item.write_end; ++i) {
                    if (downbeat_filled[i]) {
                        continue;
                    }
                    const std::size_t local = i - item.start;
                    result.downbeat_activation[i] =
                        downbeat_accessor[static_cast<long long>(b)][static_cast<long long>(local)];
                    downbeat_filled[i] = 1;
                }
            }
        }

        if (batch_items.back().start + window_frames >= frames) {
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

} // namespace beatit
