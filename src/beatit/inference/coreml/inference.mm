//
//  inference.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "beatit/config.h"
#include "beatit/logging.hpp"

#include "beatit/audio/dsp.h"
#include "model_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <vector>

namespace beatit {
namespace {

static inline float sigmoidf_stable(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

static void apply_logits_to_probs(std::vector<float>& values, float temperature) {
    if (values.empty()) {
        return;
    }
    const float temp = (temperature > 0.0f) ? temperature : 1.0f;
    for (float& v : values) {
        v = sigmoidf_stable(v / temp);
    }
}

} // namespace

#ifdef BEATIT_COREML_PLUGIN_BUILD
CoreMLResult coreml_plugin_analyze_with_coreml_impl(const std::vector<float>& samples,
                                                    double sample_rate,
                                                    const BeatitConfig& config,
                                                    float reference_bpm) {
#else
CoreMLResult analyze_with_coreml(const std::vector<float>& samples,
                                 double sample_rate,
                                 const BeatitConfig& config,
                                 float reference_bpm) {
#endif
    CoreMLResult result;
    if (samples.empty() || sample_rate <= 0.0) {
        return result;
    }

    const auto total_start = std::chrono::steady_clock::now();

    const auto mel_start = std::chrono::steady_clock::now();
    std::size_t frames = 0;
    std::vector<float> features = compute_mel_features(samples, sample_rate, config, &frames);
    const auto mel_end = std::chrono::steady_clock::now();
    if (features.empty()) {
        return result;
    }
    const auto resample_start = mel_start;
    const auto resample_end = mel_start;

    std::vector<float> phase_energy(frames, 0.0f);
    if (config.mel_bins > 0 && config.sample_rate > 0.0) {
        const float mel_min = detail::hz_to_mel(std::max(0.0f, config.f_min), config.mel_scale);
        const float mel_max = detail::hz_to_mel(std::max(config.f_min + 1.0f, config.f_max),
                                                config.mel_scale);
        std::vector<std::size_t> low_bins;
        low_bins.reserve(config.mel_bins);
        for (std::size_t m = 0; m < config.mel_bins; ++m) {
            const float t =
                (static_cast<float>(m) + 0.5f) / static_cast<float>(config.mel_bins);
            const float mel = mel_min + t * (mel_max - mel_min);
            const float hz = detail::mel_to_hz(mel, config.mel_scale);
            if (hz <= 150.0f) {
                low_bins.push_back(m);
            }
        }
        if (low_bins.empty()) {
            low_bins.push_back(0);
        }
        for (std::size_t f = 0; f < frames; ++f) {
            float sum = 0.0f;
            const std::size_t base = f * config.mel_bins;
            for (std::size_t m : low_bins) {
                const std::size_t idx = base + m;
                if (idx < features.size()) {
                    sum += std::max(0.0f, features[idx]);
                }
            }
            phase_energy[f] = sum;
        }
    }

    std::size_t used_frames = frames;
    if (config.fixed_frames > 0) {
        used_frames = config.fixed_frames;
        if (frames < used_frames) {
            std::vector<float> padded(used_frames * config.mel_bins, 0.0f);
            std::copy(features.begin(), features.end(), padded.begin());
            features.swap(padded);
        }
    }

    const auto model_start = std::chrono::steady_clock::now();
    NSError* error = nil;
    NSString* model_path = detail::resolve_model_path(config);
    if (!model_path) {
        BEATIT_LOG_ERROR("CoreML model not found on disk or in bundle.");
        return result;
    }

    NSURL* model_url = [NSURL fileURLWithPath:model_path];
    NSURL* compiled_url = detail::compile_model_if_needed(model_url, &error);
    if (error) {
        BEATIT_LOG_WARN("CoreML compile error: " << error.localizedDescription.UTF8String);
    }
    if (compiled_url) {
        model_url = compiled_url;
    }

    MLModelConfiguration* model_config = [[MLModelConfiguration alloc] init];
    switch (config.compute_units) {
        case BeatitConfig::ComputeUnits::CPUOnly:
            model_config.computeUnits = MLComputeUnitsCPUOnly;
            break;
        case BeatitConfig::ComputeUnits::CPUAndGPU:
            model_config.computeUnits = MLComputeUnitsCPUAndGPU;
            break;
        case BeatitConfig::ComputeUnits::CPUAndNeuralEngine:
            model_config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            break;
        case BeatitConfig::ComputeUnits::All:
        default:
            model_config.computeUnits = MLComputeUnitsAll;
            break;
    }

    MLModel* model = detail::load_cached_model(model_url, model_config, &error);
    if (!model) {
        if (error) {
            BEATIT_LOG_ERROR("CoreML load error: " << error.localizedDescription.UTF8String);
        } else {
            BEATIT_LOG_ERROR("CoreML load failed without NSError details.");
        }
        return result;
    }

    const auto model_end = std::chrono::steady_clock::now();
    const BeatitConfig::InputLayout inferred_input_layout =
        detail::infer_model_input_layout(model, config);

    auto run_inference = [&](const std::vector<float>& window_features,
                             std::size_t window_frames,
                             std::vector<float>* beat_out,
                             std::vector<float>* downbeat_out) -> bool {
        auto try_layout = [&](BeatitConfig::InputLayout layout) -> bool {
            error = nil;
            MLMultiArray* input_array = nil;
            if (layout == BeatitConfig::InputLayout::FramesByMels) {
                input_array = [[MLMultiArray alloc] initWithShape:@[@(1), @(window_frames), @(config.mel_bins)]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:&error];
            } else {
                input_array = [[MLMultiArray alloc] initWithShape:@[@(1), @(1), @(window_frames), @(config.mel_bins)]
                                                       dataType:MLMultiArrayDataTypeFloat32
                                                          error:&error];
            }

            if (!input_array ||
                !detail::load_multiarray_from_features(input_array,
                                                       window_features,
                                                       window_frames,
                                                       config.mel_bins,
                                                       layout)) {
                BEATIT_LOG_DEBUG("CoreML input shape mismatch or allocation failure.");
                return false;
            }

            MLDictionaryFeatureProvider* input =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                    [NSString stringWithUTF8String:config.input_name.c_str()] : input_array
                }
                                                              error:&error];
            if (!input) {
                if (error) {
                    BEATIT_LOG_DEBUG("CoreML input provider error: "
                                     << error.localizedDescription.UTF8String);
                } else {
                    BEATIT_LOG_DEBUG("CoreML input provider creation failed without NSError details.");
                }
                return false;
            }

            id<MLFeatureProvider> output = [model predictionFromFeatures:input error:&error];
            if (!output) {
                if (error) {
                    BEATIT_LOG_DEBUG("CoreML inference error: "
                                     << error.localizedDescription.UTF8String);
                } else {
                    BEATIT_LOG_DEBUG("CoreML inference failed without NSError details.");
                }
                return false;
            }

            MLFeatureValue* beat_value =
                [output featureValueForName:[NSString stringWithUTF8String:config.beat_output_name.c_str()]];
            MLFeatureValue* downbeat_value =
                [output featureValueForName:[NSString stringWithUTF8String:config.downbeat_output_name.c_str()]];

            if (!beat_value || !downbeat_value) {
                auto* outputs = model.modelDescription.outputDescriptionsByName;
                std::ostringstream names;
                bool first = true;
                for (NSString* key in outputs) {
                    if (!first) {
                        names << ", ";
                    }
                    names << key.UTF8String;
                    first = false;
                }
                BEATIT_LOG_DEBUG("CoreML output names: " << names.str());
            }

            if (beat_out) {
                *beat_out = detail::flatten_output(beat_value);
            }
            if (downbeat_out) {
                *downbeat_out = detail::flatten_output(downbeat_value);
            }
            if (config.coreml_output_logits) {
                if (beat_out) {
                    apply_logits_to_probs(*beat_out, config.coreml_logit_temperature);
                }
                if (downbeat_out) {
                    apply_logits_to_probs(*downbeat_out, config.coreml_logit_temperature);
                }
            }
            return true;
        };

        if (try_layout(inferred_input_layout)) {
            return true;
        }

        const BeatitConfig::InputLayout fallback_layout =
            (inferred_input_layout == BeatitConfig::InputLayout::FramesByMels)
                ? BeatitConfig::InputLayout::ChannelsFramesMels
                : BeatitConfig::InputLayout::FramesByMels;
        if (fallback_layout != inferred_input_layout) {
            BEATIT_LOG_DEBUG("CoreML retrying inference with alternate input layout.");
            return try_layout(fallback_layout);
        }
        return false;
    };

    std::vector<float> beat_activation;
    std::vector<float> downbeat_activation;
    double infer_ms = 0.0;
    if (config.fixed_frames > 0 && frames > config.fixed_frames) {
        result.beat_activation.assign(frames, 0.0f);
        result.downbeat_activation.assign(frames, 0.0f);
        const std::size_t hop = std::max<std::size_t>(1, config.window_hop_frames);

        for (std::size_t start = 0; start < frames; start += hop) {
            std::vector<float> window_features(config.fixed_frames * config.mel_bins, 0.0f);
            const std::size_t remaining = frames - start;
            const std::size_t copy_frames = std::min(config.fixed_frames, remaining);
            for (std::size_t f = 0; f < copy_frames; ++f) {
                const std::size_t src = (start + f) * config.mel_bins;
                const std::size_t dst = f * config.mel_bins;
                std::copy(features.begin() + src,
                          features.begin() + src + config.mel_bins,
                          window_features.begin() + dst);
            }

            beat_activation.clear();
            downbeat_activation.clear();
            const auto infer_start = std::chrono::steady_clock::now();
            if (!run_inference(window_features,
                               config.fixed_frames,
                               &beat_activation,
                               &downbeat_activation)) {
                BEATIT_LOG_ERROR("CoreML inference failed while processing windowed activations.");
                return result;
            }
            const auto infer_end = std::chrono::steady_clock::now();
            infer_ms +=
                std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

            if (beat_activation.size() > config.fixed_frames) {
                beat_activation.resize(config.fixed_frames);
            }
            if (downbeat_activation.size() > config.fixed_frames) {
                downbeat_activation.resize(config.fixed_frames);
            }

            const std::size_t border =
                std::min(config.window_border_frames, copy_frames / 2);
            const std::size_t start_frame = border;
            const std::size_t end_frame = copy_frames > border ? copy_frames - border : 0;
            for (std::size_t i = start_frame; i < end_frame; ++i) {
                const std::size_t idx = start + i;
                if (idx < result.beat_activation.size()) {
                    result.beat_activation[idx] =
                        std::max(result.beat_activation[idx], beat_activation[i]);
                }
                if (idx < result.downbeat_activation.size() && i < downbeat_activation.size()) {
                    result.downbeat_activation[idx] =
                        std::max(result.downbeat_activation[idx], downbeat_activation[i]);
                }
            }
        }
    } else {
        const auto infer_start = std::chrono::steady_clock::now();
        if (!run_inference(features,
                           used_frames,
                           &result.beat_activation,
                           &result.downbeat_activation)) {
            BEATIT_LOG_ERROR("CoreML inference failed while processing full activations.");
            return result;
        }
        const auto infer_end = std::chrono::steady_clock::now();
        infer_ms +=
            std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    }

#ifdef BEATIT_COREML_PLUGIN_BUILD
    return result;
#else
    const auto post_start = std::chrono::steady_clock::now();
    result = postprocess_coreml_activations(result.beat_activation,
                                            result.downbeat_activation,
                                            phase_energy.empty() ? nullptr : &phase_energy,
                                            config,
                                            sample_rate,
                                            reference_bpm,
                                            0,
                                            result.beat_activation.size());
    const auto post_end = std::chrono::steady_clock::now();

    if (config.profile) {
        const auto total_end = std::chrono::steady_clock::now();
        const double resample_ms =
            std::chrono::duration<double, std::milli>(resample_end - resample_start).count();
        const double mel_ms =
            std::chrono::duration<double, std::milli>(mel_end - mel_start).count();
        const double model_ms =
            std::chrono::duration<double, std::milli>(model_end - model_start).count();
        const double post_ms =
            std::chrono::duration<double, std::milli>(post_end - post_start).count();
        const double total_ms =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();
        BEATIT_LOG_INFO("Timing(coreml): resample=" << resample_ms
                        << "ms mel=" << mel_ms
                        << "ms model=" << model_ms
                        << "ms infer=" << infer_ms
                        << "ms post=" << post_ms
                        << "ms total=" << total_ms
                        << "ms");
    }

    return result;
#endif
}

} // namespace beatit
