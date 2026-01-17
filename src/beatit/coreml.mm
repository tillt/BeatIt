//
//  coreml.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "beatit/coreml.h"

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// Fallback for older SDKs that do not expose metadata key constants.
#ifndef MLModelMetadataKeyAuthor
static NSString* const MLModelMetadataKeyAuthor = @"author";
#endif
#ifndef MLModelMetadataKeyShortDescription
static NSString* const MLModelMetadataKeyShortDescription = @"shortDescription";
#endif
#ifndef MLModelMetadataKeyLicense
static NSString* const MLModelMetadataKeyLicense = @"license";
#endif
#ifndef MLModelMetadataKeyVersion
static NSString* const MLModelMetadataKeyVersion = @"version";
#endif
#ifndef MLModelMetadataKeyUserDefined
static NSString* const MLModelMetadataKeyUserDefined = @"userDefined";
#endif

@interface BeatitBundleAnchor : NSObject
@end

@implementation BeatitBundleAnchor
@end

namespace beatit {
namespace {

constexpr float kPi = 3.14159265358979323846f;

NSString* resolve_model_path(const CoreMLConfig& config) {
    NSString* model_path = nil;
    if (!config.model_path.empty()) {
        NSString* candidate = [NSString stringWithUTF8String:config.model_path.c_str()];
        if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
            model_path = candidate;
        }
    }

    if (!model_path) {
        NSArray<NSString*>* brew_candidates = @[
            @"/opt/homebrew/share/beatit/beatit.mlmodelc",
            @"/usr/local/share/beatit/beatit.mlmodelc",
        ];
        for (NSString* candidate in brew_candidates) {
            if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
                model_path = candidate;
                break;
            }
        }
    }

    if (!model_path) {
        model_path = [[NSBundle mainBundle] pathForResource:@"beatit" ofType:@"mlmodelc"];
    }

    if (!model_path) {
        NSBundle* framework_bundle = [NSBundle bundleForClass:[BeatitBundleAnchor class]];
        if (framework_bundle && framework_bundle != [NSBundle mainBundle]) {
            model_path = [framework_bundle pathForResource:@"beatit" ofType:@"mlmodelc"];
        }
    }

    return model_path;
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

std::vector<float> make_hann_window(std::size_t size) {
    std::vector<float> window(size);
    if (size == 0) {
        return window;
    }

    const float denom = static_cast<float>(size - 1);
    for (std::size_t i = 0; i < size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * kPi * (static_cast<float>(i) / denom)));
    }
    return window;
}

float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> build_mel_filterbank(std::size_t mel_bins,
                                        std::size_t fft_bins,
                                        double sample_rate) {
    std::vector<float> filters(mel_bins * fft_bins, 0.0f);
    if (mel_bins == 0 || fft_bins == 0 || sample_rate <= 0.0) {
        return filters;
    }

    const float mel_min = hz_to_mel(0.0f);
    const float mel_max = hz_to_mel(static_cast<float>(sample_rate / 2.0));
    std::vector<float> mel_points(mel_bins + 2);
    for (std::size_t i = 0; i < mel_points.size(); ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(mel_bins + 1);
        mel_points[i] = mel_min + t * (mel_max - mel_min);
    }

    std::vector<float> hz_points(mel_points.size());
    for (std::size_t i = 0; i < mel_points.size(); ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    std::vector<std::size_t> bin_points(hz_points.size());
    for (std::size_t i = 0; i < hz_points.size(); ++i) {
        bin_points[i] = static_cast<std::size_t>(std::floor((fft_bins * 2) * hz_points[i] / sample_rate));
        if (bin_points[i] >= fft_bins) {
            bin_points[i] = fft_bins - 1;
        }
    }

    for (std::size_t m = 0; m < mel_bins; ++m) {
        const std::size_t left = bin_points[m];
        const std::size_t center = bin_points[m + 1];
        const std::size_t right = bin_points[m + 2];

        for (std::size_t k = left; k < center && k < fft_bins; ++k) {
            const float weight = static_cast<float>(k - left) / static_cast<float>(center - left + 1e-6f);
            filters[m * fft_bins + k] = weight;
        }
        for (std::size_t k = center; k < right && k < fft_bins; ++k) {
            const float weight = static_cast<float>(right - k) / static_cast<float>(right - center + 1e-6f);
            filters[m * fft_bins + k] = weight;
        }
    }

    return filters;
}

std::vector<float> compute_mel(const std::vector<float>& samples,
                               std::size_t sample_rate,
                               std::size_t frame_size,
                               std::size_t hop_size,
                               std::size_t mel_bins,
                               bool use_log) {
    if (samples.empty() || frame_size == 0 || hop_size == 0 || mel_bins == 0) {
        return {};
    }
    if (samples.size() < frame_size) {
        return {};
    }

    const std::size_t fft_bins = frame_size / 2;
    const std::size_t frame_count = 1 + (samples.size() - frame_size) / hop_size;

    std::vector<float> window = make_hann_window(frame_size);
    std::vector<float> buffer(frame_size, 0.0f);
    std::vector<float> windowed(frame_size, 0.0f);
    std::vector<float> spectrum(fft_bins, 0.0f);

    std::vector<float> split_real(fft_bins, 0.0f);
    std::vector<float> split_imag(fft_bins, 0.0f);
    DSPSplitComplex split{};
    split.realp = split_real.data();
    split.imagp = split_imag.data();

    const int fft_log2 = static_cast<int>(std::log2(frame_size));
    FFTSetup setup = vDSP_create_fftsetup(fft_log2, kFFTRadix2);

    std::vector<float> mel_filters = build_mel_filterbank(mel_bins, fft_bins, sample_rate);
    std::vector<float> features(frame_count * mel_bins, 0.0f);

    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const std::size_t offset = frame * hop_size;
        std::copy(samples.begin() + offset, samples.begin() + offset + frame_size, buffer.begin());
        vDSP_vmul(buffer.data(), 1, window.data(), 1, windowed.data(), 1, frame_size);
        vDSP_ctoz(reinterpret_cast<DSPComplex*>(windowed.data()), 2, &split, 1, fft_bins);
        vDSP_fft_zrip(setup, &split, 1, fft_log2, FFT_FORWARD);
        vDSP_zvmags(&split, 1, spectrum.data(), 1, fft_bins);

        for (std::size_t m = 0; m < mel_bins; ++m) {
            float sum = 0.0f;
            const float* filter = mel_filters.data() + m * fft_bins;
            for (std::size_t k = 0; k < fft_bins; ++k) {
                sum += spectrum[k] * filter[k];
            }
            const float value = use_log ? std::log1p(sum) : sum;
            features[frame * mel_bins + m] = value;
        }
    }

    vDSP_destroy_fftsetup(setup);
    return features;
}

bool load_multiarray_from_features(MLMultiArray* array,
                                   const std::vector<float>& features,
                                   std::size_t frames,
                                   std::size_t mel_bins,
                                   CoreMLConfig::InputLayout layout) {
    if (!array) {
        return false;
    }

    if (array.dataType != MLMultiArrayDataTypeFloat32) {
        return false;
    }

    if (array.shape.count < 2) {
        return false;
    }

    const auto* strides = array.strides;
    const auto* shape = array.shape;
    float* data = static_cast<float*>(array.dataPointer);
    if (layout == CoreMLConfig::InputLayout::FramesByMels) {
        if (shape.count < 3) {
            return false;
        }
        const std::size_t rows = static_cast<std::size_t>(shape[shape.count - 2].unsignedLongValue);
        const std::size_t cols = static_cast<std::size_t>(shape[shape.count - 1].unsignedLongValue);
        if (rows < frames || cols != mel_bins) {
            return false;
        }
        const std::size_t stride0 = static_cast<std::size_t>(strides[shape.count - 2].unsignedLongValue);
        const std::size_t stride1 = static_cast<std::size_t>(strides[shape.count - 1].unsignedLongValue);
        for (std::size_t t = 0; t < frames; ++t) {
            for (std::size_t m = 0; m < mel_bins; ++m) {
                const std::size_t idx = t * mel_bins + m;
                data[t * stride0 + m * stride1] = features[idx];
            }
        }
    } else {
        if (shape.count < 4) {
            return false;
        }
        const std::size_t rows = static_cast<std::size_t>(shape[shape.count - 2].unsignedLongValue);
        const std::size_t cols = static_cast<std::size_t>(shape[shape.count - 1].unsignedLongValue);
        if (rows < frames || cols != mel_bins) {
            return false;
        }
        const std::size_t stride0 = static_cast<std::size_t>(strides[shape.count - 2].unsignedLongValue);
        const std::size_t stride1 = static_cast<std::size_t>(strides[shape.count - 1].unsignedLongValue);
        for (std::size_t t = 0; t < frames; ++t) {
            for (std::size_t m = 0; m < mel_bins; ++m) {
                const std::size_t idx = t * mel_bins + m;
                data[t * stride0 + m * stride1] = features[idx];
            }
        }
    }

    return true;
}

std::vector<float> flatten_output(MLFeatureValue* value) {
    if (!value || value.type != MLFeatureTypeMultiArray) {
        return {};
    }

    MLMultiArray* array = value.multiArrayValue;
    if (!array || array.dataType != MLMultiArrayDataTypeFloat32) {
        return {};
    }

    const std::size_t count = static_cast<std::size_t>(array.count);
    std::vector<float> output(count, 0.0f);
    const float* data = static_cast<const float*>(array.dataPointer);
    std::copy(data, data + count, output.begin());
    return output;
}

std::vector<std::size_t> pick_peaks(const std::vector<float>& activation,
                                    float threshold,
                                    std::size_t min_interval,
                                    std::size_t max_interval) {
    std::vector<std::size_t> peaks;
    if (activation.size() < 3) {
        return peaks;
    }

    std::size_t last_peak = 0;
    bool has_peak = false;
    for (std::size_t i = 1; i + 1 < activation.size(); ++i) {
        const float prev = activation[i - 1];
        const float curr = activation[i];
        const float next = activation[i + 1];
        if (curr >= threshold && curr >= prev && curr >= next) {
            if (!has_peak) {
                peaks.push_back(i);
                last_peak = i;
                has_peak = true;
                continue;
            }

            const std::size_t delta = i - last_peak;
            if (delta >= min_interval && delta <= max_interval) {
                peaks.push_back(i);
                last_peak = i;
                continue;
            }

            if (delta > max_interval) {
                // Allow a restart after long gaps instead of blocking forever.
                peaks.push_back(i);
                last_peak = i;
            }
        }
    }

    return peaks;
}

float score_peaks(const std::vector<float>& activation, const std::vector<std::size_t>& peaks) {
    float sum = 0.0f;
    for (std::size_t idx : peaks) {
        if (idx < activation.size()) {
            sum += activation[idx];
        }
    }
    return sum;
}

double median_interval_frames(const std::vector<std::size_t>& peaks) {
    if (peaks.size() < 2) {
        return 0.0;
    }
    std::vector<std::size_t> intervals;
    intervals.reserve(peaks.size() - 1);
    for (std::size_t i = 1; i < peaks.size(); ++i) {
        if (peaks[i] > peaks[i - 1]) {
            intervals.push_back(peaks[i] - peaks[i - 1]);
        }
    }
    if (intervals.empty()) {
        return 0.0;
    }
    const std::size_t mid = intervals.size() / 2;
    std::nth_element(intervals.begin(), intervals.begin() + static_cast<long>(mid), intervals.end());
    return static_cast<double>(intervals[mid]);
}

std::vector<std::size_t> fill_peaks_with_grid(const std::vector<float>& activation,
                                              std::size_t start_peak,
                                              std::size_t last_active_frame,
                                              double interval,
                                              float activation_floor) {
    std::vector<std::size_t> filled;
    if (activation.empty() || interval <= 1.0 || last_active_frame <= start_peak) {
        if (start_peak < activation.size()) {
            filled.push_back(start_peak);
        }
        return filled;
    }

    const std::size_t frames = activation.size();
    const std::size_t tail_end = std::min(last_active_frame, frames - 1);
    const std::size_t window = static_cast<std::size_t>(std::max(1.0, std::round(interval * 0.25)));
    const double min_spacing = interval * 0.5;

    filled.push_back(start_peak);
    double cursor = static_cast<double>(start_peak) + interval;
    while (cursor <= static_cast<double>(tail_end)) {
        const std::size_t center = static_cast<std::size_t>(std::llround(cursor));
        const std::size_t start = center > window ? center - window : 0;
        const std::size_t end = std::min(frames - 1, center + window);
        float best_value = -1.0f;
        std::size_t best_index = center;
        for (std::size_t k = start; k <= end; ++k) {
            const float value = activation[k];
            if (value > best_value) {
                best_value = value;
                best_index = k;
            }
        }
        std::size_t chosen = best_index;
        if (best_value < activation_floor ||
            static_cast<double>(best_index - filled.back()) < min_spacing) {
            chosen = center;
        }
        if (chosen > filled.back()) {
            filled.push_back(chosen);
        }
        cursor += interval;
    }

    return filled;
}

std::vector<std::size_t> fill_peaks_with_gaps(const std::vector<float>& activation,
                                              const std::vector<std::size_t>& peaks,
                                              double fps,
                                              float activation_floor,
                                              std::size_t last_active_frame,
                                              double base_interval_frames,
                                              float gap_tolerance,
                                              float offbeat_tolerance,
                                              std::size_t window_beats) {
    if (activation.empty() || peaks.size() < 2 || fps <= 0.0) {
        return peaks;
    }
    std::vector<std::size_t> filled;
    filled.reserve(peaks.size());
    const std::size_t frames = activation.size();

    const double gap_tolerance_ratio = 1.0 + static_cast<double>(gap_tolerance);
    const double offbeat_tolerance_ratio = 1.0 - static_cast<double>(offbeat_tolerance);
    const std::size_t window_beats_clamped = std::max<std::size_t>(1, window_beats);
    const double min_spacing =
        base_interval_frames > 1.0 ? base_interval_frames * 0.5 : 1.0;

    std::vector<std::size_t> intervals;
    intervals.reserve(peaks.size() - 1);
    for (std::size_t i = 1; i < peaks.size(); ++i) {
        if (peaks[i] > peaks[i - 1]) {
            intervals.push_back(peaks[i] - peaks[i - 1]);
        } else {
            intervals.push_back(0);
        }
    }

    for (std::size_t i = 0; i + 1 < peaks.size(); ++i) {
        const std::size_t current = peaks[i];
        const std::size_t next = peaks[i + 1];
        if (filled.empty() || current > filled.back()) {
            filled.push_back(current);
        }
        if (next <= current + 1) {
            continue;
        }

        double left = 0.0;
        double right = 0.0;
        std::size_t left_count = 0;
        std::size_t right_count = 0;
        for (std::size_t w = 0; w < window_beats_clamped && i >= 1 + w; ++w) {
            const std::size_t idx = i - 1 - w;
            if (idx < intervals.size() && intervals[idx] > 0) {
                left += static_cast<double>(intervals[idx]);
                ++left_count;
            }
        }
        for (std::size_t w = 0; w < window_beats_clamped && i + w < intervals.size(); ++w) {
            const std::size_t idx = i + w;
            if (idx < intervals.size() && intervals[idx] > 0) {
                right += static_cast<double>(intervals[idx]);
                ++right_count;
            }
        }
        double nominal_interval = 0.0;
        if (left_count > 0 && right_count > 0) {
            nominal_interval = 0.5 * (left / left_count + right / right_count);
        } else if (left_count > 0) {
            nominal_interval = left / left_count;
        } else if (right_count > 0) {
            nominal_interval = right / right_count;
        }
        if (base_interval_frames > 1.0 &&
            (nominal_interval <= 1.0 || nominal_interval > base_interval_frames * 1.5)) {
            nominal_interval = base_interval_frames;
        }
        if (nominal_interval <= 1.0) {
            continue;
        }

        const double gap = static_cast<double>(next - current);
        if (gap < nominal_interval * offbeat_tolerance_ratio) {
            continue;
        }
        if (gap <= nominal_interval * gap_tolerance_ratio) {
            continue;
        }

        const double interval = nominal_interval;
        const std::size_t window = static_cast<std::size_t>(std::max(1.0, std::round(interval * 0.25)));
        double cursor = static_cast<double>(current) + interval;
        while (cursor < static_cast<double>(next)) {
            const double remaining = static_cast<double>(next) - cursor;
            if (remaining < interval * offbeat_tolerance_ratio) {
                break;
            }
            if (remaining <= interval * gap_tolerance_ratio) {
                break;
            }
            const double target = cursor;
            const std::size_t center = static_cast<std::size_t>(std::llround(target));
            const std::size_t start = center > window ? center - window : 0;
            const std::size_t end = std::min(frames - 1, center + window);
            float best_value = -1.0f;
            std::size_t best_index = center;
            for (std::size_t k = start; k <= end; ++k) {
                const float value = activation[k];
                if (value > best_value) {
                    best_value = value;
                    best_index = k;
                }
            }
            std::size_t chosen = best_index;
            if (best_value < activation_floor ||
                (filled.size() > 0 &&
                 static_cast<double>(best_index - filled.back()) < min_spacing)) {
                chosen = center;
            }
            if (chosen > current && chosen < next) {
                if (filled.empty() || chosen > filled.back()) {
                    filled.push_back(chosen);
                }
            }
            cursor += interval;
        }
    }

    const std::size_t last = peaks.back();
    if (filled.empty() || last > filled.back()) {
        filled.push_back(last);
    }

    double tail_interval = base_interval_frames;
    if (tail_interval <= 1.0 && peaks.size() >= 2 && peaks.back() > peaks[peaks.size() - 2]) {
        tail_interval = static_cast<double>(peaks.back() - peaks[peaks.size() - 2]);
    }
    if (tail_interval > 1.0 && last_active_frame > last) {
        const std::size_t tail_end = std::min(last_active_frame, frames - 1);
        if (tail_end > last + static_cast<std::size_t>(tail_interval * 0.5)) {
            const std::size_t window =
                static_cast<std::size_t>(std::max(1.0, std::round(tail_interval * 0.25)));
            double cursor = static_cast<double>(last) + tail_interval;
            while (cursor <= static_cast<double>(tail_end)) {
                const std::size_t center = static_cast<std::size_t>(std::llround(cursor));
                const std::size_t start = center > window ? center - window : 0;
                const std::size_t end = std::min(frames - 1, center + window);
                float best_value = -1.0f;
                std::size_t best_index = center;
                for (std::size_t k = start; k <= end; ++k) {
                    const float value = activation[k];
                    if (value > best_value) {
                        best_value = value;
                        best_index = k;
                    }
                }
                std::size_t chosen = best_index;
                if (best_value < activation_floor ||
                    static_cast<double>(best_index - filled.back()) < min_spacing) {
                    chosen = center;
                }
                if (chosen > filled.back()) {
                    filled.push_back(chosen);
                } else {
                    const std::size_t fallback =
                        filled.back() + static_cast<std::size_t>(std::llround(min_spacing));
                    if (fallback <= tail_end) {
                        filled.push_back(fallback);
                    } else {
                        break;
                    }
                }
                cursor += tail_interval;
            }
        }
    }
    return filled;
}

} // namespace

CoreMLResult analyze_with_coreml(const std::vector<float>& samples,
                                 double sample_rate,
                                 const CoreMLConfig& config,
                                 float reference_bpm) {
    CoreMLResult result;
    if (samples.empty() || sample_rate <= 0.0) {
        return result;
    }

    std::vector<float> resampled = resample_linear(samples, sample_rate, config.sample_rate);
    if (resampled.size() < config.frame_size) {
        return result;
    }

    const std::size_t frames = 1 + (resampled.size() - config.frame_size) / config.hop_size;
    std::vector<float> features = compute_mel(resampled,
                                              config.sample_rate,
                                              config.frame_size,
                                              config.hop_size,
                                              config.mel_bins,
                                              config.use_log_mel);
    if (features.empty()) {
        return result;
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

    NSError* error = nil;
    NSString* model_path = resolve_model_path(config);
    if (!model_path) {
        if (config.verbose) {
            std::cerr << "CoreML model not found on disk or in bundle.\n";
        }
        return result;
    }

    NSURL* model_url = [NSURL fileURLWithPath:model_path];
    MLModelConfiguration* model_config = [[MLModelConfiguration alloc] init];
    switch (config.compute_units) {
        case CoreMLConfig::ComputeUnits::CPUOnly:
            model_config.computeUnits = MLComputeUnitsCPUOnly;
            break;
        case CoreMLConfig::ComputeUnits::CPUAndGPU:
            model_config.computeUnits = MLComputeUnitsCPUAndGPU;
            break;
        case CoreMLConfig::ComputeUnits::CPUAndNeuralEngine:
            model_config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            break;
        case CoreMLConfig::ComputeUnits::All:
        default:
            model_config.computeUnits = MLComputeUnitsAll;
            break;
    }
    MLModel* model = [MLModel modelWithContentsOfURL:model_url configuration:model_config error:&error];
    if (!model) {
        if (config.verbose && error) {
            std::cerr << "CoreML load error: " << error.localizedDescription.UTF8String << "\n";
        }
        return result;
    }

    auto run_inference = [&](const std::vector<float>& window_features,
                             std::size_t window_frames,
                             std::vector<float>* beat_out,
                             std::vector<float>* downbeat_out) -> bool {
        MLMultiArray* input_array = nil;
        if (config.input_layout == CoreMLConfig::InputLayout::FramesByMels) {
            input_array = [[MLMultiArray alloc] initWithShape:@[@(1), @(window_frames), @(config.mel_bins)]
                                                   dataType:MLMultiArrayDataTypeFloat32
                                                      error:&error];
        } else {
            input_array = [[MLMultiArray alloc] initWithShape:@[@(1), @(1), @(window_frames), @(config.mel_bins)]
                                                   dataType:MLMultiArrayDataTypeFloat32
                                                      error:&error];
        }
        if (!input_array ||
            !load_multiarray_from_features(input_array,
                                           window_features,
                                           window_frames,
                                           config.mel_bins,
                                           config.input_layout)) {
            if (config.verbose) {
                std::cerr << "CoreML input shape mismatch or allocation failure.\n";
            }
            return false;
        }

        MLDictionaryFeatureProvider* input =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                [NSString stringWithUTF8String:config.input_name.c_str()] : input_array
            }
                                                          error:&error];
        if (!input) {
            if (config.verbose && error) {
                std::cerr << "CoreML input provider error: " << error.localizedDescription.UTF8String << "\n";
            }
            return false;
        }

        id<MLFeatureProvider> output = [model predictionFromFeatures:input error:&error];
        if (!output) {
            if (config.verbose && error) {
                std::cerr << "CoreML inference error: " << error.localizedDescription.UTF8String << "\n";
            }
            return false;
        }

        MLFeatureValue* beat_value =
            [output featureValueForName:[NSString stringWithUTF8String:config.beat_output_name.c_str()]];
        MLFeatureValue* downbeat_value =
            [output featureValueForName:[NSString stringWithUTF8String:config.downbeat_output_name.c_str()]];

        if (config.verbose && (!beat_value || !downbeat_value)) {
            auto* outputs = model.modelDescription.outputDescriptionsByName;
            std::cerr << "CoreML output names: ";
            bool first = true;
            for (NSString* key in outputs) {
                if (!first) {
                    std::cerr << ", ";
                }
                std::cerr << key.UTF8String;
                first = false;
            }
            std::cerr << "\n";
        }

        if (beat_out) {
            *beat_out = flatten_output(beat_value);
        }
        if (downbeat_out) {
            *downbeat_out = flatten_output(downbeat_value);
        }
        return true;
    };

    std::vector<float> beat_activation;
    std::vector<float> downbeat_activation;
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
            if (!run_inference(window_features, config.fixed_frames, &beat_activation, &downbeat_activation)) {
                return result;
            }

            if (beat_activation.size() > config.fixed_frames) {
                beat_activation.resize(config.fixed_frames);
            }
            if (downbeat_activation.size() > config.fixed_frames) {
                downbeat_activation.resize(config.fixed_frames);
            }

            for (std::size_t i = 0; i < copy_frames; ++i) {
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
        if (!run_inference(features, used_frames, &result.beat_activation, &result.downbeat_activation)) {
            return result;
        }
    }

    result = postprocess_coreml_activations(result.beat_activation,
                                            result.downbeat_activation,
                                            config,
                                            sample_rate,
                                            reference_bpm,
                                            0);

    return result;
}

CoreMLResult postprocess_coreml_activations(const std::vector<float>& beat_activation,
                                            const std::vector<float>& downbeat_activation,
                                            const CoreMLConfig& config,
                                            double sample_rate,
                                            float reference_bpm,
                                            std::size_t last_active_frame) {
    CoreMLResult result;
    result.beat_activation = beat_activation;
    result.downbeat_activation = downbeat_activation;

    const std::size_t used_frames = result.beat_activation.size();
    if (used_frames == 0) {
        return result;
    }

    float min_bpm = config.min_bpm;
    float max_bpm = config.max_bpm;
    float min_bpm_alt = min_bpm;
    float max_bpm_alt = max_bpm;
    bool has_window = false;
    if (config.tempo_window_percent > 0.0f && reference_bpm > 0.0f) {
        const float window = config.tempo_window_percent / 100.0f;
        min_bpm = reference_bpm * (1.0f - window);
        max_bpm = reference_bpm * (1.0f + window);
        if (min_bpm < 1.0f) {
            min_bpm = 1.0f;
        }
        if (config.prefer_double_time) {
            const float doubled = reference_bpm * 2.0f;
            min_bpm_alt = doubled * (1.0f - window);
            max_bpm_alt = doubled * (1.0f + window);
            if (min_bpm_alt < 1.0f) {
                min_bpm_alt = 1.0f;
            }
            has_window = true;
        }
    }

    const double fps = static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size);
    auto compute_peaks = [&](const std::vector<float>& activation,
                             float local_min_bpm,
                             float local_max_bpm,
                             float threshold) {
        const double max_bpm_local = std::max(local_min_bpm + 1.0f, local_max_bpm);
        const double min_bpm_local = std::max(1.0f, local_min_bpm);
        const std::size_t min_interval =
            static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm_local)));
        const std::size_t max_interval =
            static_cast<std::size_t>(std::ceil((60.0 * fps) / min_bpm_local));
        return pick_peaks(activation, threshold, min_interval, max_interval);
    };

    auto adjust_threshold = [&](const std::vector<float>& activation,
                                float local_min_bpm,
                                float local_max_bpm,
                                std::vector<std::size_t>* peaks) {
        if (!peaks) {
            return;
        }
        if (config.synthetic_fill) {
            return;
        }
        const double duration = static_cast<double>(used_frames) / fps;
        const double min_expected = duration * (std::max(1.0f, local_min_bpm) / 60.0);
        if (min_expected <= 0.0) {
            return;
        }
        if (peaks->size() < static_cast<std::size_t>(min_expected * 0.5)) {
            const float lowered = std::max(0.1f, config.activation_threshold * 0.5f);
            *peaks = compute_peaks(activation, local_min_bpm, local_max_bpm, lowered);
        }
    };

    std::vector<std::size_t> peaks =
        compute_peaks(result.beat_activation, min_bpm, max_bpm, config.activation_threshold);
    adjust_threshold(result.beat_activation, min_bpm, max_bpm, &peaks);
    if (config.prefer_double_time && has_window) {
        std::vector<std::size_t> peaks_alt =
            compute_peaks(result.beat_activation, min_bpm_alt, max_bpm_alt, config.activation_threshold);
        adjust_threshold(result.beat_activation, min_bpm_alt, max_bpm_alt, &peaks_alt);
        if (score_peaks(result.beat_activation, peaks_alt) >
            score_peaks(result.beat_activation, peaks)) {
            peaks.swap(peaks_alt);
        }
    }

    const float activation_floor = std::max(0.05f, config.activation_threshold * 0.2f);
    if (config.synthetic_fill) {
        double base_interval_frames = 0.0;
        if (reference_bpm > 0.0f) {
            base_interval_frames = (60.0 * fps) / reference_bpm;
        }
        if (base_interval_frames <= 1.0) {
            base_interval_frames = median_interval_frames(peaks);
        }
        std::size_t active_end = last_active_frame;
        if (active_end == 0 && used_frames > 0) {
            active_end = used_frames - 1;
        }
        std::vector<std::size_t> filled =
            fill_peaks_with_gaps(result.beat_activation,
                                 peaks,
                                 fps,
                                 activation_floor,
                                 last_active_frame,
                                 base_interval_frames,
                                 config.gap_tolerance,
                                 config.offbeat_tolerance,
                                 config.tempo_window_beats);
        if (base_interval_frames > 1.0 && !peaks.empty()) {
            std::vector<std::size_t> grid =
                fill_peaks_with_grid(result.beat_activation,
                                     peaks.front(),
                                     active_end,
                                     base_interval_frames,
                                     activation_floor);
            if (grid.size() > filled.size()) {
                filled.swap(grid);
            }
        }
        if (filled.size() > peaks.size()) {
            peaks.swap(filled);
        }
    }

    result.beat_feature_frames.clear();
    result.beat_feature_frames.reserve(peaks.size());
    result.beat_sample_frames.clear();
    result.beat_sample_frames.reserve(peaks.size());
    result.beat_strengths.clear();
    result.beat_strengths.reserve(peaks.size());
    const double hop_scale = sample_rate / static_cast<double>(config.sample_rate);
    for (std::size_t frame : peaks) {
        result.beat_feature_frames.push_back(static_cast<unsigned long long>(frame));
        const double sample_pos = static_cast<double>(frame * config.hop_size) * hop_scale;
        result.beat_sample_frames.push_back(static_cast<unsigned long long>(std::llround(sample_pos)));
        result.beat_strengths.push_back(result.beat_activation[frame]);
    }

    if (!result.downbeat_activation.empty()) {
        std::vector<std::size_t> down_peaks =
            compute_peaks(result.downbeat_activation, min_bpm, max_bpm, config.activation_threshold);
        adjust_threshold(result.downbeat_activation, min_bpm, max_bpm, &down_peaks);
        if (config.prefer_double_time && has_window) {
            std::vector<std::size_t> peaks_alt =
                compute_peaks(result.downbeat_activation, min_bpm_alt, max_bpm_alt, config.activation_threshold);
            adjust_threshold(result.downbeat_activation, min_bpm_alt, max_bpm_alt, &peaks_alt);
            if (score_peaks(result.downbeat_activation, peaks_alt) >
                score_peaks(result.downbeat_activation, down_peaks)) {
                down_peaks.swap(peaks_alt);
            }
        }
        result.downbeat_feature_frames.clear();
        result.downbeat_feature_frames.reserve(down_peaks.size());
        for (std::size_t frame : down_peaks) {
            result.downbeat_feature_frames.push_back(static_cast<unsigned long long>(frame));
        }
    } else if (!result.beat_feature_frames.empty()) {
        result.downbeat_feature_frames.push_back(result.beat_feature_frames.front());
    }

    return result;
}

CoreMLMetadata load_coreml_metadata(const CoreMLConfig& config) {
    CoreMLMetadata metadata;

    NSString* model_path = resolve_model_path(config);
    if (!model_path) {
        return metadata;
    }

    NSURL* model_url = [NSURL fileURLWithPath:model_path];
    NSError* error = nil;
    MLModelConfiguration* model_config = [[MLModelConfiguration alloc] init];
    MLModel* model = [MLModel modelWithContentsOfURL:model_url configuration:model_config error:&error];
    if (!model || error) {
        return metadata;
    }

    NSDictionary* info = model.modelDescription.metadata;
    if (!info) {
        return metadata;
    }

    auto assign_string = [&](NSString* key, std::string* target) {
        id value = [info objectForKey:key];
        if ([value isKindOfClass:[NSString class]]) {
            *target = [static_cast<NSString*>(value) UTF8String];
        }
    };

    assign_string(MLModelMetadataKeyAuthor, &metadata.author);
    assign_string(MLModelMetadataKeyShortDescription, &metadata.short_description);
    assign_string(MLModelMetadataKeyLicense, &metadata.license);
    assign_string(MLModelMetadataKeyVersion, &metadata.version);

    id user = [info objectForKey:MLModelMetadataKeyUserDefined];
    if ([user isKindOfClass:[NSDictionary class]]) {
        NSDictionary* user_dict = static_cast<NSDictionary*>(user);
        for (id key in user_dict) {
            id value = [user_dict objectForKey:key];
            if ([key isKindOfClass:[NSString class]] && [value isKindOfClass:[NSString class]]) {
                metadata.user_defined.emplace_back([static_cast<NSString*>(key) UTF8String],
                                                   [static_cast<NSString*>(value) UTF8String]);
            }
        }
    }

    return metadata;
}

} // namespace beatit
