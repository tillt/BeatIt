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
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
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

@interface BeatitBundleAnchor : NSObject
@end

@implementation BeatitBundleAnchor
@end

namespace beatit {
namespace {

constexpr float kPi = 3.14159265358979323846f;

MLModel* load_cached_model(NSURL* model_url,
                           MLModelConfiguration* model_config,
                           NSError** error) {
    static NSMutableDictionary<NSString*, MLModel*>* cache = nil;
    static dispatch_once_t once_token;
    dispatch_once(&once_token, ^{
        cache = [NSMutableDictionary new];
    });

    NSString* cache_key = [NSString stringWithFormat:@"%@|%ld",
                           model_url.path,
                           static_cast<long>(model_config.computeUnits)];
    @synchronized(cache) {
        MLModel* cached = [cache objectForKey:cache_key];
        if (cached) {
            return cached;
        }

        MLModel* model = [MLModel modelWithContentsOfURL:model_url
                                           configuration:model_config
                                                   error:error];
        if (model) {
            [cache setObject:model forKey:cache_key];
        }
        return model;
    }
}

NSURL* compile_model_if_needed(NSURL* model_url, NSError** error) {
    if (!model_url) {
        return nil;
    }
    NSString* extension = model_url.pathExtension.lowercaseString;
    if ([extension isEqualToString:@"mlmodelc"]) {
        return model_url;
    }
    NSURL* compiled_url = [MLModel compileModelAtURL:model_url error:error];
    return compiled_url ? compiled_url : model_url;
}

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
            @"/opt/homebrew/share/beatit/BeatThis_final0.mlmodelc",
            @"/opt/homebrew/share/beatit/BeatThis_final0.mlpackage",
            @"/usr/local/share/beatit/BeatThis_final0.mlmodelc",
            @"/usr/local/share/beatit/BeatThis_final0.mlpackage",
        ];
        for (NSString* candidate in brew_candidates) {
            if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
                model_path = candidate;
                break;
            }
        }
    }

    if (!model_path) {
        model_path = [[NSBundle mainBundle] pathForResource:@"BeatThis_final0" ofType:@"mlmodelc"];
    }
    if (!model_path) {
        model_path = [[NSBundle mainBundle] pathForResource:@"BeatThis_final0" ofType:@"mlpackage"];
    }
    if (!model_path) {
        model_path = [[NSBundle mainBundle] pathForResource:@"beatit" ofType:@"mlmodelc"];
    }

    if (!model_path) {
        NSBundle* framework_bundle = [NSBundle bundleForClass:[BeatitBundleAnchor class]];
        if (framework_bundle && framework_bundle != [NSBundle mainBundle]) {
            model_path = [framework_bundle pathForResource:@"BeatThis_final0" ofType:@"mlmodelc"];
            if (!model_path) {
                model_path = [framework_bundle pathForResource:@"BeatThis_final0" ofType:@"mlpackage"];
            }
            if (!model_path) {
                model_path = [framework_bundle pathForResource:@"beatit" ofType:@"mlmodelc"];
            }
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

float hz_to_mel(float hz, CoreMLConfig::MelScale scale) {
    if (scale == CoreMLConfig::MelScale::Slaney) {
        return 1127.01048f * std::log(1.0f + hz / 700.0f);
    }
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel, CoreMLConfig::MelScale scale) {
    if (scale == CoreMLConfig::MelScale::Slaney) {
        return 700.0f * (std::exp(mel / 1127.01048f) - 1.0f);
    }
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> build_mel_filterbank(std::size_t mel_bins,
                                        std::size_t fft_bins,
                                        double sample_rate,
                                        float f_min,
                                        float f_max,
                                        CoreMLConfig::MelScale scale) {
    std::vector<float> filters(mel_bins * fft_bins, 0.0f);
    if (mel_bins == 0 || fft_bins == 0 || sample_rate <= 0.0) {
        return filters;
    }

    const float nyquist = static_cast<float>(sample_rate / 2.0);
    const float clamped_min = std::max(0.0f, f_min);
    const float clamped_max = (f_max <= 0.0f || f_max > nyquist) ? nyquist : f_max;
    const float mel_min = hz_to_mel(clamped_min, scale);
    const float mel_max = hz_to_mel(clamped_max, scale);
    std::vector<float> mel_points(mel_bins + 2);
    for (std::size_t i = 0; i < mel_points.size(); ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(mel_bins + 1);
        mel_points[i] = mel_min + t * (mel_max - mel_min);
    }

    std::vector<float> hz_points(mel_points.size());
    for (std::size_t i = 0; i < mel_points.size(); ++i) {
        hz_points[i] = mel_to_hz(mel_points[i], scale);
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
                               bool use_log,
                               float log_multiplier,
                               float f_min,
                               float f_max,
                               CoreMLConfig::MelScale mel_scale,
                               CoreMLConfig::SpectrogramNorm spectrogram_norm,
                               float power) {
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

    std::vector<float> mel_filters =
        build_mel_filterbank(mel_bins, fft_bins, sample_rate, f_min, f_max, mel_scale);
    std::vector<float> features(frame_count * mel_bins, 0.0f);

    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const std::size_t offset = frame * hop_size;
        std::copy(samples.begin() + offset, samples.begin() + offset + frame_size, buffer.begin());
        vDSP_vmul(buffer.data(), 1, window.data(), 1, windowed.data(), 1, frame_size);
        vDSP_ctoz(reinterpret_cast<DSPComplex*>(windowed.data()), 2, &split, 1, fft_bins);
        vDSP_fft_zrip(setup, &split, 1, fft_log2, FFT_FORWARD);
        if (std::abs(power - 1.0f) < 1e-6f) {
            vDSP_zvabs(&split, 1, spectrum.data(), 1, fft_bins);
        } else {
            vDSP_zvmags(&split, 1, spectrum.data(), 1, fft_bins);
        }
        if (spectrogram_norm == CoreMLConfig::SpectrogramNorm::FrameLength) {
            const float scale = 1.0f / static_cast<float>(frame_size);
            vDSP_vsmul(spectrum.data(), 1, &scale, spectrum.data(), 1, fft_bins);
        }

        for (std::size_t m = 0; m < mel_bins; ++m) {
            float sum = 0.0f;
            const float* filter = mel_filters.data() + m * fft_bins;
            for (std::size_t k = 0; k < fft_bins; ++k) {
                sum += spectrum[k] * filter[k];
            }
            const float value =
                use_log ? std::log1p(log_multiplier * sum) : sum;
            features[frame * mel_bins + m] = value;
        }
    }

    vDSP_destroy_fftsetup(setup);
    return features;
}

} // namespace

std::vector<float> compute_mel_features(const std::vector<float>& samples,
                                        double sample_rate,
                                        const CoreMLConfig& config,
                                        std::size_t* out_frames) {
    if (out_frames) {
        *out_frames = 0;
    }
    if (samples.empty() || sample_rate <= 0.0) {
        return {};
    }
    if (config.sample_rate == 0 || config.frame_size == 0 || config.hop_size == 0 || config.mel_bins == 0) {
        return {};
    }

    std::vector<float> resampled = resample_linear(samples, sample_rate, config.sample_rate);
    if (resampled.size() < config.frame_size) {
        return {};
    }

    const std::size_t frames = 1 + (resampled.size() - config.frame_size) / config.hop_size;
    if (out_frames) {
        *out_frames = frames;
    }

    return compute_mel(resampled,
                       config.sample_rate,
                       config.frame_size,
                       config.hop_size,
                       config.mel_bins,
                       config.use_log_mel,
                       config.log_multiplier,
                       config.f_min,
                       config.f_max,
                       config.mel_scale,
                       config.spectrogram_norm,
                       config.power);
}

namespace {

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

CoreMLConfig::InputLayout infer_model_input_layout(MLModel* model,
                                                   const CoreMLConfig& config) {
    if (!model) {
        return config.input_layout;
    }

    NSString* input_name = [NSString stringWithUTF8String:config.input_name.c_str()];
    if (!input_name) {
        return config.input_layout;
    }

    MLFeatureDescription* input_desc = model.modelDescription.inputDescriptionsByName[input_name];
    if (!input_desc || input_desc.type != MLFeatureTypeMultiArray) {
        return config.input_layout;
    }

    MLMultiArrayConstraint* constraint = input_desc.multiArrayConstraint;
    if (!constraint || !constraint.shape) {
        return config.input_layout;
    }

    const NSUInteger rank = constraint.shape.count;
    if (rank >= 4) {
        return CoreMLConfig::InputLayout::ChannelsFramesMels;
    }
    if (rank == 3) {
        return CoreMLConfig::InputLayout::FramesByMels;
    }
    return config.input_layout;
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

double median_interval_frames_interpolated(const std::vector<float>& activation,
                                           const std::vector<std::size_t>& peaks) {
    if (peaks.size() < 2 || activation.empty()) {
        return 0.0;
    }
    std::vector<double> positions;
    positions.reserve(peaks.size());
    for (std::size_t frame : peaks) {
        double pos = static_cast<double>(frame);
        if (frame > 0 && frame + 1 < activation.size()) {
            const double prev = activation[frame - 1];
            const double curr = activation[frame];
            const double next = activation[frame + 1];
            const double denom = prev - 2.0 * curr + next;
            if (std::abs(denom) > 1e-9) {
                double offset = 0.5 * (prev - next) / denom;
                offset = std::max(-0.5, std::min(0.5, offset));
                pos += offset;
            }
        }
        positions.push_back(pos);
    }

    std::vector<double> intervals;
    intervals.reserve(positions.size() - 1);
    for (std::size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] > positions[i - 1]) {
            intervals.push_back(positions[i] - positions[i - 1]);
        }
    }
    if (intervals.empty()) {
        return 0.0;
    }
    const std::size_t mid = intervals.size() / 2;
    std::nth_element(intervals.begin(), intervals.begin() + static_cast<long>(mid), intervals.end());
    return intervals[mid];
}

double regression_interval_frames_interpolated(const std::vector<float>& activation,
                                               const std::vector<std::size_t>& peaks) {
    if (peaks.size() < 2 || activation.empty()) {
        return 0.0;
    }

    std::vector<double> positions;
    positions.reserve(peaks.size());
    for (std::size_t frame : peaks) {
        double pos = static_cast<double>(frame);
        if (frame > 0 && frame + 1 < activation.size()) {
            const double prev = activation[frame - 1];
            const double curr = activation[frame];
            const double next = activation[frame + 1];
            const double denom = prev - 2.0 * curr + next;
            if (std::abs(denom) > 1e-9) {
                double offset = 0.5 * (prev - next) / denom;
                offset = std::max(-0.5, std::min(0.5, offset));
                pos += offset;
            }
        }
        positions.push_back(pos);
    }

    const double n = static_cast<double>(positions.size());
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const double x = static_cast<double>(i);
        const double y = positions[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    const double denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-9) {
        return 0.0;
    }
    const double slope = (n * sum_xy - sum_x * sum_y) / denom;
    return slope > 0.0 ? slope : 0.0;
}

std::vector<std::size_t> filter_short_intervals(const std::vector<std::size_t>& frames,
                                                double min_interval_frames) {
    if (frames.size() < 2 || min_interval_frames <= 0.0) {
        return frames;
    }
    std::vector<std::size_t> filtered;
    filtered.reserve(frames.size());
    filtered.push_back(frames.front());
    for (std::size_t i = 1; i < frames.size(); ++i) {
        const std::size_t prev = filtered.back();
        const std::size_t curr = frames[i];
        if (curr > prev) {
            const double interval = static_cast<double>(curr - prev);
            if (interval >= min_interval_frames) {
                filtered.push_back(curr);
            }
        }
    }
    return filtered;
}

struct IntervalStats {
    std::size_t count = 0;
    double min_interval = 0.0;
    double max_interval = 0.0;
    double mean_interval = 0.0;
    double median_interval = 0.0;
    double stdev_interval = 0.0;
    std::vector<std::pair<double, int>> top_bpm_bins;
};

IntervalStats interval_stats_interpolated(const std::vector<float>& activation,
                                          const std::vector<std::size_t>& peaks,
                                          double fps,
                                          double bpm_bin_width) {
    IntervalStats stats;
    if (peaks.size() < 2 || activation.empty() || fps <= 0.0) {
        return stats;
    }

    std::vector<double> positions;
    positions.reserve(peaks.size());
    for (std::size_t frame : peaks) {
        double pos = static_cast<double>(frame);
        if (frame > 0 && frame + 1 < activation.size()) {
            const double prev = activation[frame - 1];
            const double curr = activation[frame];
            const double next = activation[frame + 1];
            const double denom = prev - 2.0 * curr + next;
            if (std::abs(denom) > 1e-9) {
                double offset = 0.5 * (prev - next) / denom;
                offset = std::max(-0.5, std::min(0.5, offset));
                pos += offset;
            }
        }
        positions.push_back(pos);
    }

    std::vector<double> intervals;
    intervals.reserve(positions.size() - 1);
    for (std::size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] > positions[i - 1]) {
            intervals.push_back(positions[i] - positions[i - 1]);
        }
    }
    if (intervals.empty()) {
        return stats;
    }

    stats.count = intervals.size();
    stats.min_interval = *std::min_element(intervals.begin(), intervals.end());
    stats.max_interval = *std::max_element(intervals.begin(), intervals.end());
    stats.mean_interval = std::accumulate(intervals.begin(), intervals.end(), 0.0) /
        static_cast<double>(intervals.size());
    std::vector<double> sorted = intervals;
    const std::size_t mid = sorted.size() / 2;
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(mid), sorted.end());
    stats.median_interval = sorted[mid];

    double variance = 0.0;
    for (double value : intervals) {
        const double diff = value - stats.mean_interval;
        variance += diff * diff;
    }
    stats.stdev_interval = std::sqrt(variance / static_cast<double>(intervals.size()));

    if (bpm_bin_width > 0.0) {
        std::unordered_map<int, int> bin_counts;
        for (double interval : intervals) {
            if (interval <= 0.0) {
                continue;
            }
            const double bpm = (60.0 * fps) / interval;
            const int bin = static_cast<int>(std::llround(bpm / bpm_bin_width));
            bin_counts[bin] += 1;
        }
        std::vector<std::pair<double, int>> bins;
        bins.reserve(bin_counts.size());
        for (const auto& entry : bin_counts) {
            bins.emplace_back(entry.first * bpm_bin_width, entry.second);
        }
        std::sort(bins.begin(), bins.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        const std::size_t keep = std::min<std::size_t>(5, bins.size());
        bins.resize(keep);
        stats.top_bpm_bins = std::move(bins);
    }

    return stats;
}

IntervalStats interval_stats_frames(const std::vector<std::size_t>& frames,
                                    double fps,
                                    double bpm_bin_width) {
    IntervalStats stats;
    if (frames.size() < 2 || fps <= 0.0) {
        return stats;
    }

    std::vector<double> intervals;
    intervals.reserve(frames.size() - 1);
    for (std::size_t i = 1; i < frames.size(); ++i) {
        if (frames[i] > frames[i - 1]) {
            intervals.push_back(static_cast<double>(frames[i] - frames[i - 1]));
        }
    }
    if (intervals.empty()) {
        return stats;
    }

    stats.count = intervals.size();
    stats.min_interval = *std::min_element(intervals.begin(), intervals.end());
    stats.max_interval = *std::max_element(intervals.begin(), intervals.end());
    stats.mean_interval = std::accumulate(intervals.begin(), intervals.end(), 0.0) /
        static_cast<double>(intervals.size());
    std::vector<double> sorted = intervals;
    const std::size_t mid = sorted.size() / 2;
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(mid), sorted.end());
    stats.median_interval = sorted[mid];

    double variance = 0.0;
    for (double value : intervals) {
        const double diff = value - stats.mean_interval;
        variance += diff * diff;
    }
    stats.stdev_interval = std::sqrt(variance / static_cast<double>(intervals.size()));

    if (bpm_bin_width > 0.0) {
        std::unordered_map<int, int> bin_counts;
        for (double interval : intervals) {
            if (interval <= 0.0) {
                continue;
            }
            const double bpm = (60.0 * fps) / interval;
            const int bin = static_cast<int>(std::llround(bpm / bpm_bin_width));
            bin_counts[bin] += 1;
        }
        std::vector<std::pair<double, int>> bins;
        bins.reserve(bin_counts.size());
        for (const auto& entry : bin_counts) {
            bins.emplace_back(entry.first * bpm_bin_width, entry.second);
        }
        std::sort(bins.begin(), bins.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        const std::size_t keep = std::min<std::size_t>(5, bins.size());
        bins.resize(keep);
        stats.top_bpm_bins = std::move(bins);
    }

    return stats;
}

struct DBNDecodeResult {
    std::vector<std::size_t> beat_frames;
    std::vector<std::size_t> downbeat_frames;
    double bpm = 0.0;
};

std::vector<std::size_t> viterbi_beats(const std::vector<float>& activation,
                                       double fps,
                                       double bpm,
                                       double interval_tolerance,
                                       float activation_floor) {
    std::vector<std::size_t> beats;
    if (activation.empty() || fps <= 0.0 || bpm <= 0.0) {
        return beats;
    }

    const double interval = (60.0 * fps) / bpm;
    const double tolerance = std::max(0.0, interval_tolerance);
    const std::size_t min_interval =
        static_cast<std::size_t>(std::max(1.0, std::floor(interval * (1.0 - tolerance))));
    const std::size_t max_interval =
        static_cast<std::size_t>(std::max<double>(min_interval,
                                                  std::ceil(interval * (1.0 + tolerance))));

    const double floor_value = std::max(1e-6f, activation_floor);
    const std::size_t frames = activation.size();

    std::vector<double> score(frames, std::numeric_limits<double>::lowest());
    std::vector<int> prev(frames, -1);

    for (std::size_t i = 0; i < frames; ++i) {
        const double obs = std::log(std::max<double>(activation[i], floor_value));
        score[i] = obs;
        prev[i] = -1;

        const std::size_t start = (i > max_interval) ? i - max_interval : 0;
        const std::size_t end = (i > min_interval) ? i - min_interval : 0;
        for (std::size_t j = start; j <= end; ++j) {
            if (score[j] == std::numeric_limits<double>::lowest()) {
                continue;
            }
            const double candidate = score[j] + obs;
            if (candidate > score[i]) {
                score[i] = candidate;
                prev[i] = static_cast<int>(j);
            }
        }
    }

    std::size_t best_idx = 0;
    double best_score = std::numeric_limits<double>::lowest();
    for (std::size_t i = 0; i < frames; ++i) {
        if (score[i] > best_score) {
            best_score = score[i];
            best_idx = i;
        }
    }

    int cursor = static_cast<int>(best_idx);
    while (cursor >= 0) {
        beats.push_back(static_cast<std::size_t>(cursor));
        cursor = prev[static_cast<std::size_t>(cursor)];
    }

    std::reverse(beats.begin(), beats.end());
    return beats;
}

struct DBNPathResult {
    DBNDecodeResult decoded;
    double best_score = std::numeric_limits<double>::lowest();
};

DBNPathResult decode_dbn_beats_candidate(const std::vector<double>& beat_log,
                                         const std::vector<double>& downbeat_log,
                                         double fps,
                                         float min_bpm,
                                         float max_bpm,
                                         float bpm_step,
                                         std::size_t beats_per_bar,
                                         double tolerance,
                                         bool use_downbeat,
                                         double transition_reward,
                                         double tempo_change_penalty,
                                         const CoreMLConfig& config) {
    DBNPathResult result;
    if (beat_log.empty() || fps <= 0.0) {
        return result;
    }

    const std::size_t candidate_count = beat_log.size();
    if (candidate_count < 2) {
        result.decoded.beat_frames = viterbi_beats(config.dbn_use_all_candidates
                                                       ? std::vector<float>(beat_log.begin(), beat_log.end())
                                                       : std::vector<float>(),
                                                   fps,
                                                   std::max(1.0f, min_bpm),
                                                   static_cast<float>(tolerance),
                                                   0.01f);
        if (!result.decoded.beat_frames.empty()) {
            result.decoded.downbeat_frames.push_back(result.decoded.beat_frames.front());
            result.best_score = 0.0;
        }
        return result;
    }

    const std::size_t tempo_count =
        static_cast<std::size_t>(std::floor((max_bpm - min_bpm) / bpm_step)) + 1;
    const std::size_t phase_count = beats_per_bar;

    struct TempoParams {
        float bpm = 0.0f;
        std::size_t min_interval = 0;
        std::size_t max_interval = 0;
    };

    std::vector<TempoParams> tempos;
    tempos.reserve(tempo_count);
    for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
        const float bpm = min_bpm + static_cast<float>(tempo_idx) * bpm_step;
        const double interval = (60.0 * fps) / bpm;
        const double min_interval = interval * (1.0 - tolerance);
        const double max_interval = interval * (1.0 + tolerance);
        tempos.push_back({
            bpm,
            static_cast<std::size_t>(std::max(1.0, std::floor(min_interval))),
            static_cast<std::size_t>(std::max(1.0, std::ceil(max_interval))),
        });
    }

    struct Backref {
        int prev_idx = -1;
        int prev_tempo = -1;
    };

    const std::size_t state_count = tempo_count * phase_count;
    const std::size_t total_states = candidate_count * state_count;
    std::vector<double> scores(total_states, std::numeric_limits<double>::lowest());
    std::vector<Backref> backrefs(total_states);

    auto state_index = [&](std::size_t cand_idx, std::size_t tempo_idx, std::size_t phase_idx) {
        return (cand_idx * tempo_count + tempo_idx) * phase_count + phase_idx;
    };

    for (std::size_t ci = 0; ci < candidate_count; ++ci) {
        const std::size_t frame = ci;
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            const auto& tempo = tempos[tempo_idx];
            const std::size_t min_prev_frame =
                (frame > tempo.max_interval) ? frame - tempo.max_interval : 0;
            const std::size_t max_prev_frame =
                (frame > tempo.min_interval) ? frame - tempo.min_interval : 0;
            const std::size_t start_idx = min_prev_frame;
            const std::size_t end_idx = std::min(max_prev_frame + 1, ci);

            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
                const bool is_downbeat = (phase_idx == 0);
                double obs = beat_log[frame];
                if (use_downbeat && is_downbeat && frame < downbeat_log.size()) {
                    obs = downbeat_log[frame];
                }

                double best_score = obs;
                Backref best_backref;

                const std::size_t prev_phase =
                    (phase_idx + phase_count - 1) % phase_count;
                for (std::size_t cj = start_idx; cj < end_idx; ++cj) {
                    for (int tempo_delta = -1; tempo_delta <= 1; ++tempo_delta) {
                        const int prev_tempo = static_cast<int>(tempo_idx) + tempo_delta;
                        if (prev_tempo < 0 || prev_tempo >= static_cast<int>(tempo_count)) {
                            continue;
                        }
                        const std::size_t idx =
                            state_index(cj, static_cast<std::size_t>(prev_tempo), prev_phase);
                        const double prev_score = scores[idx];
                        if (prev_score == std::numeric_limits<double>::lowest()) {
                            continue;
                        }
                        const double tempo_penalty =
                            tempo_change_penalty * std::abs(tempo.bpm - tempos[prev_tempo].bpm);
                        const double candidate = prev_score + obs - tempo_penalty + transition_reward;
                        if (candidate > best_score) {
                            best_score = candidate;
                            best_backref.prev_idx = static_cast<int>(cj);
                            best_backref.prev_tempo = prev_tempo;
                        }
                    }
                }

                const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
                scores[idx] = best_score;
                backrefs[idx] = best_backref;
            }
        }
    }

    double best_score = std::numeric_limits<double>::lowest();
    std::size_t best_ci = 0;
    std::size_t best_tempo = 0;
    std::size_t best_phase = 0;
    for (std::size_t ci = 0; ci < candidate_count; ++ci) {
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
                const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
                const double score = scores[idx];
                if (score > best_score) {
                    best_score = score;
                    best_ci = ci;
                    best_tempo = tempo_idx;
                    best_phase = phase_idx;
                }
            }
        }
    }

    std::vector<std::size_t> beat_frames;
    std::vector<std::size_t> downbeat_frames;
    std::size_t ci = best_ci;
    std::size_t tempo_idx = best_tempo;
    std::size_t phase_idx = best_phase;

    while (true) {
        beat_frames.push_back(ci);
        if (phase_idx == 0) {
            downbeat_frames.push_back(ci);
        }
        const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
        const Backref ref = backrefs[idx];
        if (ref.prev_idx < 0) {
            break;
        }
        ci = static_cast<std::size_t>(ref.prev_idx);
        tempo_idx = static_cast<std::size_t>(ref.prev_tempo);
        phase_idx = (phase_idx + phase_count - 1) % phase_count;
    }

    std::reverse(beat_frames.begin(), beat_frames.end());
    std::reverse(downbeat_frames.begin(), downbeat_frames.end());

    result.decoded.beat_frames = std::move(beat_frames);
    result.decoded.downbeat_frames = std::move(downbeat_frames);
    if (result.decoded.downbeat_frames.empty() && !result.decoded.beat_frames.empty()) {
        result.decoded.downbeat_frames.push_back(result.decoded.beat_frames.front());
    }
    if (best_tempo < tempos.size()) {
        result.decoded.bpm = tempos[best_tempo].bpm;
    }
    result.best_score = best_score;
    return result;
}

DBNPathResult decode_dbn_beats_candidate_sparse(const std::vector<std::size_t>& candidate_frames,
                                                const std::vector<double>& beat_log,
                                                const std::vector<double>& downbeat_log,
                                                double fps,
                                                float min_bpm,
                                                float max_bpm,
                                                float bpm_step,
                                                std::size_t beats_per_bar,
                                                double tolerance,
                                                bool use_downbeat,
                                                double transition_reward,
                                                double tempo_change_penalty) {
    DBNPathResult result;
    if (candidate_frames.size() < 2 || beat_log.size() != candidate_frames.size() || fps <= 0.0) {
        return result;
    }

    const std::size_t candidate_count = candidate_frames.size();
    const std::size_t tempo_count =
        static_cast<std::size_t>(std::floor((max_bpm - min_bpm) / bpm_step)) + 1;
    const std::size_t phase_count = beats_per_bar;

    struct TempoParams {
        float bpm = 0.0f;
        std::size_t min_interval = 0;
        std::size_t max_interval = 0;
    };

    std::vector<TempoParams> tempos;
    tempos.reserve(tempo_count);
    for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
        const float bpm = min_bpm + static_cast<float>(tempo_idx) * bpm_step;
        const double interval = (60.0 * fps) / bpm;
        const double min_interval = interval * (1.0 - tolerance);
        const double max_interval = interval * (1.0 + tolerance);
        tempos.push_back({
            bpm,
            static_cast<std::size_t>(std::max(1.0, std::floor(min_interval))),
            static_cast<std::size_t>(std::max(1.0, std::ceil(max_interval))),
        });
    }

    struct Backref {
        int prev_idx = -1;
        int prev_tempo = -1;
    };

    const std::size_t state_count = tempo_count * phase_count;
    const std::size_t total_states = candidate_count * state_count;
    std::vector<double> scores(total_states, std::numeric_limits<double>::lowest());
    std::vector<Backref> backrefs(total_states);

    auto state_index = [&](std::size_t cand_idx, std::size_t tempo_idx, std::size_t phase_idx) {
        return (cand_idx * tempo_count + tempo_idx) * phase_count + phase_idx;
    };

    for (std::size_t ci = 0; ci < candidate_count; ++ci) {
        const std::size_t frame = candidate_frames[ci];
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            const auto& tempo = tempos[tempo_idx];
            const std::size_t min_prev_frame =
                (frame > tempo.max_interval) ? frame - tempo.max_interval : 0;
            const std::size_t max_prev_frame =
                (frame > tempo.min_interval) ? frame - tempo.min_interval : 0;

            const auto start_it = std::lower_bound(candidate_frames.begin(),
                                                   candidate_frames.end(),
                                                   min_prev_frame);
            const auto end_it = std::upper_bound(candidate_frames.begin(),
                                                 candidate_frames.end(),
                                                 max_prev_frame);
            const std::size_t start_idx =
                static_cast<std::size_t>(std::distance(candidate_frames.begin(), start_it));
            const std::size_t end_idx =
                static_cast<std::size_t>(std::distance(candidate_frames.begin(), end_it));

            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
                const bool is_downbeat = (phase_idx == 0);
                double obs = beat_log[ci];
                if (use_downbeat && is_downbeat && ci < downbeat_log.size()) {
                    obs = downbeat_log[ci];
                }

                double best_score = obs;
                Backref best_backref;

                const std::size_t prev_phase =
                    (phase_idx + phase_count - 1) % phase_count;
                for (std::size_t cj = start_idx; cj < end_idx; ++cj) {
                    if (cj >= ci) {
                        break;
                    }
                    for (int tempo_delta = -1; tempo_delta <= 1; ++tempo_delta) {
                        const int prev_tempo = static_cast<int>(tempo_idx) + tempo_delta;
                        if (prev_tempo < 0 || prev_tempo >= static_cast<int>(tempo_count)) {
                            continue;
                        }
                        const std::size_t idx =
                            state_index(cj, static_cast<std::size_t>(prev_tempo), prev_phase);
                        const double prev_score = scores[idx];
                        if (prev_score == std::numeric_limits<double>::lowest()) {
                            continue;
                        }
                        const double tempo_penalty =
                            tempo_change_penalty * std::abs(tempo.bpm - tempos[prev_tempo].bpm);
                        const double candidate = prev_score + obs - tempo_penalty + transition_reward;
                        if (candidate > best_score) {
                            best_score = candidate;
                            best_backref.prev_idx = static_cast<int>(cj);
                            best_backref.prev_tempo = prev_tempo;
                        }
                    }
                }

                const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
                scores[idx] = best_score;
                backrefs[idx] = best_backref;
            }
        }
    }

    double best_score = std::numeric_limits<double>::lowest();
    std::size_t best_ci = 0;
    std::size_t best_tempo = 0;
    std::size_t best_phase = 0;
    for (std::size_t ci = 0; ci < candidate_count; ++ci) {
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
                const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
                const double score = scores[idx];
                if (score > best_score) {
                    best_score = score;
                    best_ci = ci;
                    best_tempo = tempo_idx;
                    best_phase = phase_idx;
                }
            }
        }
    }

    std::vector<std::size_t> beat_frames;
    std::vector<std::size_t> downbeat_frames;
    std::size_t ci = best_ci;
    std::size_t tempo_idx = best_tempo;
    std::size_t phase_idx = best_phase;

    while (true) {
        beat_frames.push_back(candidate_frames[ci]);
        if (phase_idx == 0) {
            downbeat_frames.push_back(candidate_frames[ci]);
        }
        const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
        const Backref ref = backrefs[idx];
        if (ref.prev_idx < 0) {
            break;
        }
        ci = static_cast<std::size_t>(ref.prev_idx);
        tempo_idx = static_cast<std::size_t>(ref.prev_tempo);
        phase_idx = (phase_idx + phase_count - 1) % phase_count;
    }

    std::reverse(beat_frames.begin(), beat_frames.end());
    std::reverse(downbeat_frames.begin(), downbeat_frames.end());

    result.decoded.beat_frames = std::move(beat_frames);
    result.decoded.downbeat_frames = std::move(downbeat_frames);
    if (best_tempo < tempos.size()) {
        result.decoded.bpm = tempos[best_tempo].bpm;
    }
    result.best_score = best_score;
    return result;
}

DBNDecodeResult decode_dbn_beats_beatit(const std::vector<float>& beat_activation,
                                        const std::vector<float>& downbeat_activation,
                                        double fps,
                                        float min_bpm,
                                        float max_bpm,
                                        const CoreMLConfig& config,
                                        float reference_bpm) {
    DBNDecodeResult result;
    if (beat_activation.empty() || fps <= 0.0) {
        return result;
    }

    const float bpm_step = std::max(0.1f, config.dbn_bpm_step);
    const float activation_floor = std::max(1e-6f, config.dbn_activation_floor);
    const std::size_t beats_per_bar = std::max<std::size_t>(1, config.dbn_beats_per_bar);
    const float downbeat_weight = std::max(0.0f, config.dbn_downbeat_weight);
    const float tempo_change_penalty = std::max(0.0f, config.dbn_tempo_change_penalty);
    const float tempo_prior_weight = std::max(0.0f, config.dbn_tempo_prior_weight);
    const float transition_reward = config.dbn_transition_reward;
    const std::size_t max_candidates = std::max<std::size_t>(4, config.dbn_max_candidates);
    const double floor_value = std::max(1e-6f, activation_floor);
    const double tolerance = std::max(0.0, static_cast<double>(config.dbn_interval_tolerance));

    const bool use_all_candidates = config.dbn_use_all_candidates;

    std::vector<std::size_t> candidates;
    std::size_t candidate_count = beat_activation.size();
    const std::size_t raw_candidate_count = candidate_count;
    if (!use_all_candidates) {
        candidates.reserve(beat_activation.size());
        for (std::size_t i = 0; i < beat_activation.size(); ++i) {
            if (beat_activation[i] >= activation_floor) {
                candidates.push_back(i);
            }
        }
        candidate_count = candidates.size();
    }

    std::size_t pruned_count = 0;
    if (!use_all_candidates && candidates.size() > max_candidates) {
        std::vector<std::size_t> sorted = candidates;
        std::nth_element(sorted.begin(),
                         sorted.begin() + static_cast<std::ptrdiff_t>(max_candidates),
                         sorted.end(),
                         [&](std::size_t a, std::size_t b) {
                             return beat_activation[a] > beat_activation[b];
                         });
        sorted.resize(max_candidates);
        std::sort(sorted.begin(), sorted.end());
        candidates.swap(sorted);
        pruned_count = candidate_count - candidates.size();
    }

    if (config.dbn_trace) {
        const std::size_t first_window =
            std::min<std::size_t>(beat_activation.size(),
                                  static_cast<std::size_t>(std::llround(2.0 * fps)));
        auto summarize_activation = [&](const std::vector<float>& activation,
                                        const char* label) {
            if (activation.empty() || first_window == 0) {
                std::cerr << "DBN: " << label << " first2s: empty\n";
                return;
            }
            float min_val = activation[0];
            float max_val = activation[0];
            double sum = 0.0;
            std::size_t above = 0;
            for (std::size_t i = 0; i < first_window; ++i) {
                const float v = activation[i];
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
                sum += v;
                if (v >= activation_floor) {
                    ++above;
                }
            }
            const double mean = sum / static_cast<double>(first_window);
            std::cerr << "DBN: " << label << " first2s"
                      << " frames=" << first_window
                      << " min=" << min_val
                      << " max=" << max_val
                      << " mean=" << mean
                      << " above_floor=" << above
                      << " floor=" << activation_floor
                      << "\n";
            std::cerr << "DBN: " << label << " first2s hits:";
            std::size_t printed = 0;
            for (std::size_t i = 0; i < first_window && printed < 8; ++i) {
                const float v = activation[i];
                if (v >= activation_floor) {
                    const double time_s = fps > 0.0 ? static_cast<double>(i) / fps : 0.0;
                    std::cerr << " " << i << "(" << time_s << "s)->" << v;
                    ++printed;
                }
            }
            if (printed == 0) {
                std::cerr << " none";
            }
            std::cerr << "\n";
        };
        summarize_activation(beat_activation, "beat");
        summarize_activation(downbeat_activation, "downbeat");

        auto preview_candidates = [&](const std::vector<std::size_t>& frames,
                                      const std::vector<float>& activation,
                                      const char* label) {
            std::cerr << "DBN: " << label << " candidates head:";
            const std::size_t top = std::min<std::size_t>(8, frames.size());
            for (std::size_t i = 0; i < top; ++i) {
                const std::size_t frame = frames[i];
                const double time_s = fps > 0.0 ? static_cast<double>(frame) / fps : 0.0;
                float value = 0.0f;
                if (frame < activation.size()) {
                    value = activation[frame];
                }
                std::cerr << " " << frame << "(" << time_s << "s)->" << value;
            }
            std::cerr << "\n";
        };
        if (!use_all_candidates) {
            preview_candidates(candidates, beat_activation, "beat");
        } else {
            std::vector<std::size_t> first_hits;
            first_hits.reserve(8);
            for (std::size_t i = 0; i < beat_activation.size() && first_hits.size() < 8; ++i) {
                if (beat_activation[i] >= activation_floor) {
                    first_hits.push_back(i);
                }
            }
            preview_candidates(first_hits, beat_activation, "beat");
        }
        std::vector<std::size_t> downbeat_hits;
        downbeat_hits.reserve(8);
        for (std::size_t i = 0;
             i < downbeat_activation.size() && downbeat_hits.size() < 8;
             ++i) {
            if (downbeat_activation[i] >= activation_floor) {
                downbeat_hits.push_back(i);
            }
        }
        preview_candidates(downbeat_hits, downbeat_activation, "downbeat");
    }

    if (candidate_count < 2) {
        result.beat_frames = viterbi_beats(beat_activation,
                                           fps,
                                           std::max(1.0f, min_bpm),
                                           config.dbn_interval_tolerance,
                                           activation_floor);
        if (!result.beat_frames.empty()) {
            result.downbeat_frames.push_back(result.beat_frames.front());
        }
        return result;
    }

    const std::size_t tempo_count =
        static_cast<std::size_t>(std::floor((max_bpm - min_bpm) / bpm_step)) + 1;
    const std::size_t phase_count = beats_per_bar;

    if (config.verbose) {
        std::cerr << "DBN config: all_candidates="
                  << (use_all_candidates ? "true" : "false")
                  << " raw_candidates=" << raw_candidate_count
                  << " used_candidates=" << candidate_count
                  << " pruned=" << pruned_count
                  << " floor=" << activation_floor
                  << " tol=" << tolerance
                  << " max_cand=" << max_candidates
                  << " bpm=[" << min_bpm << "," << max_bpm << "]"
                  << " step=" << bpm_step
                  << " tempos=" << tempo_count
                  << " bpb=" << beats_per_bar
                  << " reference_bpm=" << reference_bpm
                  << " prior_weight=" << tempo_prior_weight
                  << "\n";
    }

    std::vector<double> beat_log(beat_activation.size(), 0.0);
    for (std::size_t i = 0; i < beat_activation.size(); ++i) {
        beat_log[i] = std::log(std::max<double>(beat_activation[i], floor_value));
    }

    std::vector<double> downbeat_log(downbeat_activation.size(), 0.0);
    for (std::size_t i = 0; i < downbeat_activation.size(); ++i) {
        downbeat_log[i] = std::log(std::max<double>(downbeat_activation[i], floor_value));
    }

    struct TempoParams {
        float bpm = 0.0f;
        std::size_t min_interval = 0;
        std::size_t max_interval = 0;
        double prior_penalty = 0.0;
    };

    std::vector<TempoParams> tempos;
    tempos.reserve(tempo_count);
    for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
        const float bpm = min_bpm + static_cast<float>(tempo_idx) * bpm_step;
        const double interval = (60.0 * fps) / bpm;
        const double min_interval = interval * (1.0 - tolerance);
        const double max_interval = interval * (1.0 + tolerance);
        double prior_penalty = 0.0;
        if (reference_bpm > 0.0f && tempo_prior_weight > 0.0f) {
            prior_penalty = tempo_prior_weight * std::abs(bpm - reference_bpm);
        }
        tempos.push_back({
            bpm,
            static_cast<std::size_t>(std::max(1.0, std::floor(min_interval))),
            static_cast<std::size_t>(std::max(1.0, std::ceil(max_interval))),
            prior_penalty
        });
    }

    struct Backref {
        int prev_idx = -1;
        int prev_tempo = -1;
    };

    const std::size_t state_count = tempo_count * phase_count;
    const std::size_t total_states = candidate_count * state_count;
    std::vector<double> scores(total_states, std::numeric_limits<double>::lowest());
    std::vector<Backref> backrefs(total_states);

    auto state_index = [&](std::size_t cand_idx, std::size_t tempo_idx, std::size_t phase_idx) {
        return (cand_idx * tempo_count + tempo_idx) * phase_count + phase_idx;
    };

    for (std::size_t ci = 0; ci < candidate_count; ++ci) {
        const std::size_t frame = use_all_candidates ? ci : candidates[ci];
        const double beat_obs = beat_log[frame];
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            const auto& tempo = tempos[tempo_idx];
            const std::size_t min_prev_frame =
                (frame > tempo.max_interval) ? frame - tempo.max_interval : 0;
            const std::size_t max_prev_frame =
                (frame > tempo.min_interval) ? frame - tempo.min_interval : 0;
            std::size_t start_idx = 0;
            std::size_t end_idx = 0;
            if (use_all_candidates) {
                start_idx = min_prev_frame;
                end_idx = std::min(max_prev_frame + 1, ci);
            } else {
                const auto start_it =
                    std::lower_bound(candidates.begin(),
                                     candidates.begin() + static_cast<std::ptrdiff_t>(ci),
                                     min_prev_frame);
                const auto end_it =
                    std::upper_bound(candidates.begin(),
                                     candidates.begin() + static_cast<std::ptrdiff_t>(ci),
                                     max_prev_frame);
                start_idx = static_cast<std::size_t>(std::distance(candidates.begin(), start_it));
                end_idx = static_cast<std::size_t>(std::distance(candidates.begin(), end_it));
            }

            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
            const bool is_downbeat = (phase_idx == 0);
            double obs = beat_obs - tempo.prior_penalty;
            if (config.dbn_use_downbeat && is_downbeat && frame < downbeat_log.size()) {
                obs += downbeat_weight * downbeat_log[frame];
            }

                double best_score = obs;
                Backref best_backref;

                const std::size_t prev_phase =
                    (phase_idx + phase_count - 1) % phase_count;
                for (std::size_t cj = start_idx; cj < end_idx; ++cj) {
                    for (int tempo_delta = -1; tempo_delta <= 1; ++tempo_delta) {
                        const int prev_tempo = static_cast<int>(tempo_idx) + tempo_delta;
                        if (prev_tempo < 0 || prev_tempo >= static_cast<int>(tempo_count)) {
                            continue;
                        }
                        const std::size_t idx =
                            state_index(cj, static_cast<std::size_t>(prev_tempo), prev_phase);
                        const double prev_score = scores[idx];
                        if (prev_score == std::numeric_limits<double>::lowest()) {
                            continue;
                        }
                        const double tempo_penalty =
                            tempo_change_penalty * std::abs(tempo.bpm - tempos[prev_tempo].bpm);
                        const double candidate = prev_score + obs - tempo_penalty;
                        const double rewarded = candidate + transition_reward;
                        if (rewarded > best_score) {
                            best_score = rewarded;
                            best_backref.prev_idx = static_cast<int>(cj);
                            best_backref.prev_tempo = prev_tempo;
                        }
                    }
                }

                const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
                scores[idx] = best_score;
                backrefs[idx] = best_backref;
            }
        }
    }

    double best_score = std::numeric_limits<double>::lowest();
    std::size_t best_ci = 0;
    std::size_t best_tempo = 0;
    std::size_t best_phase = 0;
    for (std::size_t ci = 0; ci < candidate_count; ++ci) {
        for (std::size_t tempo_idx = 0; tempo_idx < tempo_count; ++tempo_idx) {
            for (std::size_t phase_idx = 0; phase_idx < phase_count; ++phase_idx) {
                const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
                const double score = scores[idx];
                if (score > best_score) {
                    best_score = score;
                    best_ci = ci;
                    best_tempo = tempo_idx;
                    best_phase = phase_idx;
                }
            }
        }
    }

    std::vector<std::size_t> beat_frames;
    std::vector<std::size_t> downbeat_frames;
    std::size_t ci = best_ci;
    std::size_t tempo_idx = best_tempo;
    std::size_t phase_idx = best_phase;

    while (true) {
        beat_frames.push_back(use_all_candidates ? ci : candidates[ci]);
        if (phase_idx == 0) {
            downbeat_frames.push_back(use_all_candidates ? ci : candidates[ci]);
        }
        const std::size_t idx = state_index(ci, tempo_idx, phase_idx);
        const Backref ref = backrefs[idx];
        if (ref.prev_idx < 0) {
            break;
        }
        ci = static_cast<std::size_t>(ref.prev_idx);
        tempo_idx = static_cast<std::size_t>(ref.prev_tempo);
        phase_idx = (phase_idx + phase_count - 1) % phase_count;
    }

    std::reverse(beat_frames.begin(), beat_frames.end());
    std::reverse(downbeat_frames.begin(), downbeat_frames.end());

    result.beat_frames = std::move(beat_frames);
    result.downbeat_frames = std::move(downbeat_frames);

    if (result.downbeat_frames.empty() && !result.beat_frames.empty()) {
        result.downbeat_frames.push_back(result.beat_frames.front());
    }
    if (best_tempo < tempos.size()) {
        result.bpm = tempos[best_tempo].bpm;
    }

    return result;
}

DBNDecodeResult decode_dbn_beats_calmdad(const std::vector<float>& beat_activation,
                                         const std::vector<float>& downbeat_activation,
                                         double fps,
                                         float min_bpm,
                                         float max_bpm,
                                         float bpm_step,
                                         const CoreMLConfig& config) {
    DBNDecodeResult best;
    if (beat_activation.empty() || fps <= 0.0) {
        return best;
    }

    const float local_step = std::max(0.1f, bpm_step);
    const float local_min = std::max(1.0f, min_bpm);
    const float local_max = std::max(local_min + local_step, max_bpm);
    const double tolerance = std::max(0.0, static_cast<double>(config.dbn_interval_tolerance));
    const double epsilon = 1e-5;
    const double floor_value = epsilon / 2.0;
    const double transition_lambda = std::max(1e-6, static_cast<double>(config.dbn_transition_lambda));
    const double transition_reward = std::log(transition_lambda);
    const double tempo_change_penalty = transition_reward;
    const bool use_downbeat = config.dbn_use_downbeat;

    std::vector<double> beat_log(beat_activation.size(), 0.0);
    std::vector<double> downbeat_log(downbeat_activation.size(), 0.0);
    double beat_min = std::numeric_limits<double>::infinity();
    double beat_max = 0.0;
    double downbeat_min = std::numeric_limits<double>::infinity();
    double downbeat_max = 0.0;
    double combined_max = 0.0;
    std::size_t beat_above_floor = 0;
    std::size_t downbeat_above_floor = 0;
    for (std::size_t i = 0; i < beat_activation.size(); ++i) {
        const float raw_downbeat =
            (i < downbeat_activation.size()) ? downbeat_activation[i] : 0.0f;
        const double beat_value =
            static_cast<double>(beat_activation[i]) * (1.0 - epsilon) + floor_value;
        const double downbeat_value =
            static_cast<double>(raw_downbeat) * (1.0 - epsilon) + floor_value;
        const double combined_beat = std::max(floor_value, beat_value - downbeat_value);
        beat_min = std::min<double>(beat_min, beat_value);
        beat_max = std::max<double>(beat_max, beat_value);
        downbeat_min = std::min<double>(downbeat_min, downbeat_value);
        downbeat_max = std::max<double>(downbeat_max, downbeat_value);
        combined_max = std::max<double>(combined_max, combined_beat);
        if (combined_beat > floor_value) {
            ++beat_above_floor;
        }
        beat_log[i] = std::log(combined_beat);
        if (i < downbeat_activation.size()) {
            downbeat_log[i] = std::log(downbeat_value);
            if (downbeat_value > floor_value) {
                ++downbeat_above_floor;
            }
        }
    }

    std::vector<std::size_t> bpb_options = {3, 4};
    DBNPathResult best_path;
    std::size_t best_bpb = 0;
    if (config.verbose) {
        std::cerr << "DBN calmdad: frames=" << beat_activation.size()
                  << " floor=" << floor_value
                  << " epsilon=" << epsilon
                  << " tol=" << tolerance
                  << " bpm=[" << local_min << "," << local_max << "]"
                  << " step=" << local_step
                  << " lambda=" << transition_lambda
                  << " use_downbeat=" << (use_downbeat ? "true" : "false")
                  << " beat[min,max]=[" << beat_min << "," << beat_max << "]"
                  << " downbeat[min,max]=[" << downbeat_min << "," << downbeat_max << "]"
                  << " combined_max=" << combined_max
                  << " beat>floor=" << beat_above_floor
                  << " downbeat>floor=" << downbeat_above_floor
                  << "\n";
        const std::size_t dump_count = 10;
        std::size_t emitted = 0;
        std::cerr << "DBN calmdad: first beat candidates (frame->s):";
        for (std::size_t i = 0; i < beat_activation.size() && emitted < dump_count; ++i) {
            const double combined = std::exp(beat_log[i]);
            if (combined > floor_value) {
                std::cerr << " " << i << "->" << (static_cast<double>(i) / fps);
                ++emitted;
            }
        }
        if (emitted == 0) {
            std::cerr << " none";
        }
        std::cerr << "\n";
        emitted = 0;
        std::cerr << "DBN calmdad: first downbeat candidates (frame->s):";
        for (std::size_t i = 0; i < downbeat_log.size() && emitted < dump_count; ++i) {
            const double downbeat_val = std::exp(downbeat_log[i]);
            if (downbeat_val > floor_value) {
                std::cerr << " " << i << "->" << (static_cast<double>(i) / fps);
                ++emitted;
            }
        }
        if (emitted == 0) {
            std::cerr << " none";
        }
        std::cerr << "\n";
    }
    for (std::size_t bpb : bpb_options) {
        DBNPathResult path = decode_dbn_beats_candidate(
            beat_log,
            downbeat_log,
            fps,
            local_min,
            local_max,
            local_step,
            bpb,
            tolerance,
            use_downbeat,
            transition_reward,
            tempo_change_penalty,
            config);
        if (path.best_score > best_path.best_score) {
            best_path = std::move(path);
            best_bpb = bpb;
        }
    }

    best = std::move(best_path.decoded);
    if (config.verbose) {
        std::cerr << "DBN calmdad: best_bpb=" << best_bpb
                  << " beats=" << best.beat_frames.size()
                  << " downbeats=" << best.downbeat_frames.size()
                  << " best_score=" << best_path.best_score
                  << "\n";
        const std::size_t dump_count = 10;
        std::cerr << "DBN calmdad: first beats (frame->s):";
        for (std::size_t i = 0; i < best.beat_frames.size() && i < dump_count; ++i) {
            const auto frame = best.beat_frames[i];
            std::cerr << " " << frame << "->" << (static_cast<double>(frame) / fps);
        }
        if (best.beat_frames.empty()) {
            std::cerr << " none";
        }
        std::cerr << "\n";
        std::cerr << "DBN calmdad: first downbeats (frame->s):";
        for (std::size_t i = 0; i < best.downbeat_frames.size() && i < dump_count; ++i) {
            const auto frame = best.downbeat_frames[i];
            std::cerr << " " << frame << "->" << (static_cast<double>(frame) / fps);
        }
        if (best.downbeat_frames.empty()) {
            std::cerr << " none";
        }
        std::cerr << "\n";
    }
    return best;
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

    const bool should_profile =
        config.profile && (config.fixed_frames == 0 || config.profile_per_window);
    const auto total_start = std::chrono::steady_clock::now();

    const auto resample_start = std::chrono::steady_clock::now();
    std::vector<float> resampled = resample_linear(samples, sample_rate, config.sample_rate);
    const auto resample_end = std::chrono::steady_clock::now();
    if (resampled.size() < config.frame_size) {
        return result;
    }

    const std::size_t frames = 1 + (resampled.size() - config.frame_size) / config.hop_size;
    const auto mel_start = std::chrono::steady_clock::now();
    std::vector<float> features = compute_mel(resampled,
                                              config.sample_rate,
                                              config.frame_size,
                                              config.hop_size,
                                              config.mel_bins,
                                              config.use_log_mel,
                                              config.log_multiplier,
                                              config.f_min,
                                              config.f_max,
                                              config.mel_scale,
                                              config.spectrogram_norm,
                                              config.power);
    const auto mel_end = std::chrono::steady_clock::now();
    if (features.empty()) {
        return result;
    }

    std::vector<float> phase_energy(frames, 0.0f);
    if (config.mel_bins > 0 && config.sample_rate > 0.0) {
        const float mel_min = hz_to_mel(std::max(0.0f, config.f_min), config.mel_scale);
        const float mel_max = hz_to_mel(std::max(config.f_min + 1.0f, config.f_max),
                                        config.mel_scale);
        std::vector<std::size_t> low_bins;
        low_bins.reserve(config.mel_bins);
        for (std::size_t m = 0; m < config.mel_bins; ++m) {
            const float t =
                (static_cast<float>(m) + 0.5f) / static_cast<float>(config.mel_bins);
            const float mel = mel_min + t * (mel_max - mel_min);
            const float hz = mel_to_hz(mel, config.mel_scale);
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
    NSString* model_path = resolve_model_path(config);
    if (!model_path) {
        if (config.verbose) {
            std::cerr << "CoreML model not found on disk or in bundle.\n";
        }
        return result;
    }

    NSURL* model_url = [NSURL fileURLWithPath:model_path];
    NSURL* compiled_url = compile_model_if_needed(model_url, &error);
    if (error && config.verbose) {
        std::cerr << "CoreML compile error: " << error.localizedDescription.UTF8String << "\n";
    }
    if (compiled_url) {
        model_url = compiled_url;
    }
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
    MLModel* model = load_cached_model(model_url, model_config, &error);
    if (!model) {
        if (config.verbose && error) {
            std::cerr << "CoreML load error: " << error.localizedDescription.UTF8String << "\n";
        }
        return result;
    }
    const auto model_end = std::chrono::steady_clock::now();
    const CoreMLConfig::InputLayout inferred_input_layout = infer_model_input_layout(model, config);

    auto run_inference = [&](const std::vector<float>& window_features,
                             std::size_t window_frames,
                             std::vector<float>* beat_out,
                             std::vector<float>* downbeat_out) -> bool {
        auto try_layout = [&](CoreMLConfig::InputLayout layout) -> bool {
            error = nil;
            MLMultiArray* input_array = nil;
            if (layout == CoreMLConfig::InputLayout::FramesByMels) {
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
                                               layout)) {
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
                    std::cerr << "CoreML input provider error: "
                              << error.localizedDescription.UTF8String << "\n";
                }
                return false;
            }

            id<MLFeatureProvider> output = [model predictionFromFeatures:input error:&error];
            if (!output) {
                if (config.verbose && error) {
                    std::cerr << "CoreML inference error: " << error.localizedDescription.UTF8String
                              << "\n";
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
        const CoreMLConfig::InputLayout fallback_layout =
            (inferred_input_layout == CoreMLConfig::InputLayout::FramesByMels)
                ? CoreMLConfig::InputLayout::ChannelsFramesMels
                : CoreMLConfig::InputLayout::FramesByMels;
        if (fallback_layout != inferred_input_layout) {
            if (config.verbose) {
                std::cerr << "CoreML retrying inference with alternate input layout.\n";
            }
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
            return result;
        }
        const auto infer_end = std::chrono::steady_clock::now();
        infer_ms +=
            std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    }

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

    if (should_profile) {
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
        std::cerr << "Timing(coreml): resample=" << resample_ms
                  << "ms mel=" << mel_ms
                  << "ms model=" << model_ms
                  << "ms infer=" << infer_ms
                  << "ms post=" << post_ms
                  << "ms total=" << total_ms
                  << "ms\n";
    }

    return result;
}

CoreMLResult postprocess_coreml_activations(const std::vector<float>& beat_activation,
                                            const std::vector<float>& downbeat_activation,
                                            const std::vector<float>* phase_energy,
                                            const CoreMLConfig& config,
                                            double sample_rate,
                                            float reference_bpm,
                                            std::size_t last_active_frame,
                                            std::size_t total_frames_full) {
    double dbn_ms = 0.0;
    double peaks_ms = 0.0;

    CoreMLResult result;
    result.beat_activation = beat_activation;
    result.downbeat_activation = downbeat_activation;

    const std::size_t used_frames = result.beat_activation.size();
    if (used_frames == 0) {
        return result;
    }
    const std::size_t grid_total_frames =
        total_frames_full > used_frames ? total_frames_full : used_frames;

    const float hard_min_bpm = std::max(1.0f, config.min_bpm);
    const float hard_max_bpm = std::max(hard_min_bpm + 1.0f, config.max_bpm);
    auto clamp_bpm_range = [&](float* min_value, float* max_value) {
        *min_value = std::max(hard_min_bpm, *min_value);
        *max_value = std::min(hard_max_bpm, *max_value);
        if (*max_value <= *min_value) {
            *min_value = hard_min_bpm;
            *max_value = hard_max_bpm;
        }
    };

    float min_bpm = hard_min_bpm;
    float max_bpm = hard_max_bpm;
    float min_bpm_alt = min_bpm;
    float max_bpm_alt = max_bpm;
    bool has_window = false;
    if (config.tempo_window_percent > 0.0f && reference_bpm > 0.0f) {
        const float window = config.tempo_window_percent / 100.0f;
        min_bpm = reference_bpm * (1.0f - window);
        max_bpm = reference_bpm * (1.0f + window);
        if (config.prefer_double_time) {
            const float doubled = reference_bpm * 2.0f;
            min_bpm_alt = doubled * (1.0f - window);
            max_bpm_alt = doubled * (1.0f + window);
            has_window = true;
        }
        clamp_bpm_range(&min_bpm, &max_bpm);
        clamp_bpm_range(&min_bpm_alt, &max_bpm_alt);
    }

    const double fps = static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size);
    const double hop_scale = sample_rate / static_cast<double>(config.sample_rate);
    const bool windowed_inference =
        config.fixed_frames > 0 && used_frames > config.fixed_frames;
    const std::size_t analysis_latency_frames =
        windowed_inference ? std::min(config.window_border_frames,
                                      config.fixed_frames / 2)
                           : 0;
    const double analysis_latency_frames_f =
        static_cast<double>(analysis_latency_frames);

    if (config.debug_activations_start_s >= 0.0 &&
        config.debug_activations_end_s > config.debug_activations_start_s) {
        const std::size_t start_frame = static_cast<std::size_t>(
            std::max(0.0, std::floor(config.debug_activations_start_s * fps)));
        const std::size_t end_frame = static_cast<std::size_t>(
            std::min<double>(used_frames - 1,
                             std::ceil(config.debug_activations_end_s * fps)));
        const double epsilon = 1e-5;
        const double floor_value = epsilon / 2.0;
        std::size_t emitted = 0;

        std::cerr << "Activation window: start=" << config.debug_activations_start_s
                  << "s end=" << config.debug_activations_end_s
                  << "s fps=" << fps
                  << " hop=" << config.hop_size
                  << " hop_scale=" << hop_scale
                  << " frames=[" << start_frame << "," << end_frame << "]\n";
        std::cerr << "Activations (frame,time_s,sample_frame,"
                  << "beat_raw,downbeat_raw,beat_prob,downbeat_prob,combined)\n";
        std::cerr << std::fixed << std::setprecision(6);
        for (std::size_t frame = start_frame; frame <= end_frame; ++frame) {
            const float beat_raw = result.beat_activation[frame];
            const float downbeat_raw =
                (frame < result.downbeat_activation.size()) ? result.downbeat_activation[frame] : 0.0f;
            double beat_prob = beat_raw;
            double downbeat_prob = downbeat_raw;
            double combined = beat_raw;
            if (config.dbn_mode == CoreMLConfig::DBNMode::Calmdad) {
                beat_prob = static_cast<double>(beat_raw) * (1.0 - epsilon) + floor_value;
                downbeat_prob = static_cast<double>(downbeat_raw) * (1.0 - epsilon) + floor_value;
                combined = std::max(floor_value, beat_prob - downbeat_prob);
            }
            const double time_s = static_cast<double>(frame) / fps;
            const double sample_pos = static_cast<double>(frame * config.hop_size) * hop_scale;
            std::cerr << frame << "," << time_s << "," << static_cast<unsigned long long>(std::llround(sample_pos))
                      << "," << beat_raw << "," << downbeat_raw
                      << "," << beat_prob << "," << downbeat_prob << "," << combined << "\n";
            if (config.debug_activations_max > 0 &&
                ++emitted >= config.debug_activations_max) {
                std::cerr << "Activations truncated at " << emitted << " rows\n";
                break;
            }
        }
        std::cerr << std::defaultfloat;
    }

    constexpr std::size_t kRefineWindow = 2;
    auto refine_frame_to_peak = [&](std::size_t frame,
                                    const std::vector<float>& activation) -> std::size_t {
        if (activation.empty()) {
            return frame;
        }
        const std::size_t start = frame > kRefineWindow ? frame - kRefineWindow : 0;
        const std::size_t end = std::min(frame + kRefineWindow, activation.size() - 1);
        std::size_t best_index = frame;
        float best_value = activation[frame];
        for (std::size_t i = start; i <= end; ++i) {
            const float value = activation[i];
            if (value > best_value) {
                best_value = value;
                best_index = i;
            }
        }
        return best_index;
    };

    auto fill_beats_from_frames = [&](const std::vector<std::size_t>& frames) {
        result.beat_feature_frames.clear();
        result.beat_feature_frames.reserve(frames.size());
        result.beat_sample_frames.clear();
        result.beat_sample_frames.reserve(frames.size());
        result.beat_strengths.clear();
        result.beat_strengths.reserve(frames.size());

        const long long latency_samples = static_cast<long long>(
            std::llround(config.output_latency_seconds * sample_rate));

        for (std::size_t frame : frames) {
            const std::size_t output_frame =
                (analysis_latency_frames > 0 && frame > analysis_latency_frames)
                    ? (frame - analysis_latency_frames)
                    : (analysis_latency_frames > 0 ? 0 : frame);
            result.beat_feature_frames.push_back(static_cast<unsigned long long>(output_frame));
            const std::size_t peak_frame = refine_frame_to_peak(frame, result.beat_activation);

            double frame_pos = static_cast<double>(peak_frame);
            if (peak_frame > 0 && peak_frame + 1 < result.beat_activation.size()) {
                const double prev = result.beat_activation[peak_frame - 1];
                const double curr = result.beat_activation[peak_frame];
                const double next = result.beat_activation[peak_frame + 1];
                const double denom = prev - 2.0 * curr + next;
                if (std::abs(denom) > 1e-9) {
                    double offset = 0.5 * (prev - next) / denom;
                    offset = std::max(-0.5, std::min(0.5, offset));
                    frame_pos += offset;
                }
            }
            if (analysis_latency_frames > 0) {
                frame_pos = std::max(0.0, frame_pos - analysis_latency_frames_f);
            }
            const double sample_pos =
                (frame_pos * static_cast<double>(config.hop_size)) * hop_scale;
            long long sample_frame = static_cast<long long>(std::llround(sample_pos)) - latency_samples;
            if (sample_frame < 0) {
                sample_frame = 0;
            }
            result.beat_sample_frames.push_back(static_cast<unsigned long long>(sample_frame));
            if (!result.beat_activation.empty()) {
                result.beat_strengths.push_back(result.beat_activation[peak_frame]);
            } else {
                result.beat_strengths.push_back(0.0f);
            }
        }

        if (!result.beat_sample_frames.empty()) {
            std::size_t write = 1;
            unsigned long long last = result.beat_sample_frames[0];
            for (std::size_t i = 1; i < result.beat_sample_frames.size(); ++i) {
                const unsigned long long current = result.beat_sample_frames[i];
                if (current <= last) {
                    continue;
                }
                result.beat_sample_frames[write] = current;
                result.beat_feature_frames[write] = result.beat_feature_frames[i];
                result.beat_strengths[write] = result.beat_strengths[i];
                last = current;
                ++write;
            }
            result.beat_sample_frames.resize(write);
            result.beat_feature_frames.resize(write);
            result.beat_strengths.resize(write);
        }
    };

    auto fill_beats_from_frames_raw = [&](const std::vector<std::size_t>& frames) {
        result.beat_feature_frames.clear();
        result.beat_feature_frames.reserve(frames.size());
        result.beat_sample_frames.clear();
        result.beat_sample_frames.reserve(frames.size());
        result.beat_strengths.clear();
        result.beat_strengths.reserve(frames.size());

        const long long latency_samples = static_cast<long long>(
            std::llround(config.output_latency_seconds * sample_rate));

        for (std::size_t frame : frames) {
            const std::size_t output_frame =
                (analysis_latency_frames > 0 && frame > analysis_latency_frames)
                    ? (frame - analysis_latency_frames)
                    : (analysis_latency_frames > 0 ? 0 : frame);
            result.beat_feature_frames.push_back(static_cast<unsigned long long>(output_frame));
            double frame_pos = static_cast<double>(frame);
            if (analysis_latency_frames > 0) {
                frame_pos = std::max(0.0, frame_pos - analysis_latency_frames_f);
            }
            const double sample_pos =
                (frame_pos * static_cast<double>(config.hop_size)) * hop_scale;
            long long sample_frame = static_cast<long long>(std::llround(sample_pos)) - latency_samples;
            if (sample_frame < 0) {
                sample_frame = 0;
            }
            result.beat_sample_frames.push_back(static_cast<unsigned long long>(sample_frame));
            if (!result.beat_activation.empty() && frame < result.beat_activation.size()) {
                result.beat_strengths.push_back(result.beat_activation[frame]);
            } else {
                result.beat_strengths.push_back(0.0f);
            }
        }

        if (!result.beat_sample_frames.empty()) {
            std::size_t write = 1;
            unsigned long long last = result.beat_sample_frames[0];
            for (std::size_t i = 1; i < result.beat_sample_frames.size(); ++i) {
                const unsigned long long current = result.beat_sample_frames[i];
                if (current <= last) {
                    continue;
                }
                result.beat_sample_frames[write] = current;
                result.beat_feature_frames[write] = result.beat_feature_frames[i];
                result.beat_strengths[write] = result.beat_strengths[i];
                last = current;
                ++write;
            }
            result.beat_sample_frames.resize(write);
            result.beat_feature_frames.resize(write);
            result.beat_strengths.resize(write);
        }
    };

    auto fill_beats_from_bpm_grid_into = [&](std::size_t start_frame,
                                             double bpm,
                                             std::size_t total_frames,
                                             std::vector<unsigned long long>& out_feature_frames,
                                             std::vector<unsigned long long>& out_sample_frames,
                                             std::vector<float>& out_strengths) {
        out_feature_frames.clear();
        out_sample_frames.clear();
        out_strengths.clear();

        if (bpm <= 0.0 || fps <= 0.0 || total_frames == 0) {
            return;
        }

        const double step_frames = (60.0 * fps) / bpm;
        if (step_frames <= 0.0) {
            return;
        }

        const double start_frame_adjusted =
            analysis_latency_frames > 0
                ? std::max(0.0, static_cast<double>(start_frame) - analysis_latency_frames_f)
                : static_cast<double>(start_frame);

        if (config.dbn_trace) {
            const double start_time = start_frame_adjusted / fps;
            const double start_sample_pos =
                (start_frame_adjusted * static_cast<double>(config.hop_size)) * hop_scale;
            const long long start_sample_frame =
                static_cast<long long>(std::llround(start_sample_pos));
            const double start_time_after_latency =
                sample_rate > 0.0
                    ? static_cast<double>(std::max<long long>(0, start_sample_frame)) /
                        sample_rate
                    : 0.0;
            std::cerr << "DBN grid project: start_frame=" << start_frame
                      << " start_time=" << start_time
                      << " bpm=" << bpm
                      << " step_frames=" << step_frames
                      << " total_frames=" << total_frames
                      << " hop_size=" << config.hop_size
                      << " hop_scale=" << hop_scale
                      << " start_sample_frame=" << start_sample_frame
                      << " start_time_adj=" << start_time_after_latency
                      << "\n";
        }

        const double step_samples = (60.0 * sample_rate) / bpm;
        if (step_samples <= 0.0) {
            return;
        }
        const double start_sample_pos =
            (start_frame_adjusted * static_cast<double>(config.hop_size)) * hop_scale;
        const long long start_sample_frame = static_cast<long long>(std::llround(start_sample_pos));
        std::vector<unsigned long long> grid_samples;
        grid_samples.reserve(static_cast<std::size_t>(
            std::ceil(static_cast<double>(total_frames) / step_frames)) + 4);
        std::size_t backward_count = 0;
        std::size_t forward_count = 0;

        double sample_pos = static_cast<double>(start_sample_frame);
        while (sample_pos >= step_samples) {
            sample_pos -= step_samples;
            grid_samples.push_back(static_cast<unsigned long long>(
                std::llround(sample_pos)));
        }
        std::reverse(grid_samples.begin(), grid_samples.end());
        backward_count = grid_samples.size();
        grid_samples.push_back(static_cast<unsigned long long>(start_sample_frame));

        sample_pos = static_cast<double>(start_sample_frame) + step_samples;
        const double total_samples = static_cast<double>(total_frames) *
            static_cast<double>(config.hop_size) * hop_scale;
        while (sample_pos < total_samples) {
            grid_samples.push_back(static_cast<unsigned long long>(
                std::llround(sample_pos)));
            ++forward_count;
            sample_pos += step_samples;
        }

        out_feature_frames.reserve(grid_samples.size());
        out_sample_frames.reserve(grid_samples.size());
        out_strengths.reserve(grid_samples.size());

        for (unsigned long long sample_frame : grid_samples) {
            out_sample_frames.push_back(sample_frame);
            const double feature_pos =
                (static_cast<double>(sample_frame) / hop_scale) /
                static_cast<double>(config.hop_size);
            const std::size_t frame = static_cast<std::size_t>(std::llround(feature_pos));
            out_feature_frames.push_back(static_cast<unsigned long long>(frame));

            if (!result.beat_activation.empty() && frame < result.beat_activation.size()) {
                out_strengths.push_back(result.beat_activation[frame]);
            } else {
                out_strengths.push_back(0.0f);
            }
        }

        if (!out_sample_frames.empty()) {
            std::size_t write = 1;
            unsigned long long last = out_sample_frames[0];
            for (std::size_t i = 1; i < out_sample_frames.size(); ++i) {
                const unsigned long long current = out_sample_frames[i];
                if (current <= last) {
                    continue;
                }
                out_sample_frames[write] = current;
                out_feature_frames[write] = out_feature_frames[i];
                out_strengths[write] = out_strengths[i];
                last = current;
                ++write;
            }
            out_sample_frames.resize(write);
            out_feature_frames.resize(write);
            out_strengths.resize(write);
        }

        if (config.dbn_trace) {
            const std::size_t preview = std::min<std::size_t>(6, out_feature_frames.size());
            std::cerr << "DBN grid beats head:";
            for (std::size_t i = 0; i < preview; ++i) {
                const std::size_t frame = static_cast<std::size_t>(out_feature_frames[i]);
                const double time_s = static_cast<double>(frame) / fps;
                std::cerr << " " << frame << "(" << time_s << "s)";
            }
            std::cerr << "\n";
            std::cerr << "DBN grid beats total=" << out_feature_frames.size()
                      << " backward=" << backward_count
                      << " forward=" << forward_count
                      << "\n";
        }
    };

    auto dedupe_frames = [&](std::vector<std::size_t>& frames) {
        if (frames.empty()) {
            return;
        }
        std::size_t write = 1;
        std::size_t last = frames[0];
        for (std::size_t i = 1; i < frames.size(); ++i) {
            const std::size_t current = frames[i];
            if (current <= last) {
                continue;
            }
            frames[write++] = current;
            last = current;
        }
        frames.resize(write);
    };

    auto dedupe_frames_tolerant = [&](std::vector<std::size_t>& frames,
                                      std::size_t tolerance) {
        if (frames.empty()) {
            return;
        }
        if (tolerance == 0) {
            dedupe_frames(frames);
            return;
        }
        std::size_t write = 1;
        std::size_t last = frames[0];
        for (std::size_t i = 1; i < frames.size(); ++i) {
            const std::size_t current = frames[i];
            if (current <= last + tolerance) {
                continue;
            }
            frames[write++] = current;
            last = current;
        }
        frames.resize(write);
    };

    auto apply_latency_to_frames = [&](const std::vector<std::size_t>& frames) {
        if (analysis_latency_frames == 0 || frames.empty()) {
            return frames;
        }
        std::vector<std::size_t> adjusted;
        adjusted.reserve(frames.size());
        for (std::size_t frame : frames) {
            if (frame > analysis_latency_frames) {
                adjusted.push_back(frame - analysis_latency_frames);
            } else {
                adjusted.push_back(0);
            }
        }
        dedupe_frames(adjusted);
        return adjusted;
    };

    auto window_tempo_score = [&](const std::vector<float>& activation,
                                  std::size_t start,
                                  std::size_t end,
                                  float min_bpm,
                                  float max_bpm,
                                  float peak_threshold) {
        if (end <= start || activation.empty() || fps <= 0.0) {
            return 0.0;
        }
        const double min_interval_frames =
            std::max(1.0, (60.0 * fps) / std::max(1.0f, max_bpm));
        const double max_interval_frames =
            std::max(1.0, (60.0 * fps) / std::max(1.0f, min_bpm));
        const std::size_t peak_min_interval =
            static_cast<std::size_t>(std::max(1.0, std::floor(min_interval_frames)));
        const std::size_t peak_max_interval =
            static_cast<std::size_t>(std::max<double>(peak_min_interval,
                                                      std::ceil(max_interval_frames)));

        std::vector<float> slice;
        slice.reserve(end - start);
        for (std::size_t i = start; i < end; ++i) {
            slice.push_back(activation[i]);
        }

        std::vector<std::size_t> peaks =
            pick_peaks(slice, peak_threshold, peak_min_interval, peak_max_interval);
        if (peaks.size() < 4) {
            return 0.0;
        }
        std::vector<double> intervals;
        intervals.reserve(peaks.size() - 1);
        for (std::size_t i = 1; i < peaks.size(); ++i) {
            if (peaks[i] > peaks[i - 1]) {
                intervals.push_back(static_cast<double>(peaks[i] - peaks[i - 1]));
            }
        }
        if (intervals.empty()) {
            return 0.0;
        }
        std::nth_element(intervals.begin(),
                         intervals.begin() + intervals.size() / 2,
                         intervals.end());
        const double median = intervals[intervals.size() / 2];
        if (median <= 1.0) {
            return 0.0;
        }
        std::vector<double> deviations;
        deviations.reserve(intervals.size());
        for (double v : intervals) {
            deviations.push_back(std::abs(v - median));
        }
        std::nth_element(deviations.begin(),
                         deviations.begin() + deviations.size() / 2,
                         deviations.end());
        const double mad = deviations[deviations.size() / 2];
        const double consistency = 1.0 / (1.0 + (mad / median));
        return static_cast<double>(peaks.size()) * consistency;
    };

    auto select_dbn_window = [&](const std::vector<float>& activation,
                                 double window_seconds,
                                 bool intro_mid_outro,
                                 float min_bpm,
                                 float max_bpm,
                                 float peak_threshold) -> std::pair<std::size_t, std::size_t> {
        if (activation.empty() || window_seconds <= 0.0 || fps <= 0.0) {
            return {0, activation.size()};
        }
        const std::size_t total_frames = activation.size();
        std::size_t window_frames =
            static_cast<std::size_t>(std::max(1.0, std::round(window_seconds * fps)));
        if (window_frames >= total_frames) {
            return {0, total_frames};
        }

        auto clamp_window = [&](std::size_t start) {
            const std::size_t end = std::min(total_frames, start + window_frames);
            return std::make_pair(start, end);
        };

        if (intro_mid_outro && total_frames > window_frames) {
            const std::size_t intro_start = 0;
            const std::size_t mid_center = total_frames / 2;
            const std::size_t mid_start =
                (mid_center > (window_frames / 2)) ? (mid_center - (window_frames / 2)) : 0;
            const std::size_t outro_start = total_frames - window_frames;

            const auto intro = clamp_window(intro_start);
            const auto mid = clamp_window(mid_start);
            const auto outro = clamp_window(outro_start);

            std::array<std::pair<std::size_t, std::size_t>, 3> windows = {intro, mid, outro};
            double best_score = -1.0;
            std::pair<std::size_t, std::size_t> best = intro;
            for (const auto& w : windows) {
                const double score =
                    window_tempo_score(activation,
                                       w.first,
                                       w.second,
                                       min_bpm,
                                       max_bpm,
                                       peak_threshold);
                if (score > best_score) {
                    best_score = score;
                    best = w;
                }
            }
            return best;
        }

        const std::size_t step = std::max<std::size_t>(1, window_frames / 4);
        double best_score = -1.0;
        std::size_t best_start = 0;
        for (std::size_t start = 0; start + window_frames <= total_frames; start += step) {
            const double score =
                window_tempo_score(activation,
                                   start,
                                   start + window_frames,
                                   min_bpm,
                                   max_bpm,
                                   peak_threshold);
            if (score > best_score) {
                best_score = score;
                best_start = start;
            }
        }
        if (best_score <= 1e-6) {
            return {0, total_frames};
        }
        return {best_start, best_start + window_frames};
    };

    auto select_dbn_window_energy = [&](const std::vector<float>& energy,
                                        double window_seconds,
                                        bool intro_mid_outro) -> std::pair<std::size_t, std::size_t> {
        if (energy.empty() || window_seconds <= 0.0 || fps <= 0.0) {
            return {0, energy.size()};
        }
        const std::size_t total_frames = energy.size();
        const std::size_t window_frames =
            static_cast<std::size_t>(std::max(1.0, std::round(window_seconds * fps)));
        if (window_frames >= total_frames) {
            return {0, total_frames};
        }

        auto clamp_window = [&](std::size_t start) {
            const std::size_t end = std::min(total_frames, start + window_frames);
            return std::make_pair(start, end);
        };

        auto mean_energy = [&](std::size_t start, std::size_t end) {
            double sum = 0.0;
            for (std::size_t i = start; i < end; ++i) {
                sum += static_cast<double>(energy[i]);
            }
            const double denom = std::max<std::size_t>(1, end - start);
            return sum / static_cast<double>(denom);
        };

        if (intro_mid_outro && total_frames > window_frames) {
            const std::size_t intro_start = 0;
            const std::size_t mid_center = total_frames / 2;
            const std::size_t mid_start =
                (mid_center > (window_frames / 2)) ? (mid_center - (window_frames / 2)) : 0;
            const std::size_t outro_start = total_frames - window_frames;

            const auto intro = clamp_window(intro_start);
            const auto mid = clamp_window(mid_start);
            const auto outro = clamp_window(outro_start);

            std::array<std::pair<std::size_t, std::size_t>, 3> windows = {intro, mid, outro};
            double best_score = -1.0;
            std::pair<std::size_t, std::size_t> best = intro;
            for (const auto& w : windows) {
                const double score = mean_energy(w.first, w.second);
                if (score > best_score) {
                    best_score = score;
                    best = w;
                }
            }
            return best;
        }

        const std::size_t step = std::max<std::size_t>(1, window_frames / 4);
        double best_score = -1.0;
        std::size_t best_start = 0;
        for (std::size_t start = 0; start + window_frames <= total_frames; start += step) {
            const double score = mean_energy(start, start + window_frames);
            if (score > best_score) {
                best_score = score;
                best_start = start;
            }
        }
        if (best_score <= 1e-9) {
            return {0, total_frames};
        }
        return {best_start, best_start + window_frames};
    };

    auto deduplicate_peaks = [](const std::vector<std::size_t>& peaks, std::size_t width) {
        std::vector<std::size_t> result;
        if (peaks.empty()) {
            return result;
        }
        double p = static_cast<double>(peaks.front());
        std::size_t count = 1;
        for (std::size_t i = 1; i < peaks.size(); ++i) {
            const double next = static_cast<double>(peaks[i]);
            if (next - p <= static_cast<double>(width)) {
                ++count;
                p += (next - p) / static_cast<double>(count);
            } else {
                result.push_back(static_cast<std::size_t>(std::llround(p)));
                p = next;
                count = 1;
            }
        }
        result.push_back(static_cast<std::size_t>(std::llround(p)));
        return result;
    };

    auto compute_minimal_peaks = [&](const std::vector<float>& activation) {
        constexpr std::size_t window = 7;
        constexpr std::size_t half = window / 2;
        constexpr float threshold = 0.5f;
        std::vector<std::size_t> peaks;
        if (activation.empty()) {
            return peaks;
        }
        peaks.reserve(activation.size() / 10);
        for (std::size_t i = 0; i < activation.size(); ++i) {
            const float value = activation[i];
            if (value <= threshold) {
                continue;
            }
            const std::size_t start = (i > half) ? i - half : 0;
            const std::size_t end = std::min(activation.size() - 1, i + half);
            float local_max = value;
            for (std::size_t j = start; j <= end; ++j) {
                local_max = std::max(local_max, activation[j]);
            }
            if (value >= local_max) {
                peaks.push_back(i);
            }
        }
        return deduplicate_peaks(peaks, 1);
    };

    auto align_downbeats_to_beats = [&](const std::vector<std::size_t>& beats,
                                        const std::vector<std::size_t>& downbeats) {
        if (beats.empty()) {
            return downbeats;
        }
        std::vector<std::size_t> aligned;
        aligned.reserve(downbeats.size());
        for (std::size_t db : downbeats) {
            std::size_t best = beats.front();
            std::size_t best_dist = (db > best) ? (db - best) : (best - db);
            for (std::size_t beat : beats) {
                const std::size_t dist = (db > beat) ? (db - beat) : (beat - db);
                if (dist < best_dist) {
                    best = beat;
                    best_dist = dist;
                }
            }
            aligned.push_back(best);
        }
        std::sort(aligned.begin(), aligned.end());
        aligned.erase(std::unique(aligned.begin(), aligned.end()), aligned.end());
        return aligned;
    };

    auto infer_bpb_phase = [&](const std::vector<std::size_t>& beats,
                               const std::vector<std::size_t>& downbeats,
                               const std::vector<std::size_t>& candidates) {
        std::size_t best_bpb = candidates.empty() ? config.dbn_beats_per_bar : candidates.front();
        std::size_t best_phase = 0;
        std::size_t best_hits = 0;
        if (beats.empty() || downbeats.empty()) {
            return std::make_pair(best_bpb, best_phase);
        }
        if (config.dbn_trace) {
            std::cerr << "DBN bpb inference: beats=" << beats.size()
                      << " downbeats=" << downbeats.size() << " candidates=";
            for (std::size_t bpb : candidates) {
                std::cerr << " " << bpb;
            }
            std::cerr << "\n";
        }
        std::unordered_map<std::size_t, std::size_t> beat_index;
        beat_index.reserve(beats.size());
        for (std::size_t i = 0; i < beats.size(); ++i) {
            beat_index[beats[i]] = i;
        }
        std::unordered_map<std::size_t, std::size_t> hits_by_bpb;
        std::unordered_map<std::size_t, std::size_t> phase_by_bpb;
        hits_by_bpb.reserve(candidates.size());
        phase_by_bpb.reserve(candidates.size());
        for (std::size_t bpb : candidates) {
            if (bpb == 0) {
                continue;
            }
            std::size_t phase = 0;
            auto it = beat_index.find(downbeats.front());
            if (it != beat_index.end()) {
                phase = it->second % bpb;
            }
            std::size_t hits = 0;
            for (std::size_t db : downbeats) {
                auto idx = beat_index.find(db);
                if (idx != beat_index.end() && (idx->second % bpb) == phase) {
                    ++hits;
                }
            }
            hits_by_bpb[bpb] = hits;
            phase_by_bpb[bpb] = phase;
            if (config.dbn_trace) {
                std::cerr << "DBN bpb inference: bpb=" << bpb
                          << " phase=" << phase
                          << " hits=" << hits << "\n";
            }
            if (hits > best_hits) {
                best_hits = hits;
                best_bpb = bpb;
                best_phase = phase;
            }
        }
        if (candidates.size() >= 2 && best_bpb == 3 &&
            hits_by_bpb.count(3) && hits_by_bpb.count(4)) {
            const double hits3 = static_cast<double>(hits_by_bpb[3]);
            const double hits4 = static_cast<double>(hits_by_bpb[4]);
            if (hits4 > 0.0 && hits3 < (hits4 * 1.5)) {
                best_bpb = 4;
                best_phase = phase_by_bpb[4];
                best_hits = hits_by_bpb[4];
                if (config.dbn_trace) {
                    std::cerr << "DBN bpb inference: biasing to 4/4 (hits3="
                              << hits3 << " hits4=" << hits4 << ")\n";
                }
            }
        }
        if (config.dbn_trace) {
            std::cerr << "DBN bpb inference: best_bpb=" << best_bpb
                      << " best_phase=" << best_phase
                      << " best_hits=" << best_hits << "\n";
        }
        return std::make_pair(best_bpb, best_phase);
    };

    auto project_downbeats_from_beats = [&](const std::vector<std::size_t>& beats,
                                            std::size_t bpb,
                                            std::size_t phase) {
        std::vector<std::size_t> downbeats;
        if (beats.empty() || bpb == 0) {
            return downbeats;
        }
        downbeats.reserve((beats.size() / bpb) + 1);
        for (std::size_t i = 0; i < beats.size(); ++i) {
            if ((i % bpb) == phase) {
                downbeats.push_back(beats[i]);
            }
        }
        return downbeats;
    };

    auto summarize_window = [&](const std::vector<float>& activation,
                                std::size_t start,
                                std::size_t end,
                                float floor_value) {
        struct Summary {
            std::size_t frames = 0;
            std::size_t above = 0;
            float min = 0.0f;
            float max = 0.0f;
            double mean = 0.0;
        };
        Summary summary;
        if (start >= end || end > activation.size()) {
            return summary;
        }
        summary.frames = end - start;
        summary.min = std::numeric_limits<float>::infinity();
        summary.max = -std::numeric_limits<float>::infinity();
        double total = 0.0;
        for (std::size_t i = start; i < end; ++i) {
            const float value = activation[i];
            summary.min = std::min(summary.min, value);
            summary.max = std::max(summary.max, value);
            total += value;
            if (value >= floor_value) {
                ++summary.above;
            }
        }
        summary.mean = (summary.frames > 0) ? (total / static_cast<double>(summary.frames)) : 0.0;
        if (!std::isfinite(summary.min)) {
            summary.min = 0.0f;
        }
        if (!std::isfinite(summary.max)) {
            summary.max = 0.0f;
        }
        return summary;
    };

    auto median_interval_bpm = [&](const std::vector<std::size_t>& frames) {
        if (frames.size() < 2 || fps <= 0.0) {
            return 0.0;
        }
        std::vector<double> intervals;
        intervals.reserve(frames.size() - 1);
        for (std::size_t i = 1; i < frames.size(); ++i) {
            const std::size_t delta = frames[i] - frames[i - 1];
            if (delta > 0) {
                intervals.push_back(static_cast<double>(delta));
            }
        }
        if (intervals.empty()) {
            return 0.0;
        }
        std::nth_element(intervals.begin(),
                         intervals.begin() + intervals.size() / 2,
                         intervals.end());
        const double median = intervals[intervals.size() / 2];
        if (median <= 1.0) {
            return 0.0;
        }
        return (60.0 * fps) / median;
    };

    if (!config.use_dbn && config.use_logit_consensus) {
        auto refine_peak_position = [&](std::size_t frame,
                                        const std::vector<float>& activation) -> double {
            if (activation.empty()) {
                return static_cast<double>(frame);
            }
            const std::size_t peak_frame = refine_frame_to_peak(frame, activation);
            double pos = static_cast<double>(peak_frame);
            if (peak_frame > 0 && peak_frame + 1 < activation.size()) {
                const double prev = activation[peak_frame - 1];
                const double curr = activation[peak_frame];
                const double next = activation[peak_frame + 1];
                const double denom = prev - 2.0 * curr + next;
                if (std::abs(denom) > 1e-9) {
                    double offset = 0.5 * (prev - next) / denom;
                    offset = std::max(-0.5, std::min(0.5, offset));
                    pos += offset;
                }
            }
            return pos;
        };

        auto compute_peaks_for_range = [&](const std::vector<float>& activation,
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

        std::vector<float> onset_activation;
        onset_activation.reserve(result.beat_activation.size());
        const bool phase_energy_ok =
            phase_energy && !phase_energy->empty() && phase_energy->size() >= used_frames;
        const std::vector<float>* onset_source =
            phase_energy_ok ? phase_energy : &result.beat_activation;
        if (!onset_source->empty()) {
            onset_activation.push_back(0.0f);
            for (std::size_t i = 1; i < onset_source->size(); ++i) {
                const float delta = (*onset_source)[i] - (*onset_source)[i - 1];
                const float onset = std::max(0.0f, delta);
                onset_activation.push_back(onset);
            }
        }

        float max_activation = 0.0f;
        for (float v : result.beat_activation) {
            if (v > max_activation) {
                max_activation = v;
            }
        }

        float peak_threshold = std::max(0.05f, config.activation_threshold);
        if (max_activation > 0.0f) {
            const float adaptive = std::max(0.1f, max_activation * 0.5f);
            peak_threshold = std::min(peak_threshold, adaptive);
        }

        std::vector<std::size_t> beat_peaks =
            compute_peaks_for_range(result.beat_activation, min_bpm, max_bpm, peak_threshold);
        if (beat_peaks.size() < config.logit_min_peaks) {
            float lowered = std::max(0.05f, peak_threshold * 0.5f);
            if (max_activation > 0.0f) {
                lowered = std::min(lowered, std::max(0.05f, max_activation * 0.25f));
            }
            beat_peaks =
                compute_peaks_for_range(result.beat_activation, min_bpm, max_bpm, lowered);
        }

        double interval_frames =
            median_interval_frames_interpolated(result.beat_activation, beat_peaks);
        if (config.verbose) {
            std::cerr << "Logit consensus: max_activation=" << max_activation
                      << " peak_threshold=" << peak_threshold
                      << " peaks=" << beat_peaks.size()
                      << " interval_frames=" << interval_frames
                      << " fps=" << fps << "\n";
        }

        double bpm = 0.0;
        double sweep_phase = 0.0;
        double sweep_score = -1.0;
        if (fps > 0.0 && min_bpm > 0.0f && max_bpm > min_bpm) {
            const double step = 0.05;
            for (double candidate = min_bpm; candidate <= max_bpm + 1e-6; candidate += step) {
                const double period = (60.0 * fps) / candidate;
                if (period <= 1.0) {
                    continue;
                }
                const double omega = (2.0 * kPi) / period;
                double sum_cos = 0.0;
                double sum_sin = 0.0;
                double sum_weight = 0.0;
                for (std::size_t i = 0; i < used_frames; ++i) {
                    const double weight = result.beat_activation[i];
                    if (weight <= 0.0) {
                        continue;
                    }
                    const double angle = omega * static_cast<double>(i);
                    sum_cos += weight * std::cos(angle);
                    sum_sin += weight * std::sin(angle);
                    sum_weight += weight;
                }
                if (sum_weight <= 0.0) {
                    continue;
                }
                const double magnitude =
                    std::hypot(sum_cos, sum_sin) / sum_weight;
                if (magnitude > sweep_score) {
                    sweep_score = magnitude;
                    bpm = candidate;
                    const double phase_angle = std::atan2(sum_sin, sum_cos);
                    sweep_phase = (phase_angle / omega);
                    sweep_phase = std::fmod(sweep_phase, period);
                    if (sweep_phase < 0.0) {
                        sweep_phase += period;
                    }
                }
            }
            if (config.verbose) {
                std::cerr << "Logit sweep: bpm=" << bpm
                          << " phase=" << sweep_phase
                          << " score=" << sweep_score << "\n";
            }
        }

        if (bpm <= 0.0) {
            if (interval_frames <= 0.0) {
                const std::vector<std::size_t> fallback_peaks =
                    compute_minimal_peaks(result.beat_activation);
                fill_beats_from_frames(fallback_peaks);
                const std::vector<std::size_t> aligned_downbeats =
                    align_downbeats_to_beats(fallback_peaks,
                                             compute_minimal_peaks(result.downbeat_activation));
                result.downbeat_feature_frames.clear();
                result.downbeat_feature_frames.reserve(aligned_downbeats.size());
                for (std::size_t frame : aligned_downbeats) {
                    result.downbeat_feature_frames.push_back(
                        static_cast<unsigned long long>(frame));
                }
                result.beat_projected_feature_frames.clear();
                result.beat_projected_sample_frames.clear();
                result.beat_projected_strengths.clear();
                result.downbeat_projected_feature_frames.clear();
                if (config.profile) {
                    std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                              << "ms peaks=" << peaks_ms << "ms\n";
                }
                return result;
            }
            bpm = (60.0 * fps) / interval_frames;
        }

        if (bpm > 0.0) {
            while (bpm < min_bpm && bpm * 2.0 <= max_bpm) {
                bpm *= 2.0;
            }
            while (bpm > max_bpm && bpm / 2.0 >= min_bpm) {
                bpm /= 2.0;
            }
            if (bpm < min_bpm) {
                bpm = min_bpm;
            } else if (bpm > max_bpm) {
                bpm = max_bpm;
            }
        }

        const double step_frames = (60.0 * fps) / std::max(1.0, bpm);
        if (config.verbose) {
            std::cerr << "Logit consensus: bpm=" << bpm
                      << " step_frames=" << step_frames
                      << " used_frames=" << used_frames
                      << " block_seconds=" << config.logit_block_seconds
                      << " hop_seconds=" << config.logit_block_hop_seconds
                      << " max_shift_s=" << config.logit_phase_max_shift_seconds
                      << "\n";
        }
        if (step_frames <= 0.0) {
            return result;
        }

        std::size_t strongest_peak = 0;
        float strongest_value = 0.0f;
        for (std::size_t i = 0; i < result.beat_activation.size(); ++i) {
            const float value = result.beat_activation[i];
            if (value > strongest_value) {
                strongest_value = value;
                strongest_peak = i;
            }
        }

        const std::vector<float>& phase_signal =
            onset_activation.empty() ? result.beat_activation : onset_activation;
        float max_phase_signal = 0.0f;
        for (float value : phase_signal) {
            if (value > max_phase_signal) {
                max_phase_signal = value;
            }
        }

        // Shift peaks backward to the attack onset (earliest rise), to avoid late-phase bias.
        std::vector<float> phase_onsets;
        phase_onsets.assign(phase_signal.size(), 0.0f);
        const float onset_ratio = 0.35f;
        const std::size_t onset_max_back = static_cast<std::size_t>(
            std::max(1.0, std::round(0.2 * fps)));
        const float onset_peak_threshold = std::max(0.02f, max_phase_signal * 0.25f);

        if (phase_signal.size() >= 3) {
            const std::size_t limit = std::min(used_frames, phase_signal.size());
            for (std::size_t i = 1; i + 1 < limit; ++i) {
                const float curr = phase_signal[i];
                if (curr < onset_peak_threshold) {
                    continue;
                }
                if (curr < phase_signal[i - 1] || curr < phase_signal[i + 1]) {
                    continue;
                }
                const float threshold = curr * onset_ratio;
                std::size_t frame = i;
                std::size_t steps = 0;
                while (frame > 0 && steps < onset_max_back &&
                       phase_signal[frame] > threshold) {
                    --frame;
                    ++steps;
                }
                if (frame < phase_onsets.size()) {
                    phase_onsets[frame] = std::max(phase_onsets[frame], curr);
                }
            }
        }

        const std::vector<float>& phase_score_signal =
            phase_onsets.empty() ? phase_signal : phase_onsets;

        auto score_phase_global = [&](double phase_frame) -> double {
            if (phase_signal.empty() || step_frames <= 0.0) {
                return -1.0;
            }

            double cursor = phase_frame;
            if (cursor < 0.0) {
                const double k = std::ceil((-cursor) / step_frames);
                cursor += k * step_frames;
            }

            double sum = 0.0;
            std::size_t count = 0;
            while (cursor < static_cast<double>(used_frames) &&
                   cursor < static_cast<double>(phase_score_signal.size())) {
                const std::size_t idx = static_cast<std::size_t>(std::llround(cursor));
                if (idx < phase_score_signal.size()) {
                    sum += phase_score_signal[idx];
                    ++count;
                }
                cursor += step_frames;
            }

            return count > 0 ? (sum / static_cast<double>(count)) : -1.0;
        };

        double global_phase = sweep_phase;
        double best_score = -1.0;
        if (step_frames > 0.0) {
            const double phase_step = 1.0;
            const double max_phase = std::max(1.0, step_frames);
            for (double phase = 0.0; phase < max_phase; phase += phase_step) {
                const double score = score_phase_global(phase);
                if (score > best_score) {
                    best_score = score;
                    global_phase = phase;
                }
            }
        }

        if (best_score < 0.0) {
            global_phase = refine_peak_position(strongest_peak, result.beat_activation);
        }

        if (config.verbose) {
            std::cerr << "Logit consensus: global_phase=" << global_phase
                      << " best_score=" << best_score << "\n";
        }

        const std::size_t block_frames = static_cast<std::size_t>(
            std::max(1.0, std::round(config.logit_block_seconds * fps)));
        const std::size_t hop_frames = static_cast<std::size_t>(
            std::max(1.0, std::round(config.logit_block_hop_seconds * fps)));

        (void)block_frames;
        (void)hop_frames;
        auto build_grid_frames = [&](double phase_seed) {
            std::vector<std::size_t> grid_frames;
            grid_frames.reserve(static_cast<std::size_t>(
                std::ceil(static_cast<double>(used_frames) / step_frames)) + 8);

            for (std::size_t block_start = 0;
                 block_start < used_frames;
                 block_start += hop_frames) {
                const std::size_t block_end =
                    std::min(block_start + block_frames, used_frames);

                const double block_phase = phase_seed;

                double cursor = block_phase;
                if (cursor < static_cast<double>(block_start)) {
                    const double k =
                        std::ceil((static_cast<double>(block_start) - cursor) / step_frames);
                    cursor += k * step_frames;
                }
                while (cursor < static_cast<double>(block_end)) {
                    grid_frames.push_back(static_cast<std::size_t>(std::llround(cursor)));
                    cursor += step_frames;
                }
            }

            std::vector<std::size_t> projected_frames;
            if (step_frames > 0.0 && grid_total_frames > 0) {
                const double total_frames = static_cast<double>(grid_total_frames);
                const double k = std::floor((0.0 - phase_seed) / step_frames);
                double cursor = phase_seed + k * step_frames;
                while (cursor < 0.0) {
                    cursor += step_frames;
                }
                while (cursor < total_frames) {
                    projected_frames.push_back(static_cast<std::size_t>(std::llround(cursor)));
                    cursor += step_frames;
                }
            }

            if (!projected_frames.empty()) {
                grid_frames.insert(grid_frames.end(),
                                   projected_frames.begin(),
                                   projected_frames.end());
            }

            return grid_frames;
        };

        std::vector<std::size_t> grid_frames = build_grid_frames(global_phase);

        if (!grid_frames.empty()) {
            std::sort(grid_frames.begin(), grid_frames.end());
            const std::size_t tolerance_frames = static_cast<std::size_t>(
                std::max(0.0, std::round(0.025 * fps)));
            dedupe_frames_tolerant(grid_frames, tolerance_frames);
        }
        if (config.verbose) {
            std::cerr << "Logit consensus: grid_frames=" << grid_frames.size();
            if (!grid_frames.empty()) {
                std::cerr << " first=" << grid_frames.front()
                          << " last=" << grid_frames.back();
            }
            std::cerr << "\n";
        }

        fill_beats_from_frames(grid_frames);
        if (config.verbose) {
            std::cerr << "Logit consensus: beats_out=" << result.beat_feature_frames.size()
                      << " samples_out=" << result.beat_sample_frames.size() << "\n";
        }

        const std::size_t bpb = std::max<std::size_t>(1, config.dbn_beats_per_bar);
        if (!result.downbeat_activation.empty()) {
            const float activation_floor = std::max(0.05f, config.activation_threshold * 0.2f);
            const float min_db_bpm = std::max(1.0f, min_bpm / static_cast<float>(bpb));
            const float max_db_bpm = std::max(min_db_bpm + 1.0f,
                                              max_bpm / static_cast<float>(bpb));
            const std::size_t min_interval =
                static_cast<std::size_t>(std::max(1.0,
                                                  std::floor((60.0 * fps) / max_db_bpm)));
            const std::size_t max_interval =
                static_cast<std::size_t>(std::ceil((60.0 * fps) / min_db_bpm));
            std::vector<std::size_t> downbeat_peaks =
                pick_peaks(result.downbeat_activation,
                           activation_floor,
                           min_interval,
                           max_interval);
            const std::vector<std::size_t> aligned_downbeats =
                align_downbeats_to_beats(grid_frames, downbeat_peaks);
            result.downbeat_feature_frames.clear();
            result.downbeat_feature_frames.reserve(aligned_downbeats.size());
            for (std::size_t frame : aligned_downbeats) {
                result.downbeat_feature_frames.push_back(
                    static_cast<unsigned long long>(frame));
            }
        } else if (!result.beat_feature_frames.empty()) {
            result.downbeat_feature_frames.push_back(result.beat_feature_frames.front());
        }

        result.beat_projected_feature_frames.clear();
        result.beat_projected_sample_frames.clear();
        result.beat_projected_strengths.clear();
        result.downbeat_projected_feature_frames.clear();

        if (config.profile) {
            std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                      << "ms peaks=" << peaks_ms << "ms\n";
        }
        return result;
    }

    if (config.use_dbn) {
        if (config.prefer_double_time && has_window) {
            min_bpm = std::min(min_bpm, min_bpm_alt);
            max_bpm = std::max(max_bpm, max_bpm_alt);
        }
        clamp_bpm_range(&min_bpm, &max_bpm);
        const float window_peak_threshold =
            std::max(config.activation_threshold, config.dbn_activation_floor);
        std::pair<std::size_t, std::size_t> window{0, used_frames};
        bool window_energy = false;
        const bool use_best_window = config.dbn_window_best;
        const bool phase_energy_ok =
            phase_energy && !phase_energy->empty() && phase_energy->size() >= used_frames;
        if (use_best_window) {
            const bool prefer_phase = config.dbn_window_use_phase_energy && phase_energy_ok;
            if (prefer_phase) {
                window = select_dbn_window_energy(*phase_energy,
                                                  config.dbn_window_seconds,
                                                  false);
                window_energy = true;
            } else {
                window = select_dbn_window_energy(result.beat_activation,
                                                  config.dbn_window_seconds,
                                                  false);
                window_energy = false;
            }
        } else if (config.dbn_window_use_phase_energy && phase_energy_ok) {
            window = select_dbn_window_energy(*phase_energy,
                                              config.dbn_window_seconds,
                                              config.dbn_window_intro_mid_outro);
            window_energy = true;
        } else {
            window = select_dbn_window(result.beat_activation,
                                       config.dbn_window_seconds,
                                       config.dbn_window_intro_mid_outro,
                                       min_bpm,
                                       max_bpm,
                                       window_peak_threshold);
        }
        const std::size_t window_start = window.first;
        const std::size_t window_end = window.second;
        const bool use_window = (window_start > 0 || window_end < used_frames);
        std::vector<float> beat_slice;
        std::vector<float> downbeat_slice;
        std::vector<std::pair<std::size_t, std::size_t>> window_candidates;
        std::vector<DBNDecodeResult> window_decodes;
        std::vector<double> window_bpms;
        double window_consensus_bpm = 0.0;
        bool consensus_phase_valid = false;
        double consensus_phase_frames = 0.0;
        if (use_window) {
            beat_slice.assign(result.beat_activation.begin() + window_start,
                              result.beat_activation.begin() + window_end);
            if (!result.downbeat_activation.empty()) {
                downbeat_slice.assign(result.downbeat_activation.begin() + window_start,
                                      result.downbeat_activation.begin() + window_end);
            }
            if (config.verbose) {
                std::cerr << "DBN window: start=" << window_start
                          << " end=" << window_end
                          << " frames=" << (window_end - window_start)
                          << " (" << ((window_end - window_start) / fps) << "s)"
                          << " selector=" << (use_best_window ? "best-energy"
                                       : (window_energy ? "energy" : "tempo"))
                          << " energy=" << (window_energy ? "phase" : "beat")
                          << "\n";
            }
        }
        double quality_qpar = 0.0;
        double quality_qmax = 0.0;
        double quality_qkur = 0.0;
        bool quality_valid = false;
        if (fps > 0.0) {
            const std::vector<float>& quality_src =
                use_window ? beat_slice : result.beat_activation;
            if (quality_src.size() >= 16) {
                const double min_bpm_q = std::max(1.0, static_cast<double>(config.min_bpm));
                const double max_bpm_q = std::max(min_bpm_q + 1.0,
                                                  static_cast<double>(config.max_bpm));
                const std::size_t min_lag =
                    static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm_q)));
                const std::size_t max_lag =
                    static_cast<std::size_t>(std::max<double>(min_lag + 1,
                                                              std::ceil((60.0 * fps) / min_bpm_q)));
                const std::size_t max_lag_clamped =
                    std::min<std::size_t>(max_lag, quality_src.size() - 1);
                if (max_lag_clamped > min_lag) {
                    std::vector<double> salience;
                    salience.reserve(max_lag_clamped - min_lag + 1);
                    for (std::size_t lag = min_lag; lag <= max_lag_clamped; ++lag) {
                        double sum = 0.0;
                        std::size_t count = 0;
                        for (std::size_t i = lag; i < quality_src.size(); ++i) {
                            sum += static_cast<double>(quality_src[i]) *
                                   static_cast<double>(quality_src[i - lag]);
                            ++count;
                        }
                        const double value = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
                        salience.push_back(value);
                    }
                    double mean = 0.0;
                    for (double v : salience) {
                        mean += v;
                    }
                    mean /= static_cast<double>(salience.size());
                    double var = 0.0;
                    for (double v : salience) {
                        const double d = v - mean;
                        var += d * d;
                    }
                    var /= static_cast<double>(salience.size());
                    const double rms = std::sqrt(var + mean * mean);
                    double max_val = 0.0;
                    for (double v : salience) {
                        if (v > max_val) {
                            max_val = v;
                        }
                    }
                    double kurtosis = 0.0;
                    if (var > 1e-12) {
                        double m4 = 0.0;
                        for (double v : salience) {
                            const double d = v - mean;
                            m4 += d * d * d * d;
                        }
                        m4 /= static_cast<double>(salience.size());
                        kurtosis = m4 / (var * var);
                    }
                    quality_qpar = (rms > 1e-12) ? (max_val / rms) : 0.0;
                    quality_qmax = max_val;
                    quality_qkur = kurtosis;
                    quality_valid = true;
                    if (config.dbn_trace) {
                        std::cerr << "DBN quality: qpar=" << quality_qpar
                                  << " qmax=" << quality_qmax
                                  << " qkur=" << quality_qkur
                                  << " lags=[" << min_lag << "," << max_lag_clamped << "]"
                                  << " frames=" << quality_src.size()
                                  << "\n";
                    }
                }
            }
        }
        const auto dbn_start = std::chrono::steady_clock::now();
        DBNDecodeResult decoded;
        if (config.dbn_mode == CoreMLConfig::DBNMode::Calmdad) {
            if (config.dbn_tempo_prior_weight > 0.0f) {
                const double tolerance =
                    std::max(0.0, static_cast<double>(config.dbn_interval_tolerance));
                const double min_interval_frames =
                    std::max(1.0, (60.0 * fps) / std::max(1.0f, max_bpm)) * (1.0 - tolerance);
                const double max_interval_frames =
                    std::max(1.0, (60.0 * fps) / std::max(1.0f, min_bpm)) * (1.0 + tolerance);
                const std::size_t peak_min_interval =
                    static_cast<std::size_t>(std::max(1.0, std::floor(min_interval_frames)));
                const std::size_t peak_max_interval =
                    static_cast<std::size_t>(std::max<double>(peak_min_interval,
                                                              std::ceil(max_interval_frames)));
                const float peak_threshold =
                    std::max(config.activation_threshold, config.dbn_activation_floor);

                const std::vector<float>& prior_src =
                    use_window ? beat_slice : result.beat_activation;
                std::vector<std::size_t> prior_peaks =
                    pick_peaks(prior_src, peak_threshold, peak_min_interval, peak_max_interval);
                const double prior_interval = median_interval_frames(prior_peaks);
                if (prior_interval > 1.0) {
                    const double prior_bpm = (60.0 * fps) / prior_interval;
                    const double window_pct = config.tempo_window_percent > 0.0f
                        ? (static_cast<double>(config.tempo_window_percent) / 100.0)
                        : 0.10;
                    min_bpm = static_cast<float>(prior_bpm * (1.0 - window_pct));
                    max_bpm = static_cast<float>(prior_bpm * (1.0 + window_pct));
                    clamp_bpm_range(&min_bpm, &max_bpm);
                    if (config.verbose) {
                        std::cerr << "DBN calmdad prior: bpm=" << prior_bpm
                                  << " peaks=" << prior_peaks.size()
                                  << " window_pct=" << window_pct
                                  << " clamp=[" << min_bpm << "," << max_bpm << "]\n";
                    }
                } else if (config.verbose) {
                    std::cerr << "DBN calmdad prior: insufficient peaks for clamp\n";
                }
            }
            if (use_window && config.dbn_window_intro_mid_outro && config.dbn_window_stitch) {
                const std::size_t window_frames = static_cast<std::size_t>(
                    std::round(config.dbn_window_seconds * fps));
                const std::size_t intro_end = std::min<std::size_t>(used_frames, window_frames);
                const auto intro = std::pair<std::size_t, std::size_t>{0, intro_end};
                const std::size_t mid_center = used_frames / 2;
                const std::size_t mid_start =
                    (mid_center > (intro.second - intro.first) / 2)
                        ? (mid_center - ((intro.second - intro.first) / 2))
                        : 0;
                const auto mid = std::pair<std::size_t, std::size_t>{
                    mid_start,
                    std::min<std::size_t>(used_frames, mid_start + (intro.second - intro.first))};
                const auto outro = std::pair<std::size_t, std::size_t>{
                    (used_frames > (intro.second - intro.first))
                        ? (used_frames - (intro.second - intro.first))
                        : 0,
                    used_frames};
                window_candidates = {intro, mid, outro};

                const float activation_floor = std::max(1e-6f, config.dbn_activation_floor);
                const std::size_t max_candidates =
                    std::max<std::size_t>(4, config.dbn_max_candidates);

                float peak_min_bpm = min_bpm;
                float peak_max_bpm = max_bpm;
                const double base_min_interval =
                    std::max(1.0, (60.0 * fps) / std::max(1.0f, peak_max_bpm)) *
                    (1.0 - std::max(0.0, static_cast<double>(config.dbn_interval_tolerance)));
                const double base_max_interval =
                    std::max(1.0, (60.0 * fps) / std::max(1.0f, peak_min_bpm)) *
                    (1.0 + std::max(0.0, static_cast<double>(config.dbn_interval_tolerance)));
                const double min_interval_frames = base_min_interval;
                const double max_interval_frames = base_max_interval;
                const std::size_t peak_min_interval =
                    static_cast<std::size_t>(std::max(1.0, std::floor(min_interval_frames)));
                const std::size_t peak_max_interval =
                    static_cast<std::size_t>(std::max<double>(peak_min_interval,
                                                              std::ceil(max_interval_frames)));
                const float peak_threshold = std::max(config.activation_threshold, activation_floor);

                std::vector<std::size_t> candidate_frames;
                candidate_frames.reserve(max_candidates);

                for (const auto& w : window_candidates) {
                    if (w.first >= w.second) {
                        continue;
                    }
                    std::vector<float> slice(result.beat_activation.begin() + w.first,
                                             result.beat_activation.begin() + w.second);
                    std::vector<std::size_t> peaks =
                        pick_peaks(slice, peak_threshold, peak_min_interval, peak_max_interval);
                    for (std::size_t idx : peaks) {
                        candidate_frames.push_back(w.first + idx);
                    }
                }

                if (candidate_frames.empty()) {
                    for (const auto& w : window_candidates) {
                        for (std::size_t i = w.first; i < w.second; ++i) {
                            if (result.beat_activation[i] >= peak_threshold) {
                                candidate_frames.push_back(i);
                            }
                        }
                    }
                }

                std::sort(candidate_frames.begin(), candidate_frames.end());
                candidate_frames.erase(std::unique(candidate_frames.begin(),
                                                   candidate_frames.end()),
                                       candidate_frames.end());

                if (config.verbose) {
                    const auto intro_sum =
                        summarize_window(result.beat_activation, intro.first, intro.second, activation_floor);
                    const auto mid_sum =
                        summarize_window(result.beat_activation, mid.first, mid.second, activation_floor);
                    const auto outro_sum =
                        summarize_window(result.beat_activation, outro.first, outro.second, activation_floor);
                    std::cerr << "DBN stitch windows:\n"
                              << "  intro frames=" << intro_sum.frames
                              << " above=" << intro_sum.above
                              << " min=" << intro_sum.min
                              << " mean=" << intro_sum.mean
                              << " max=" << intro_sum.max << "\n"
                              << "  mid   frames=" << mid_sum.frames
                              << " above=" << mid_sum.above
                              << " min=" << mid_sum.min
                              << " mean=" << mid_sum.mean
                              << " max=" << mid_sum.max << "\n"
                              << "  outro frames=" << outro_sum.frames
                              << " above=" << outro_sum.above
                              << " min=" << outro_sum.min
                              << " mean=" << outro_sum.mean
                              << " max=" << outro_sum.max << "\n";
                }

                if (candidate_frames.size() > max_candidates) {
                    std::vector<std::size_t> sorted = candidate_frames;
                    std::nth_element(sorted.begin(),
                                     sorted.begin() + static_cast<std::ptrdiff_t>(max_candidates),
                                     sorted.end(),
                                     [&](std::size_t a, std::size_t b) {
                                         const float beat_a = result.beat_activation[a];
                                         const float beat_b = result.beat_activation[b];
                                         const float down_a =
                                             (a < result.downbeat_activation.size())
                                                 ? result.downbeat_activation[a]
                                                 : 0.0f;
                                         const float down_b =
                                             (b < result.downbeat_activation.size())
                                                 ? result.downbeat_activation[b]
                                                 : 0.0f;
                                         return std::max(beat_a, down_a) > std::max(beat_b, down_b);
                                     });
                    sorted.resize(max_candidates);
                    std::sort(sorted.begin(), sorted.end());
                    candidate_frames.swap(sorted);
                }

                if (config.verbose) {
                    std::vector<std::size_t> candidate_intervals;
                    candidate_intervals.reserve(candidate_frames.size() > 1
                                                    ? candidate_frames.size() - 1
                                                    : 0);
                    for (std::size_t i = 1; i < candidate_frames.size(); ++i) {
                        const std::size_t delta = candidate_frames[i] - candidate_frames[i - 1];
                        if (delta > 0) {
                            candidate_intervals.push_back(delta);
                        }
                    }
                    const double bpm_from_candidates =
                        median_interval_bpm(candidate_frames);
                    double median_interval = 0.0;
                    if (!candidate_intervals.empty()) {
                        std::nth_element(candidate_intervals.begin(),
                                         candidate_intervals.begin() + candidate_intervals.size() / 2,
                                         candidate_intervals.end());
                        median_interval = static_cast<double>(candidate_intervals[candidate_intervals.size() / 2]);
                    }
                    std::cerr << "DBN stitch candidates: count=" << candidate_frames.size()
                              << " median_interval=" << median_interval
                              << " bpm_from_candidates=" << bpm_from_candidates << "\n";
                }

                double candidate_median_interval = 0.0;
                if (candidate_frames.size() >= 2) {
                    std::vector<std::size_t> candidate_intervals;
                    candidate_intervals.reserve(candidate_frames.size() - 1);
                    for (std::size_t i = 1; i < candidate_frames.size(); ++i) {
                        const std::size_t delta = candidate_frames[i] - candidate_frames[i - 1];
                        if (delta > 0) {
                            candidate_intervals.push_back(delta);
                        }
                    }
                    if (!candidate_intervals.empty()) {
                        std::nth_element(candidate_intervals.begin(),
                                         candidate_intervals.begin() + candidate_intervals.size() / 2,
                                         candidate_intervals.end());
                        candidate_median_interval =
                            static_cast<double>(candidate_intervals[candidate_intervals.size() / 2]);
                    }
                }
                double candidate_bpm = 0.0;
                if (candidate_median_interval > 1.0) {
                    candidate_bpm = (60.0 * fps) / candidate_median_interval;
                }
                if (candidate_bpm > 180.0 && (candidate_bpm * 0.5) >= min_bpm) {
                    const double tol = std::max(0.0, static_cast<double>(config.dbn_interval_tolerance));
                    const double target_interval = candidate_median_interval * 2.0;
                    const double min_gap = std::max(1.0, target_interval * (1.0 - tol));
                    std::vector<std::size_t> filtered;
                    filtered.reserve(candidate_frames.size());
                    std::size_t last = candidate_frames.front();
                    filtered.push_back(last);
                    for (std::size_t i = 1; i < candidate_frames.size(); ++i) {
                        const std::size_t frame = candidate_frames[i];
                        if ((frame - last) >= min_gap) {
                            filtered.push_back(frame);
                            last = frame;
                        }
                    }
                    if (filtered.size() >= 2) {
                        candidate_frames.swap(filtered);
                        candidate_median_interval = 0.0;
                        std::vector<std::size_t> candidate_intervals;
                        candidate_intervals.reserve(candidate_frames.size() - 1);
                        for (std::size_t i = 1; i < candidate_frames.size(); ++i) {
                            const std::size_t delta = candidate_frames[i] - candidate_frames[i - 1];
                            if (delta > 0) {
                                candidate_intervals.push_back(delta);
                            }
                        }
                        if (!candidate_intervals.empty()) {
                            std::nth_element(candidate_intervals.begin(),
                                             candidate_intervals.begin() + candidate_intervals.size() / 2,
                                             candidate_intervals.end());
                            candidate_median_interval =
                                static_cast<double>(candidate_intervals[candidate_intervals.size() / 2]);
                        }
                        if (candidate_median_interval > 1.0) {
                            candidate_bpm = (60.0 * fps) / candidate_median_interval;
                        }
                        if (config.verbose) {
                            std::cerr << "DBN stitch half-time filter: "
                                      << "min_gap=" << min_gap
                                      << " candidates=" << candidate_frames.size()
                                      << " median_interval=" << candidate_median_interval
                                      << " bpm=" << candidate_bpm << "\n";
                        }
                    }
                }
                if (candidate_bpm > 0.0) {
                    double adjusted_bpm = candidate_bpm;
                    if (candidate_bpm > 180.0) {
                        adjusted_bpm = candidate_bpm * 0.5;
                    }
                    const double window_pct = config.tempo_window_percent > 0.0f
                        ? (static_cast<double>(config.tempo_window_percent) / 100.0)
                        : 0.10;
                    min_bpm = static_cast<float>(adjusted_bpm * (1.0 - window_pct));
                    max_bpm = static_cast<float>(adjusted_bpm * (1.0 + window_pct));
                    clamp_bpm_range(&min_bpm, &max_bpm);
                    if (config.verbose) {
                        std::cerr << "DBN stitch tempo prior: bpm=" << candidate_bpm
                                  << " adjusted=" << adjusted_bpm
                                  << " clamp=[" << min_bpm << "," << max_bpm << "]\n";
                    }
                }

                const double epsilon = 1e-5;
                const double floor_value = epsilon / 2.0;
                std::vector<double> beat_log;
                std::vector<double> downbeat_log;
                beat_log.reserve(candidate_frames.size());
                downbeat_log.reserve(candidate_frames.size());
                for (std::size_t frame : candidate_frames) {
                    const double beat_value =
                        static_cast<double>(result.beat_activation[frame]) * (1.0 - epsilon)
                        + floor_value;
                    const double downbeat_value =
                        (frame < result.downbeat_activation.size())
                            ? (static_cast<double>(result.downbeat_activation[frame]) * (1.0 - epsilon)
                               + floor_value)
                            : floor_value;
                    const double combined_beat = std::max(floor_value, beat_value - downbeat_value);
                    beat_log.push_back(std::log(combined_beat));
                    downbeat_log.push_back(std::log(downbeat_value));
                }

                const double tolerance =
                    std::max(0.0, static_cast<double>(config.dbn_interval_tolerance));
                const double transition_lambda =
                    std::max(1e-6, static_cast<double>(config.dbn_transition_lambda));
                const double transition_reward = std::log(transition_lambda);
                const double tempo_change_penalty = transition_reward;
                const bool use_downbeat = config.dbn_use_downbeat;

                DBNPathResult best_path;
                std::size_t best_bpb = 0;
                for (std::size_t bpb : {3, 4}) {
                    DBNPathResult path = decode_dbn_beats_candidate_sparse(
                        candidate_frames,
                        beat_log,
                        downbeat_log,
                        fps,
                        min_bpm,
                        max_bpm,
                        config.dbn_bpm_step,
                        bpb,
                        tolerance,
                        use_downbeat,
                        transition_reward,
                        tempo_change_penalty);
                    if (path.best_score > best_path.best_score) {
                        best_path = std::move(path);
                        best_bpb = bpb;
                    }
                }
                decoded = std::move(best_path.decoded);
                if (config.verbose) {
                    std::cerr << "DBN stitch: windows=3"
                              << " candidates=" << candidate_frames.size()
                              << " best_bpb=" << best_bpb
                              << " beats=" << decoded.beat_frames.size()
                              << " downbeats=" << decoded.downbeat_frames.size()
                              << "\n";
                }
            } else if (use_window && config.dbn_window_intro_mid_outro && config.dbn_window_consensus) {
                const std::size_t window_frames = static_cast<std::size_t>(
                    std::round(config.dbn_window_seconds * fps));
                const auto best_span =
                    select_dbn_window(result.beat_activation,
                                      config.dbn_window_seconds,
                                      false,
                                      min_bpm,
                                      max_bpm,
                                      window_peak_threshold);
                std::vector<std::pair<std::size_t, std::size_t>> windows;
                windows.reserve(5);
                auto add_window = [&](std::size_t start) {
                    if (window_frames == 0 || start >= used_frames) {
                        return;
                    }
                    const std::size_t end = std::min<std::size_t>(used_frames, start + window_frames);
                    if (end <= start) {
                        return;
                    }
                    windows.emplace_back(start, end);
                };
                if (window_frames > 0 && used_frames > window_frames) {
                    const std::size_t max_start = used_frames - window_frames;
                    const std::array<double, 5> fractions = {0.2, 0.35, 0.5, 0.65, 0.8};
                    for (double frac : fractions) {
                        const std::size_t start = static_cast<std::size_t>(
                            std::llround(static_cast<double>(max_start) * frac));
                        add_window(start);
                    }
                } else {
                    add_window(0);
                }
                window_candidates = std::move(windows);
                if (window_candidates.empty()) {
                    window_candidates = {best_span};
                }
                if (window_candidates.size() > 1) {
                    std::sort(window_candidates.begin(), window_candidates.end());
                    window_candidates.erase(
                        std::unique(window_candidates.begin(), window_candidates.end()),
                        window_candidates.end());
                }
                if (config.dbn_trace) {
                    std::cerr << "DBN consensus windows:";
                    for (const auto& w : window_candidates) {
                        const double start_s = fps > 0.0 ? (w.first / fps) : 0.0;
                        const double end_s = fps > 0.0 ? (w.second / fps) : 0.0;
                        std::cerr << " [" << w.first << "-" << w.second
                                  << " (" << start_s << "s-" << end_s << "s)]";
                    }
                    std::cerr << "\n";
                }

                const auto bpm_from_beats = [&](const std::vector<std::size_t>& beats) {
                    const double interval = median_interval_frames(beats);
                    if (interval <= 1.0) {
                        return 0.0;
                    }
                    return (60.0 * fps) / interval;
                };

                const auto downbeat_peak_score = [&](const std::vector<float>& downbeat_activation) {
                    if (downbeat_activation.empty()) {
                        return 0.0;
                    }
                    std::vector<std::size_t> peaks =
                        pick_peaks(downbeat_activation,
                                   window_peak_threshold,
                                   1,
                                   downbeat_activation.size());
                    if (peaks.empty()) {
                        return 0.0;
                    }
                    double sum = 0.0;
                    std::size_t first_peak = peaks.front();
                    for (std::size_t idx : peaks) {
                        if (idx < downbeat_activation.size()) {
                            sum += downbeat_activation[idx];
                        }
                        if (idx < first_peak) {
                            first_peak = idx;
                        }
                    }
                    const double avg = sum / static_cast<double>(peaks.size());
                    const double first_peak_seconds = fps > 0.0
                        ? static_cast<double>(first_peak) / fps
                        : 0.0;
                    const double early_boost = 1.0 / (1.0 + first_peak_seconds);
                    return avg * early_boost;
                };

                window_decodes.clear();
                window_bpms.clear();
                std::vector<double> window_down_scores;
                window_down_scores.reserve(window_candidates.size());
                std::vector<std::size_t> window_first_down;
                window_first_down.reserve(window_candidates.size());
                std::vector<std::size_t> window_down_counts;
                window_down_counts.reserve(window_candidates.size());
                std::vector<std::size_t> window_starts;
                window_starts.reserve(window_candidates.size());
                for (const auto& w : window_candidates) {
                    std::vector<float> w_beat(result.beat_activation.begin() + w.first,
                                              result.beat_activation.begin() + w.second);
                    std::vector<float> w_downbeat;
                    if (!result.downbeat_activation.empty()) {
                        w_downbeat.assign(result.downbeat_activation.begin() + w.first,
                                          result.downbeat_activation.begin() + w.second);
                    }
                    DBNDecodeResult tmp = decode_dbn_beats_calmdad(w_beat,
                                                                 w_downbeat,
                                                                 fps,
                                                                 min_bpm,
                                                                 max_bpm,
                                                                 config.dbn_bpm_step,
                                                                 config);
                    window_decodes.push_back(tmp);
                    window_bpms.push_back(bpm_from_beats(tmp.beat_frames));
                    window_down_scores.push_back(downbeat_peak_score(w_downbeat));
                    window_down_counts.push_back(tmp.downbeat_frames.size());
                    window_starts.push_back(w.first);
                    if (!tmp.downbeat_frames.empty()) {
                        window_first_down.push_back(tmp.downbeat_frames.front());
                    } else {
                        window_first_down.push_back(std::numeric_limits<std::size_t>::max());
                    }
                }
                std::vector<double> valid_bpms;
                for (double bpm : window_bpms) {
                    if (bpm > 0.0) {
                        valid_bpms.push_back(bpm);
                    }
                }
                if (!valid_bpms.empty()) {
                    std::nth_element(valid_bpms.begin(),
                                     valid_bpms.begin() + valid_bpms.size() / 2,
                                     valid_bpms.end());
                    window_consensus_bpm = valid_bpms[valid_bpms.size() / 2];
                }
                double best_down_score = -1.0;
                for (double score : window_down_scores) {
                    if (score > best_down_score) {
                        best_down_score = score;
                    }
                }
                const double downbeat_score_floor =
                    (best_down_score > 0.0) ? (best_down_score * 0.6) : 0.0;
                std::vector<std::size_t> kept_indices;
                kept_indices.reserve(window_candidates.size());
                for (std::size_t i = 0; i < window_candidates.size(); ++i) {
                    if (window_down_counts[i] < 4) {
                        continue;
                    }
                    if (window_down_scores[i] < downbeat_score_floor) {
                        continue;
                    }
                    if (window_bpms[i] <= 0.0) {
                        continue;
                    }
                    kept_indices.push_back(i);
                }
                if (kept_indices.empty()) {
                    for (std::size_t i = 0; i < window_candidates.size(); ++i) {
                        if (window_bpms[i] > 0.0) {
                            kept_indices.push_back(i);
                        }
                    }
                }
                std::size_t best_idx = 0;
                std::size_t best_first = std::numeric_limits<std::size_t>::max();
                double best_score = -1.0;
                for (std::size_t i : kept_indices) {
                    const std::size_t first = window_first_down[i];
                    if (first == std::numeric_limits<std::size_t>::max()) {
                        continue;
                    }
                    const double score = window_down_scores[i];
                    if (first < best_first || (first == best_first && score > best_score)) {
                        best_first = first;
                        best_score = score;
                        best_idx = i;
                    }
                }
                decoded = window_decodes[best_idx];
                if (config.verbose) {
                    std::cerr << "DBN consensus bpm=" << window_consensus_bpm
                              << " window_idx=" << best_idx
                              << " downbeat_score=" << best_down_score << "\n";
                }
                const auto& best_window = window_candidates[best_idx];
                for (std::size_t& frame : decoded.beat_frames) {
                    frame += best_window.first;
                }
                for (std::size_t& frame : decoded.downbeat_frames) {
                    frame += best_window.first;
                }
                if (!kept_indices.empty()) {
                    std::vector<std::pair<double, double>> phase_offsets;
                    phase_offsets.reserve(kept_indices.size());
                    for (std::size_t i : kept_indices) {
                        const double bpm = window_bpms[i];
                        if (bpm <= 0.0 || fps <= 0.0) {
                            continue;
                        }
                        const std::size_t first = window_first_down[i];
                        if (first == std::numeric_limits<std::size_t>::max()) {
                            continue;
                        }
                        const std::size_t global_first = window_starts[i] + first;
                        const double period_frames = (60.0 * fps) / bpm;
                        if (period_frames <= 1.0) {
                            continue;
                        }
                        const double offset =
                            std::fmod(static_cast<double>(global_first), period_frames);
                        const double weight = std::max(0.0, window_down_scores[i]);
                        if (weight > 0.0) {
                            phase_offsets.emplace_back(offset, weight);
                        }
                    }
                    if (!phase_offsets.empty()) {
                        std::sort(phase_offsets.begin(),
                                  phase_offsets.end(),
                                  [](const auto& a, const auto& b) { return a.first < b.first; });
                        double total_weight = 0.0;
                        for (const auto& item : phase_offsets) {
                            total_weight += item.second;
                        }
                        double acc = 0.0;
                        double median_offset = phase_offsets.back().first;
                        const double half = total_weight * 0.5;
                        for (const auto& item : phase_offsets) {
                            acc += item.second;
                            if (acc >= half) {
                                median_offset = item.first;
                                break;
                            }
                        }
                        consensus_phase_frames = median_offset;
                        consensus_phase_valid = true;
                    }
                }
                if (config.dbn_trace) {
                    std::cerr << "DBN consensus kept_windows=" << kept_indices.size()
                              << " downbeat_score_floor=" << downbeat_score_floor
                              << " phase_consensus="
                              << (consensus_phase_valid ? consensus_phase_frames : -1.0)
                              << "\n";
                }
            } else {
                decoded = decode_dbn_beats_calmdad(use_window ? beat_slice : result.beat_activation,
                                                   use_window ? downbeat_slice : result.downbeat_activation,
                                                   fps,
                                                   min_bpm,
                                                   max_bpm,
                                                   config.dbn_bpm_step,
                                                   config);
            }
        } else {
            decoded = decode_dbn_beats_beatit(use_window ? beat_slice : result.beat_activation,
                                              use_window ? downbeat_slice : result.downbeat_activation,
                                              fps,
                                              min_bpm,
                                              max_bpm,
                                              config,
                                              reference_bpm);
        }
        const auto dbn_end = std::chrono::steady_clock::now();
        dbn_ms += std::chrono::duration<double, std::milli>(dbn_end - dbn_start).count();
        if (!decoded.beat_frames.empty()) {
            double projected_bpm = 0.0;
            if (use_window) {
                for (std::size_t& frame : decoded.beat_frames) {
                    frame += window_start;
                }
                for (std::size_t& frame : decoded.downbeat_frames) {
                    frame += window_start;
                }
            }
            if (config.dbn_project_grid) {
                std::vector<std::size_t> refined_beats;
                refined_beats.reserve(decoded.beat_frames.size());
                for (std::size_t frame : decoded.beat_frames) {
                    refined_beats.push_back(refine_frame_to_peak(frame, result.beat_activation));
                }
                decoded.beat_frames = std::move(refined_beats);
                dedupe_frames(decoded.beat_frames);

                if (!decoded.downbeat_frames.empty()) {
                    const std::vector<float>& downbeat_source =
                        result.downbeat_activation.empty() ? result.beat_activation
                                                           : result.downbeat_activation;
                    std::vector<std::size_t> refined_downbeats;
                    refined_downbeats.reserve(decoded.downbeat_frames.size());
                    for (std::size_t frame : decoded.downbeat_frames) {
                        refined_downbeats.push_back(refine_frame_to_peak(frame, downbeat_source));
                    }
                    decoded.downbeat_frames = std::move(refined_downbeats);
                    dedupe_frames(decoded.downbeat_frames);
                }

                const double min_interval_frames =
                    (max_bpm > 1.0f && fps > 0.0) ? (60.0 * fps) / max_bpm : 0.0;
                const double short_interval_threshold =
                    (min_interval_frames > 0.0) ? std::max(1.0, min_interval_frames * 0.5) : 0.0;
                const std::vector<std::size_t> filtered_beats =
                    filter_short_intervals(decoded.beat_frames, short_interval_threshold);
                const std::vector<std::size_t> aligned_downbeats =
                    align_downbeats_to_beats(filtered_beats, decoded.downbeat_frames);

                std::vector<std::size_t> bpb_candidates;
                if (config.dbn_mode == CoreMLConfig::DBNMode::Calmdad) {
                    bpb_candidates = {3, 4};
                } else {
                    bpb_candidates = {config.dbn_beats_per_bar};
                }
                const auto [bpb, phase] =
                    infer_bpb_phase(filtered_beats, aligned_downbeats, bpb_candidates);
                const double base_interval = median_interval_frames(filtered_beats);
                const std::vector<float>& tempo_activation =
                    use_window ? beat_slice : result.beat_activation;
                const float tempo_threshold =
                    std::max(config.dbn_activation_floor, config.activation_threshold * 0.5f);
                const std::size_t tempo_min_interval =
                    static_cast<std::size_t>(std::max(1.0,
                                                      std::floor((60.0 * fps) /
                                                                 std::max(1.0f, max_bpm))));
                const std::size_t tempo_max_interval =
                    static_cast<std::size_t>(std::max<double>(tempo_min_interval,
                                                              std::ceil((60.0 * fps) /
                                                                        std::max(1.0f, min_bpm))));
                const std::vector<std::size_t> tempo_peaks =
                    pick_peaks(tempo_activation,
                               tempo_threshold,
                               tempo_min_interval,
                               tempo_max_interval);
                const std::vector<std::size_t> tempo_peaks_full =
                    use_window
                        ? pick_peaks(result.beat_activation,
                                     tempo_threshold,
                                     tempo_min_interval,
                                     tempo_max_interval)
                        : tempo_peaks;
                double bpm_from_peaks = 0.0;
                double bpm_from_peaks_median = 0.0;
                double bpm_from_peaks_reg = 0.0;
                double bpm_from_peaks_median_full = 0.0;
                double bpm_from_peaks_reg_full = 0.0;
                if (tempo_peaks.size() >= 2) {
                    const double interval_median =
                        median_interval_frames_interpolated(tempo_activation, tempo_peaks);
                    const double interval_reg =
                        regression_interval_frames_interpolated(tempo_activation, tempo_peaks);
                    if (interval_median > 0.0) {
                        bpm_from_peaks_median = (60.0 * fps) / interval_median;
                        bpm_from_peaks = bpm_from_peaks_median;
                    }
                    if (interval_reg > 0.0) {
                        bpm_from_peaks_reg = (60.0 * fps) / interval_reg;
                        if (bpm_from_peaks_median > 0.0) {
                            const double ratio =
                                std::abs(bpm_from_peaks_reg - bpm_from_peaks_median) /
                                bpm_from_peaks_median;
                            if (ratio <= 0.02) {
                                bpm_from_peaks = bpm_from_peaks_reg;
                            }
                        } else {
                            bpm_from_peaks = bpm_from_peaks_reg;
                        }
                    }
                }
                if (tempo_peaks_full.size() >= 2) {
                    const double interval_median =
                        median_interval_frames_interpolated(result.beat_activation, tempo_peaks_full);
                    const double interval_reg =
                        regression_interval_frames_interpolated(result.beat_activation, tempo_peaks_full);
                    if (interval_median > 0.0) {
                        bpm_from_peaks_median_full = (60.0 * fps) / interval_median;
                    }
                    if (interval_reg > 0.0) {
                        bpm_from_peaks_reg_full = (60.0 * fps) / interval_reg;
                    }
                }
                double bpm_from_downbeats = 0.0;
                double bpm_from_downbeats_median = 0.0;
                double bpm_from_downbeats_reg = 0.0;
                std::vector<std::size_t> downbeat_peaks;
                IntervalStats downbeat_stats;
                if (!result.downbeat_activation.empty() && bpb > 0) {
                    const std::vector<float>& downbeat_activation =
                        use_window ? downbeat_slice : result.downbeat_activation;
                    const float downbeat_min_bpm =
                        std::max(1.0f, min_bpm / static_cast<float>(bpb));
                    const float downbeat_max_bpm =
                        std::max(downbeat_min_bpm + 1.0f, max_bpm / static_cast<float>(bpb));
                    const std::size_t downbeat_min_interval =
                        static_cast<std::size_t>(std::max(1.0,
                                                          std::floor((60.0 * fps) /
                                                                     downbeat_max_bpm)));
                    const std::size_t downbeat_max_interval =
                        static_cast<std::size_t>(std::max<double>(downbeat_min_interval,
                                                                  std::ceil((60.0 * fps) /
                                                                            downbeat_min_bpm)));
                    downbeat_peaks = pick_peaks(downbeat_activation,
                                                tempo_threshold,
                                                downbeat_min_interval,
                                                downbeat_max_interval);
                    if (downbeat_peaks.size() >= 2) {
                        const double interval_median =
                            median_interval_frames_interpolated(downbeat_activation, downbeat_peaks);
                        const double interval_reg =
                            regression_interval_frames_interpolated(downbeat_activation, downbeat_peaks);
                        if (interval_median > 0.0) {
                            const double downbeat_bpm = (60.0 * fps) / interval_median;
                            bpm_from_downbeats_median = downbeat_bpm * static_cast<double>(bpb);
                            bpm_from_downbeats = bpm_from_downbeats_median;
                        }
                        if (interval_reg > 0.0) {
                            const double downbeat_bpm = (60.0 * fps) / interval_reg;
                            bpm_from_downbeats_reg = downbeat_bpm * static_cast<double>(bpb);
                            if (bpm_from_downbeats_median > 0.0) {
                                const double ratio =
                                    std::abs(bpm_from_downbeats_reg - bpm_from_downbeats_median) /
                                    bpm_from_downbeats_median;
                                if (ratio <= 0.02) {
                                    bpm_from_downbeats = bpm_from_downbeats_reg;
                                }
                            } else {
                                bpm_from_downbeats = bpm_from_downbeats_reg;
                            }
                        }
                    }
                }
                if (!downbeat_peaks.empty()) {
                    downbeat_stats = interval_stats_interpolated(
                        use_window ? downbeat_slice : result.downbeat_activation,
                        downbeat_peaks,
                        fps,
                        0.2);
                }
                if (config.dbn_trace) {
                    const IntervalStats tempo_stats =
                        interval_stats_interpolated(tempo_activation, tempo_peaks, fps, 0.2);
                    const IntervalStats decoded_stats =
                        interval_stats_frames(decoded.beat_frames, fps, 0.2);
                    const IntervalStats decoded_filtered_stats =
                        interval_stats_frames(filtered_beats, fps, 0.2);
                    auto print_stats = [&](const char* label, const IntervalStats& stats) {
                        if (stats.count == 0 || stats.median_interval <= 0.0) {
                            std::cerr << "DBN stats: " << label << " empty\n";
                            return;
                        }
                        const double bpm_median = (60.0 * fps) / stats.median_interval;
                        const double bpm_mean = (60.0 * fps) / stats.mean_interval;
                        const double interval_cv = stats.mean_interval > 0.0
                            ? (stats.stdev_interval / stats.mean_interval)
                            : 0.0;
                        std::cerr << "DBN stats: " << label
                                  << " count=" << stats.count
                                  << " bpm_median=" << bpm_median
                                  << " bpm_mean=" << bpm_mean
                                  << " interval_cv=" << interval_cv
                                  << " interval_range=[" << stats.min_interval
                                  << "," << stats.max_interval << "]";
                        if (!stats.top_bpm_bins.empty()) {
                            std::cerr << " bpm_bins:";
                            for (const auto& bin : stats.top_bpm_bins) {
                                std::cerr << " " << bin.first << "(" << bin.second << ")";
                            }
                        }
                        std::cerr << "\n";
                    };
                    print_stats("tempo_peaks", tempo_stats);
                    if (!downbeat_peaks.empty()) {
                        print_stats("downbeat_peaks", downbeat_stats);
                    }
                    print_stats("decoded_beats", decoded_stats);
                    print_stats("decoded_beats_filtered", decoded_filtered_stats);
                    if (short_interval_threshold > 0.0) {
                        std::cerr << "DBN stats: filter_threshold=" << short_interval_threshold
                                  << " min_interval=" << min_interval_frames << "\n";
                    }
                }
                const auto bpm_from_linear_fit = [&](const std::vector<std::size_t>& beats) -> double {
                    if (beats.size() < 4 || fps <= 0.0) {
                        return 0.0;
                    }
                    double sum_x = 0.0;
                    double sum_y = 0.0;
                    double sum_xx = 0.0;
                    double sum_xy = 0.0;
                    const double n = static_cast<double>(beats.size());
                    for (std::size_t i = 0; i < beats.size(); ++i) {
                        const double x = static_cast<double>(i);
                        const double y = static_cast<double>(beats[i]);
                        sum_x += x;
                        sum_y += y;
                        sum_xx += x * x;
                        sum_xy += x * y;
                    }
                    const double denom = (n * sum_xx - sum_x * sum_x);
                    if (std::abs(denom) < 1e-9) {
                        return 0.0;
                    }
                    const double slope = (n * sum_xy - sum_x * sum_y) / denom;
                    if (slope <= 0.0) {
                        return 0.0;
                    }
                    return (60.0 * fps) / slope;
                };
                const double bpm_from_fit = bpm_from_linear_fit(filtered_beats);
                double bpm_from_global_fit = 0.0;
                if (config.dbn_window_intro_mid_outro && used_frames > 0) {
                    const std::size_t window_frames = static_cast<std::size_t>(
                        std::round(config.dbn_window_seconds * fps));
                    const std::size_t edge_pad_frames = static_cast<std::size_t>(
                        std::round(30.0 * fps));
                    const std::size_t intro_start = std::min<std::size_t>(used_frames, edge_pad_frames);
                    const std::size_t intro_end = std::min<std::size_t>(
                        used_frames, intro_start + window_frames);
                    const auto intro = std::pair<std::size_t, std::size_t>{intro_start, intro_end};
                    const std::size_t mid_center = used_frames / 2;
                    const std::size_t mid_start =
                        (mid_center > (intro.second - intro.first) / 2)
                            ? (mid_center - ((intro.second - intro.first) / 2))
                            : 0;
                    const auto mid = std::pair<std::size_t, std::size_t>{
                        mid_start,
                        std::min<std::size_t>(used_frames, mid_start + (intro.second - intro.first))};
                    const std::size_t outro_end = used_frames > edge_pad_frames
                        ? (used_frames - edge_pad_frames)
                        : used_frames;
                    const std::size_t outro_start =
                        (outro_end > window_frames) ? (outro_end - window_frames) : 0;
                    const auto outro = std::pair<std::size_t, std::size_t>{outro_start, outro_end};
                    const std::vector<std::pair<std::size_t, std::size_t>> tempo_windows = {
                        intro, mid, outro
                    };
                    if (config.dbn_trace) {
                        auto print_window = [&](const char* name,
                                                const std::pair<std::size_t, std::size_t>& w) {
                            const double start_s = fps > 0.0 ? static_cast<double>(w.first) / fps : 0.0;
                            const double end_s = fps > 0.0 ? static_cast<double>(w.second) / fps : 0.0;
                            std::cerr << "DBN tempo window " << name
                                      << ": frames=" << w.first << "-" << w.second
                                      << " (" << start_s << "s-" << end_s << "s)\n";
                        };
                        print_window("intro", intro);
                        print_window("mid", mid);
                        print_window("outro", outro);
                    }
                    std::vector<std::size_t> global_beats;
                    for (const auto& w : tempo_windows) {
                        if (w.second <= w.first) {
                            continue;
                        }
                        std::vector<float> w_beat(result.beat_activation.begin() + w.first,
                                                  result.beat_activation.begin() + w.second);
                        std::vector<float> w_downbeat;
                        if (!result.downbeat_activation.empty()) {
                            w_downbeat.assign(result.downbeat_activation.begin() + w.first,
                                              result.downbeat_activation.begin() + w.second);
                        }
                        DBNDecodeResult tmp = decode_dbn_beats_calmdad(w_beat,
                                                                      w_downbeat,
                                                                      fps,
                                                                      min_bpm,
                                                                      max_bpm,
                                                                      config.dbn_bpm_step,
                                                                      config);
                        for (std::size_t frame : tmp.beat_frames) {
                            global_beats.push_back(frame + w.first);
                        }
                    }
                    if (!global_beats.empty()) {
                        std::sort(global_beats.begin(), global_beats.end());
                        dedupe_frames(global_beats);
                        bpm_from_global_fit = bpm_from_linear_fit(global_beats);
                    }
                }
                const bool quality_low =
                    quality_valid && (quality_qkur < 3.6);
                const bool drop_global = quality_low && bpm_from_global_fit > 0.0;
                const bool drop_fit = quality_low && bpm_from_fit > 0.0;
                const std::size_t downbeat_count = downbeat_stats.count;
                const double downbeat_cv = (downbeat_count > 0 && downbeat_stats.mean_interval > 0.0)
                    ? (downbeat_stats.stdev_interval / downbeat_stats.mean_interval)
                    : 0.0;
                const bool downbeat_override_ok =
                    !quality_low && downbeat_count >= 6 && downbeat_cv <= 0.25;
                const double ref_downbeat_ratio =
                    (downbeat_override_ok && bpm_from_downbeats > 0.0)
                        ? (std::abs(reference_bpm - bpm_from_downbeats) / bpm_from_downbeats)
                        : 0.0;
                const bool ref_mismatch =
                    downbeat_override_ok && bpm_from_downbeats > 0.0 && ref_downbeat_ratio > 0.005;
                const bool drop_ref = (quality_low || ref_mismatch) && reference_bpm > 0.0f;
                double bpm_for_grid = 0.0;
                std::string bpm_source = "none";
                if (reference_bpm > 0.0f && !quality_low && !ref_mismatch &&
                    config.dbn_window_consensus) {
                    bpm_for_grid = reference_bpm;
                    bpm_source = "reference";
                } else if (!quality_low && bpm_from_global_fit > 0.0) {
                    bpm_for_grid = bpm_from_global_fit;
                    bpm_source = "global_fit";
                } else if (downbeat_override_ok && bpm_from_downbeats > 0.0) {
                    bpm_for_grid = bpm_from_downbeats;
                    bpm_source = "downbeats_primary";
                } else if (!quality_low && bpm_from_peaks_reg_full > 0.0) {
                    bpm_for_grid = bpm_from_peaks_reg_full;
                    bpm_source = "peaks_reg_full";
                } else if (!quality_low && bpm_from_fit > 0.0) {
                    bpm_for_grid = bpm_from_fit;
                    bpm_source = "fit";
                } else if (bpm_from_peaks_median > 0.0) {
                    bpm_for_grid = bpm_from_peaks_median;
                    bpm_source = "peaks_median";
                } else if (bpm_from_peaks > 0.0) {
                    bpm_for_grid = bpm_from_peaks;
                    bpm_source = "peaks";
                }
                if (bpm_for_grid <= 0.0 && reference_bpm > 0.0f &&
                    config.dbn_window_consensus) {
                    bpm_for_grid = reference_bpm;
                    bpm_source = "reference_fallback";
                }
                const double bpm_before_downbeat = bpm_for_grid;
                const std::string bpm_source_before_downbeat = bpm_source;
                if (downbeat_override_ok && bpm_from_downbeats > 0.0 && bpm_for_grid > 0.0 &&
                    bpm_source != "peaks_reg_full" &&
                    bpm_source != "downbeats_primary") {
                    const double ratio =
                        std::abs(bpm_from_downbeats - bpm_for_grid) / bpm_for_grid;
                    if (ratio <= 0.005) {
                        bpm_for_grid = bpm_from_downbeats;
                        bpm_source = "downbeats_override";
                    }
                }
                if (bpm_for_grid <= 0.0 && decoded.bpm > 0.0) {
                    bpm_for_grid = decoded.bpm;
                    bpm_source = "decoded";
                }
                if (bpm_for_grid <= 0.0 && base_interval > 0.0) {
                    bpm_for_grid = (60.0 * fps) / base_interval;
                    bpm_source = "base_interval";
                }
                projected_bpm = bpm_for_grid;
                double step_frames =
                    (bpm_for_grid > 0.0) ? (60.0 * fps) / bpm_for_grid : base_interval;
                if (config.verbose) {
                    std::cerr << "DBN grid: bpm=" << decoded.bpm
                              << " bpm_from_fit=" << bpm_from_fit
                              << " bpm_from_global_fit=" << bpm_from_global_fit
                              << " bpm_from_peaks=" << bpm_from_peaks
                              << " bpm_from_peaks_median=" << bpm_from_peaks_median
                              << " bpm_from_peaks_reg=" << bpm_from_peaks_reg
                              << " bpm_from_peaks_median_full=" << bpm_from_peaks_median_full
                              << " bpm_from_peaks_reg_full=" << bpm_from_peaks_reg_full
                              << " bpm_from_downbeats=" << bpm_from_downbeats
                              << " bpm_from_downbeats_median=" << bpm_from_downbeats_median
                              << " bpm_from_downbeats_reg=" << bpm_from_downbeats_reg
                              << " base_interval=" << base_interval
                              << " bpm_window_consensus=" << window_consensus_bpm
                              << " bpm_reference=" << reference_bpm
                              << " quality_qpar=" << quality_qpar
                              << " quality_qkur=" << quality_qkur
                              << " quality_low=" << (quality_low ? 1 : 0)
                              << " bpm_for_grid=" << bpm_for_grid
                              << " step_frames=" << step_frames
                              << " start_frame=" << decoded.beat_frames.front()
                              << "\n";
                }
                if (config.dbn_trace) {
                    std::cerr << "DBN quality gate: low=" << (quality_low ? 1 : 0)
                              << " drop_ref=" << (drop_ref ? 1 : 0)
                              << " drop_global=" << (drop_global ? 1 : 0)
                              << " drop_fit=" << (drop_fit ? 1 : 0)
                              << " downbeat_ok=" << (downbeat_override_ok ? 1 : 0)
                              << " downbeat_cv=" << downbeat_cv
                              << " downbeat_count=" << downbeat_count
                              << " used=" << bpm_source
                              << " pre_override=" << bpm_source_before_downbeat
                              << " pre_bpm=" << bpm_before_downbeat
                              << "\n";
                }
                std::size_t earliest_peak = decoded.beat_frames.front();
                std::size_t earliest_downbeat_peak = decoded.beat_frames.front();
                float earliest_downbeat_value = 0.0f;
                std::size_t strongest_peak = decoded.beat_frames.front();
                float strongest_peak_value = -1.0f;
                const float activation_floor =
                    std::max(0.01f, config.activation_threshold * 0.1f);
                if (base_interval > 1.0 && !decoded.beat_frames.empty()) {
                    const std::size_t peak_search_start = use_window ? window_start : 0;
                    const std::size_t peak_search_end = use_window
                        ? std::min<std::size_t>(
                              used_frames - 1,
                              window_start + static_cast<std::size_t>(
                                  std::llround(base_interval)))
                        : std::min<std::size_t>(
                              used_frames - 1,
                              static_cast<std::size_t>(std::llround(base_interval)));
                    if (!result.beat_activation.empty()) {
                        if (peak_search_start + 1 <= peak_search_end) {
                            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                                const float prev = result.beat_activation[i - 1];
                                const float curr = result.beat_activation[i];
                                const float next = result.beat_activation[i + 1];
                                if (curr >= activation_floor && curr >= prev && curr >= next) {
                                    earliest_peak = i;
                                    break;
                                }
                            }
                        }
                        if (peak_search_start + 1 <= peak_search_end) {
                            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                                const float prev = result.beat_activation[i - 1];
                                const float curr = result.beat_activation[i];
                                const float next = result.beat_activation[i + 1];
                                if (curr >= activation_floor && curr >= prev && curr >= next) {
                                    if (curr > strongest_peak_value) {
                                        strongest_peak_value = curr;
                                        strongest_peak = i;
                                    }
                                }
                            }
                        }
                        if (strongest_peak_value < 0.0f &&
                            earliest_peak < result.beat_activation.size()) {
                            strongest_peak = earliest_peak;
                            strongest_peak_value = result.beat_activation[earliest_peak];
                        }
                    }
                    if (!result.downbeat_activation.empty()) {
                        float max_downbeat = 0.0f;
                        for (std::size_t i = peak_search_start; i <= peak_search_end; ++i) {
                            max_downbeat = std::max(max_downbeat, result.downbeat_activation[i]);
                        }
                        const float onset_threshold =
                            std::max(activation_floor,
                                     max_downbeat * config.dbn_downbeat_phase_peak_ratio);
                        for (std::size_t i = peak_search_start; i <= peak_search_end; ++i) {
                            const float curr = result.downbeat_activation[i];
                            if (curr >= onset_threshold) {
                                earliest_downbeat_peak = i;
                                earliest_downbeat_value = curr;
                                break;
                            }
                        }
                    }
                    if (config.verbose) {
                        std::cerr << "DBN grid: earliest_peak=" << earliest_peak
                                  << " earliest_downbeat_peak=" << earliest_downbeat_peak
                                  << " earliest_downbeat_value=" << earliest_downbeat_value
                                  << " strongest_peak=" << strongest_peak
                                  << " strongest_peak_value=" << strongest_peak_value
                                  << " activation_floor=" << activation_floor
                                  << "\n";
                    }
                    std::size_t start_peak = decoded.beat_frames.front();
                    if (!result.beat_activation.empty()) {
                        std::size_t earliest = start_peak;
                        if (peak_search_start + 1 <= peak_search_end) {
                            for (std::size_t i = peak_search_start + 1; i + 1 <= peak_search_end; ++i) {
                                const float prev = result.beat_activation[i - 1];
                                const float curr = result.beat_activation[i];
                                const float next = result.beat_activation[i + 1];
                                if (curr >= activation_floor && curr >= prev && curr >= next) {
                                    earliest = i;
                                    break;
                                }
                            }
                        }
                        if (earliest < start_peak) {
                            start_peak = earliest;
                        }
                        if (config.dbn_grid_start_strong_peak &&
                            strongest_peak_value >= activation_floor) {
                            start_peak = strongest_peak;
                        }
                    }
                    std::vector<std::size_t> forward =
                        fill_peaks_with_grid(result.beat_activation,
                                             start_peak,
                                             used_frames - 1,
                                             base_interval,
                                             activation_floor);
                    std::vector<std::size_t> backward;
                    double cursor = static_cast<double>(start_peak) - base_interval;
                    const std::size_t window = static_cast<std::size_t>(
                        std::max(1.0, std::round(base_interval * 0.25)));
                    while (cursor >= 0.0) {
                        const std::size_t center = static_cast<std::size_t>(std::llround(cursor));
                        const std::size_t start = center > window ? center - window : 0;
                        const std::size_t end = std::min(used_frames - 1, center + window);
                        float best_value = -1.0f;
                        std::size_t best_index = center;
                        for (std::size_t k = start; k <= end; ++k) {
                            const float value = result.beat_activation[k];
                            if (value > best_value) {
                                best_value = value;
                                best_index = k;
                            }
                        }
                        std::size_t chosen = best_index;
                        if (best_value < activation_floor) {
                            chosen = center;
                        }
                        backward.push_back(chosen);
                        if (cursor < base_interval) {
                            break;
                        }
                        cursor -= base_interval;
                    }
                    std::sort(backward.begin(), backward.end());
                    std::vector<std::size_t> combined;
                    combined.reserve(backward.size() + forward.size());
                    combined.insert(combined.end(), backward.begin(), backward.end());
                    if (!forward.empty() && (combined.empty() || combined.back() != forward.front())) {
                        combined.insert(combined.end(), forward.begin(), forward.end());
                    } else if (forward.size() > 1) {
                        combined.insert(combined.end(), forward.begin() + 1, forward.end());
                    }
                    decoded.beat_frames = std::move(combined);
                }

                std::size_t best_phase = phase;
                double best_score = -std::numeric_limits<double>::infinity();
                const double local_frame_rate =
                    config.hop_size > 0 ? static_cast<double>(config.sample_rate) /
                        static_cast<double>(config.hop_size) : 0.0;
                const std::size_t phase_window_frames =
                    (config.dbn_downbeat_phase_window_seconds > 0.0 && local_frame_rate > 0.0)
                        ? static_cast<std::size_t>(std::llround(
                            config.dbn_downbeat_phase_window_seconds * local_frame_rate))
                        : 0;
                const std::size_t phase_window_start = use_window ? window_start : 0;
                const std::size_t phase_window_end =
                    (phase_window_frames > 0)
                        ? std::min(used_frames, phase_window_start + phase_window_frames)
                        : phase_window_start;
                const bool allow_downbeat_phase =
                    !result.downbeat_activation.empty() && !quality_low && downbeat_override_ok;
                const std::size_t max_delay_frames =
                    (config.dbn_downbeat_phase_max_delay_seconds > 0.0 && local_frame_rate > 0.0)
                        ? static_cast<std::size_t>(std::llround(
                            config.dbn_downbeat_phase_max_delay_seconds * local_frame_rate))
                        : 0;
                const float onset_ratio = 0.35f;
                const std::size_t onset_max_back =
                    max_delay_frames > 0 ? max_delay_frames : 8;
                auto onset_from_peak = [&](const std::vector<float>& activation,
                                           std::size_t peak_frame) -> std::size_t {
                    if (activation.empty() || peak_frame >= activation.size()) {
                        return peak_frame;
                    }
                    const float peak_value = activation[peak_frame];
                    if (peak_value <= 0.0f) {
                        return peak_frame;
                    }
                    const float threshold = peak_value * onset_ratio;
                    std::size_t frame = peak_frame;
                    std::size_t steps = 0;
                    while (frame > 0 && steps < onset_max_back) {
                        if (activation[frame] < threshold) {
                            return frame + 1;
                        }
                        --frame;
                        ++steps;
                    }
                    return frame;
                };
                auto build_onset_frames = [&](const std::vector<std::size_t>& frames,
                                              const std::vector<float>& activation) {
                    std::vector<std::size_t> out;
                    out.reserve(frames.size());
                    for (std::size_t frame : frames) {
                        out.push_back(onset_from_peak(activation, frame));
                    }
                    dedupe_frames(out);
                    return out;
                };

                float max_downbeat = 0.0f;
                float max_beat = 0.0f;
                std::vector<uint8_t> phase_peak_mask;
                bool has_phase_peaks = false;
                if (phase_window_frames > 0) {
                    for (std::size_t i = phase_window_start; i < phase_window_end; ++i) {
                        if (allow_downbeat_phase) {
                            max_downbeat = std::max(max_downbeat, result.downbeat_activation[i]);
                        }
                        if (i < result.beat_activation.size()) {
                            max_beat = std::max(max_beat, result.beat_activation[i]);
                        }
                    }
                    phase_peak_mask.assign(used_frames, 0);
                    if (max_beat > 0.0f && phase_window_end > phase_window_start + 2) {
                        const float beat_threshold =
                            static_cast<float>(max_beat * config.dbn_downbeat_phase_peak_ratio);
                        const float peak_eps = std::max(1e-6f, max_beat * 0.01f);
                        for (std::size_t i = phase_window_start + 1; i + 1 < phase_window_end; ++i) {
                            const float prev = result.beat_activation[i - 1];
                            const float curr = result.beat_activation[i];
                            const float next = result.beat_activation[i + 1];
                            if (curr >= beat_threshold &&
                                curr >= prev + peak_eps &&
                                curr >= next + peak_eps) {
                                const std::size_t onset_frame =
                                    onset_from_peak(result.beat_activation, i);
                                if (onset_frame < phase_peak_mask.size()) {
                                    phase_peak_mask[onset_frame] = 1;
                                }
                                has_phase_peaks = true;
                            }
                        }
                    }
                    if (config.dbn_trace && allow_downbeat_phase) {
                        const std::size_t limit =
                            phase_window_end > phase_window_start
                                ? (phase_window_end - phase_window_start)
                                : 0;
                        const std::size_t beat_preview =
                            std::min<std::size_t>(12, result.beat_activation.size());
                        std::cerr << "DBN: beat head:";
                        for (std::size_t i = 0; i < beat_preview; ++i) {
                            std::cerr << " " << i << "->" << result.beat_activation[i];
                        }
                        std::cerr << "\n";
                        const std::size_t preview =
                            std::min<std::size_t>(12, result.downbeat_activation.size());
                        std::cerr << "DBN: downbeat head:";
                        for (std::size_t i = 0; i < preview; ++i) {
                            std::cerr << " " << i << "->" << result.downbeat_activation[i];
                        }
                        std::cerr << "\n";
                        std::cerr << "DBN: downbeat max=" << max_downbeat
                                  << " beat max=" << max_beat
                                  << " activation_floor=" << activation_floor
                                  << "\n";
                        struct Peak {
                            float value;
                            std::size_t frame;
                        };
                        std::vector<Peak> peaks;
                        const std::size_t start =
                            (phase_window_start + 1 < phase_window_end)
                                ? (phase_window_start + 1)
                                : phase_window_start;
                        const std::size_t end =
                            phase_window_end > 1 ? phase_window_end - 1 : phase_window_end;
                        for (std::size_t i = start; i <= end; ++i) {
                            const float prev = result.downbeat_activation[i - 1];
                            const float curr = result.downbeat_activation[i];
                            const float next = result.downbeat_activation[i + 1];
                            if (curr >= prev && curr >= next) {
                                peaks.push_back({curr, i});
                            }
                        }
                        std::sort(peaks.begin(), peaks.end(),
                                  [](const Peak& a, const Peak& b) { return a.value > b.value; });
                        std::cerr << "DBN: downbeat peaks (phase window "
                                  << phase_window_start << "-" << phase_window_end << "):";
                        const std::size_t top = std::min<std::size_t>(5, peaks.size());
                        for (std::size_t i = 0; i < top; ++i) {
                            std::cerr << " " << peaks[i].frame << "->" << peaks[i].value;
                        }
                        std::cerr << "\n";
                        std::vector<Peak> global_peaks;
                        const std::size_t global_end =
                            result.downbeat_activation.empty() ? 0 : used_frames > 1
                                ? used_frames - 1 : 0;
                        for (std::size_t i = start; i <= global_end; ++i) {
                            const float prev = result.downbeat_activation[i - 1];
                            const float curr = result.downbeat_activation[i];
                            const float next = result.downbeat_activation[i + 1];
                            if (curr >= prev && curr >= next) {
                                global_peaks.push_back({curr, i});
                            }
                        }
                        std::sort(global_peaks.begin(), global_peaks.end(),
                                  [](const Peak& a, const Peak& b) { return a.value > b.value; });
                        std::cerr << "DBN: downbeat peaks (global top):";
                        const std::size_t global_top = std::min<std::size_t>(6, global_peaks.size());
                        for (std::size_t i = 0; i < global_top; ++i) {
                            const std::size_t frame = global_peaks[i].frame;
                            const double time_s = fps > 0.0 ? static_cast<double>(frame) / fps : 0.0;
                            std::cerr << " " << frame << "(" << time_s << "s)"
                                      << "->" << global_peaks[i].value;
                        }
                        std::cerr << "\n";
                        std::cerr << "DBN: phase peaks for selection (beat-only, strict): "
                                  << (has_phase_peaks ? "picked" : "none")
                                  << " (limit=" << limit << ")\n";
                    }
                }
                const float downbeat_threshold =
                    max_downbeat > 0.0f
                        ? static_cast<float>(max_downbeat * config.dbn_downbeat_phase_peak_ratio)
                        : 0.0f;
                struct PhaseDebug {
                    std::size_t phase = 0;
                    double score = -std::numeric_limits<double>::infinity();
                    std::size_t first_frame = 0;
                    std::size_t hits = 0;
                    double mean = 0.0;
                    double delay_penalty = 0.0;
                    const char* source = "none";
                };
                std::vector<PhaseDebug> phase_debug;
                if (config.dbn_trace) {
                    phase_debug.reserve(bpb);
                }
                for (std::size_t candidate_phase = 0; candidate_phase < bpb; ++candidate_phase) {
                    const auto projected =
                        project_downbeats_from_beats(decoded.beat_frames, bpb, candidate_phase);
                    if (projected.empty()) {
                        continue;
                    }
                    const std::vector<float>& onset_activation =
                        allow_downbeat_phase
                            ? result.downbeat_activation
                            : result.beat_activation;
                    const auto projected_onsets =
                        build_onset_frames(projected, onset_activation);
                    const auto& phase_frames =
                        projected_onsets.empty() ? projected : projected_onsets;
                    double score = -std::numeric_limits<double>::infinity();
                    std::size_t hits = 0;
                    double mean = 0.0;
                    const char* source = "none";
                    bool score_set = false;

                if (allow_downbeat_phase) {
                    const std::size_t frame = phase_frames.front();
                    // Force earliest phase when downbeat activations exist.
                    score = 1000000.0 - static_cast<double>(frame);
                    hits = 1;
                    mean = (frame < result.downbeat_activation.size())
                            ? static_cast<double>(result.downbeat_activation[frame])
                            : 0.0;
                        source = "downbeat_earliest";
                        score_set = true;
                    }
                    if (phase_window_frames > 0 && !allow_downbeat_phase) {
                        if (has_phase_peaks) {
                            double sum = 0.0;
                            double weight = 0.0;
                            for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                                const std::size_t frame = phase_frames[i];
                                if (frame >= phase_window_frames) {
                                    break;
                                }
                                if (frame < phase_peak_mask.size() && phase_peak_mask[frame]) {
                                    const double value =
                                        (frame < result.beat_activation.size())
                                            ? static_cast<double>(result.beat_activation[frame])
                                            : 0.0;
                                    sum += value;
                                    weight += 1.0;
                                }
                            }
                            if (weight > 0.0) {
                                score = (sum / weight);
                                hits = static_cast<std::size_t>(weight);
                                mean = score;
                                source = "beat_peak_mask";
                            }
                        } else {
                            double sum = 0.0;
                            double weight = 0.0;
                            const float threshold = max_beat > 0.0f
                                ? static_cast<float>(
                                    max_beat * config.dbn_downbeat_phase_peak_ratio)
                                : 0.0f;
                            for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                                const std::size_t frame = phase_frames[i];
                                if (frame >= phase_window_frames) {
                                    break;
                                }
                                if (frame < result.beat_activation.size()) {
                                    const float value = result.beat_activation[frame];
                                    if (value >= threshold) {
                                        sum += static_cast<double>(value);
                                        weight += 1.0;
                                    }
                                }
                            }
                            if (weight > 0.0) {
                                score = (sum / weight);
                                hits = static_cast<std::size_t>(weight);
                                mean = score;
                                source = "beat_threshold";
                            }
                        }
                    } else if (!score_set && phase_window_frames > 0 && allow_downbeat_phase) {
                        double sum = 0.0;
                        double weight = 0.0;
                        const float threshold = max_downbeat > 0.0f
                            ? static_cast<float>(
                                max_downbeat * config.dbn_downbeat_phase_peak_ratio)
                            : 0.0f;
                        const std::size_t end =
                            std::min(phase_window_end, result.downbeat_activation.size());
                        const double decay = std::max(1.0, phase_window_frames * 0.2);
                        for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                            const std::size_t frame = phase_frames[i];
                            if (frame >= end) {
                                break;
                            }
                            const float value = result.downbeat_activation[frame];
                            if (value >= threshold) {
                                const double w = std::exp(-static_cast<double>(frame) / decay);
                                sum += static_cast<double>(value) * w;
                                weight += w;
                            }
                        }
                        if (weight > 0.0) {
                            score = (sum / weight);
                            hits = static_cast<std::size_t>(weight + 0.5);
                            mean = score;
                            source = "downbeat_window_decay";
                            score_set = true;
                        }
                    }
                    if (!score_set && !std::isfinite(score)) {
                        const std::size_t max_checks = std::min<std::size_t>(phase_frames.size(), 3);
                        double sum = 0.0;
                        if (allow_downbeat_phase) {
                            const float threshold = max_downbeat > 0.0f
                                ? static_cast<float>(
                                    max_downbeat * config.dbn_downbeat_phase_peak_ratio)
                                : 0.0f;
                            for (std::size_t i = 0; i < max_checks; ++i) {
                                const std::size_t frame = phase_frames[i];
                                if (frame < result.downbeat_activation.size()) {
                                    const float value = result.downbeat_activation[frame];
                                    if (value >= threshold) {
                                        sum += value;
                                        hits += 1;
                                    }
                                }
                            }
                            if (hits == 0) {
                                for (std::size_t i = 0; i < max_checks; ++i) {
                                    const std::size_t frame = phase_frames[i];
                                    if (frame < result.downbeat_activation.size()) {
                                        sum += result.downbeat_activation[frame];
                                    }
                                }
                                hits = max_checks;
                            }
                        } else {
                            for (std::size_t i = 0; i < max_checks; ++i) {
                                const std::size_t frame = phase_frames[i];
                                if (frame < result.beat_activation.size()) {
                                    sum += result.beat_activation[frame];
                                }
                            }
                            hits = max_checks;
                        }
                        const double penalty = 0.01 * static_cast<double>(phase_frames.front());
                        score = sum - penalty;
                        mean = (max_checks > 0) ? (sum / static_cast<double>(max_checks)) : 0.0;
                        source = allow_downbeat_phase ? "fallback_downbeat" : "fallback_beat";
                    }
                    if (!score_set && !std::isfinite(score) && allow_downbeat_phase) {
                        const std::size_t early_limit = std::min<std::size_t>(
                            phase_window_end,
                            static_cast<std::size_t>(std::max(1.0, local_frame_rate * 2.0)));
                        double sum = 0.0;
                        std::size_t count = 0;
                        for (std::size_t i = 0; i < phase_frames.size(); ++i) {
                            const std::size_t frame = phase_frames[i];
                            if (frame >= early_limit) {
                                break;
                            }
                            if (frame < result.downbeat_activation.size()) {
                                sum += result.downbeat_activation[frame];
                                count += 1;
                            }
                        }
                        if (count > 0) {
                            score = sum / static_cast<double>(count);
                            hits = count;
                            mean = score;
                            source = "downbeat_early_energy";
                            score_set = true;
                        }
                    }
                    double delay_penalty = 0.0;
                    if (max_delay_frames > 0 && phase_frames.front() > max_delay_frames &&
                        source != nullptr && std::strncmp(source, "fallback", 8) != 0 &&
                        std::strncmp(source, "downbeat_", 9) != 0 &&
                        !(quality_low &&
                          (std::strncmp(source, "beat_peak_mask", 14) == 0 ||
                           std::strncmp(source, "beat_threshold", 14) == 0))) {
                        const double delay = static_cast<double>(phase_frames.front() - max_delay_frames);
                        delay_penalty = delay * 1000.0;
                        score -= delay_penalty;
                    }
                    if (score > best_score) {
                        best_score = score;
                        best_phase = candidate_phase;
                    }
                    if (config.dbn_trace) {
                        PhaseDebug entry;
                        entry.phase = candidate_phase;
                        entry.score = score;
                        entry.first_frame = phase_frames.front();
                        entry.hits = hits;
                        entry.mean = mean;
                        entry.delay_penalty = delay_penalty;
                        entry.source = source;
                        phase_debug.push_back(entry);
                    }
                }
                if (config.verbose) {
                    std::cerr << "DBN: phase_window_frames=" << phase_window_frames
                              << " max_downbeat=" << max_downbeat
                              << " threshold=" << downbeat_threshold
                              << " best_phase=" << best_phase
                              << " best_score=" << best_score
                              << "\n";
                }
                if (config.dbn_trace && !phase_debug.empty()) {
                    std::vector<PhaseDebug> sorted = phase_debug;
                    std::sort(sorted.begin(), sorted.end(),
                              [](const PhaseDebug& a, const PhaseDebug& b) {
                                  return a.score > b.score;
                              });
                    const PhaseDebug& top = sorted.front();
                    const PhaseDebug* runner = (sorted.size() > 1) ? &sorted[1] : nullptr;
                    std::cerr << "DBN: phase winner="
                              << top.phase
                              << " score=" << top.score
                              << " first=" << top.first_frame
                              << " hits=" << top.hits
                              << " mean=" << top.mean
                              << " penalty=" << top.delay_penalty
                              << " src=" << top.source;
                    if (runner) {
                        std::cerr << " runner=" << runner->phase
                                  << " score=" << runner->score
                                  << " first=" << runner->first_frame
                                  << " hits=" << runner->hits
                                  << " mean=" << runner->mean
                                  << " penalty=" << runner->delay_penalty
                                  << " src=" << runner->source;
                    }
                    std::cerr << "\n";
                    std::cerr << "DBN: phase candidates:";
                    for (const auto& entry : phase_debug) {
                        std::cerr << " p" << entry.phase
                                  << " score=" << entry.score
                                  << " first=" << entry.first_frame
                                  << " hits=" << entry.hits
                                  << " mean=" << entry.mean
                                  << " penalty=" << entry.delay_penalty
                                  << " src=" << entry.source;
                    }
                    std::cerr << "\n";
                }
                decoded.downbeat_frames =
                    project_downbeats_from_beats(decoded.beat_frames, bpb, best_phase);
                if (config.dbn_trace) {
                    const std::size_t preview =
                        std::min<std::size_t>(6, decoded.downbeat_frames.size());
                    std::cerr << "DBN: downbeat frames head:";
                    for (std::size_t i = 0; i < preview; ++i) {
                        const std::size_t frame = decoded.downbeat_frames[i];
                        const double time_s = fps > 0.0 ? static_cast<double>(frame) / fps : 0.0;
                        std::cerr << " " << frame << "(" << time_s << "s)";
                    }
                    std::cerr << "\n";
                    if (!decoded.downbeat_frames.empty()) {
                        const std::size_t first_frame = decoded.downbeat_frames.front();
                        const double first_time =
                            fps > 0.0 ? static_cast<double>(first_frame) / fps : 0.0;
                        std::cerr << "DBN: downbeat selection start="
                                  << first_frame << " (" << first_time << "s)"
                                  << " bpb=" << bpb
                                  << " phase=" << best_phase
                                  << " score=" << best_score
                                  << "\n";
                    }
                }
                // Force a uniform grid so projection yields evenly spaced beats.
                if (decoded.beat_frames.size() >= 2 && step_frames > 1.0) {
                    if (config.dbn_grid_global_fit && decoded.beat_frames.size() >= 8) {
                        const double n = static_cast<double>(decoded.beat_frames.size());
                        double sx = 0.0;
                        double sy = 0.0;
                        double sxx = 0.0;
                        double sxy = 0.0;
                        for (std::size_t i = 0; i < decoded.beat_frames.size(); ++i) {
                            const double x = static_cast<double>(i);
                            const double y = static_cast<double>(decoded.beat_frames[i]);
                            sx += x;
                            sy += y;
                            sxx += x * x;
                            sxy += x * y;
                        }
                        const double den = n * sxx - sx * sx;
                        if (std::abs(den) > 1e-9) {
                            const double fit_step = (n * sxy - sx * sy) / den;
                            if (fit_step > 1.0) {
                                if (config.verbose) {
                                    std::cerr << "DBN grid fit: step_frames(raw)=" << step_frames
                                              << " step_frames(fit)=" << fit_step
                                              << " beats=" << decoded.beat_frames.size()
                                              << "\n";
                                }
                                step_frames = fit_step;
                            }
                        }
                    }
                    const bool have_downbeat_start = !decoded.downbeat_frames.empty();
                    const bool reliable_downbeat_start =
                        have_downbeat_start &&
                        max_downbeat > activation_floor;
                    const std::size_t start = reliable_downbeat_start
                        ? decoded.downbeat_frames.front()
                        : std::min(decoded.beat_frames.front(),
                                   std::min(earliest_peak, earliest_downbeat_peak));
                    double grid_start = static_cast<double>(start);
                    if (earliest_downbeat_peak > 0 && earliest_downbeat_peak < start) {
                        grid_start = static_cast<double>(earliest_downbeat_peak);
                    }
                    if (!reliable_downbeat_start &&
                        config.dbn_grid_start_strong_peak &&
                        strongest_peak_value >= activation_floor) {
                        grid_start = static_cast<double>(strongest_peak);
                    }
                    if (!reliable_downbeat_start &&
                        config.dbn_grid_align_downbeat_peak &&
                        earliest_downbeat_peak > 0 &&
                        step_frames > 1.0) {
                        const double offset =
                            static_cast<double>(earliest_downbeat_peak) -
                            grid_start;
                        const double half_step = step_frames * 0.5;
                        if (std::abs(offset) <= half_step) {
                            const double adjusted = grid_start + offset;
                            if (adjusted >= 0.0) {
                                grid_start = adjusted;
                            }
                        } else if (earliest_downbeat_peak < grid_start) {
                            // If the first downbeat peak is clearly earlier, bias the grid to it.
                            grid_start = static_cast<double>(earliest_downbeat_peak);
                        }
                    }
                    if (config.dbn_grid_start_advance_seconds > 0.0f &&
                        fps > 0.0) {
                        const double frames_per_second = fps;
                        grid_start -= static_cast<double>(config.dbn_grid_start_advance_seconds) *
                            frames_per_second;
                    }
                    if (consensus_phase_valid && step_frames > 1.0 && !reliable_downbeat_start) {
                        double target = std::fmod(consensus_phase_frames, step_frames);
                        if (target < 0.0) {
                            target += step_frames;
                        }
                        const double anchor = (earliest_downbeat_peak > 0)
                            ? static_cast<double>(earliest_downbeat_peak)
                            : static_cast<double>(strongest_peak);
                        const double k = std::round((anchor - target) / step_frames);
                        grid_start = target + k * step_frames;
                        while (grid_start < 0.0) {
                            grid_start += step_frames;
                        }
                    }
                    if (!reliable_downbeat_start && step_frames > 1.0 &&
                        result.beat_activation.size() >= 8) {
                        const auto phase_score = [&](double start_frame) -> double {
                            if (start_frame < 0.0) {
                                return -1.0;
                            }
                            double score = 0.0;
                            std::size_t hits = 0;
                            double cursor_local = start_frame;
                            while (cursor_local >= step_frames) {
                                cursor_local -= step_frames;
                            }
                            while (cursor_local < static_cast<double>(used_frames) && hits < 128) {
                                const long long idx_ll = static_cast<long long>(std::llround(cursor_local));
                                if (idx_ll >= 0 &&
                                    static_cast<std::size_t>(idx_ll) < result.beat_activation.size()) {
                                    const std::size_t idx = static_cast<std::size_t>(idx_ll);
                                    float value = result.beat_activation[idx];
                                    if (idx > 0) {
                                        value = std::max(value, result.beat_activation[idx - 1]);
                                    }
                                    if (idx + 1 < result.beat_activation.size()) {
                                        value = std::max(value, result.beat_activation[idx + 1]);
                                    }
                                    score += static_cast<double>(value);
                                    hits += 1;
                                }
                                cursor_local += step_frames;
                            }
                            return hits > 0 ? (score / static_cast<double>(hits)) : -1.0;
                        };
                        const double alt_start = grid_start + (0.5 * step_frames);
                        const double base_score = phase_score(grid_start);
                        const double alt_score = phase_score(alt_start);
                        if (alt_score > base_score) {
                            grid_start = alt_start;
                            if (config.verbose) {
                                std::cerr << "DBN grid: half-step phase shift selected"
                                          << " base_score=" << base_score
                                          << " alt_score=" << alt_score
                                          << "\n";
                            }
                        }
                    }
                    if (grid_start < 0.0) {
                        grid_start = 0.0;
                    }
                    std::vector<std::size_t> uniform_beats;
                    double cursor = grid_start;
                    while (cursor >= step_frames) {
                        cursor -= step_frames;
                        uniform_beats.push_back(
                            static_cast<std::size_t>(std::llround(cursor)));
                    }
                    std::reverse(uniform_beats.begin(), uniform_beats.end());
                    cursor = grid_start;
                    while (cursor < static_cast<double>(used_frames)) {
                        uniform_beats.push_back(
                            static_cast<std::size_t>(std::llround(cursor)));
                        cursor += step_frames;
                    }
                    dedupe_frames(uniform_beats);
                    decoded.beat_frames = std::move(uniform_beats);
                    decoded.downbeat_frames =
                        project_downbeats_from_beats(decoded.beat_frames, bpb, best_phase);
                    if (config.dbn_trace && fps > 0.0) {
                        auto collect_peaks = [](const std::vector<float>& activation,
                                                float floor,
                                                std::vector<std::size_t>& peaks_out) {
                            peaks_out.clear();
                            if (activation.size() < 3) {
                                return;
                            }
                            const std::size_t end = activation.size() - 1;
                            for (std::size_t i = 1; i < end; ++i) {
                                const float prev = activation[i - 1];
                                const float curr = activation[i];
                                const float next = activation[i + 1];
                                if (curr >= floor && curr >= prev && curr >= next) {
                                    peaks_out.push_back(i);
                                }
                            }
                        };
                        auto compute_offsets = [&](const std::vector<std::size_t>& grid,
                                                   const std::vector<std::size_t>& peaks,
                                                   const char* label) {
                            if (grid.empty() || peaks.empty()) {
                                std::cerr << "DBN align: " << label
                                          << "_peak_offset_s mean=nan std=nan count=0\n";
                                return;
                            }
                            double sum = 0.0;
                            double sum_sq = 0.0;
                            std::size_t count = 0;
                            for (const auto frame : grid) {
                                auto it = std::lower_bound(peaks.begin(), peaks.end(), frame);
                                std::size_t best = *peaks.begin();
                                if (it == peaks.end()) {
                                    best = peaks.back();
                                } else {
                                    best = *it;
                                    if (it != peaks.begin()) {
                                        const std::size_t prev = *(it - 1);
                                        if (frame - prev < best - frame) {
                                            best = prev;
                                        }
                                    }
                                }
                                const double delta = (static_cast<double>(best) -
                                                      static_cast<double>(frame)) / fps;
                                sum += delta;
                                sum_sq += delta * delta;
                                count += 1;
                                if (count >= 64) {
                                    break;
                                }
                            }
                            const double mean = sum / static_cast<double>(count);
                            const double var = (sum_sq / static_cast<double>(count)) - mean * mean;
                            const double stddev = var > 0.0 ? std::sqrt(var) : 0.0;
                            std::cerr << "DBN align: " << label
                                      << "_peak_offset_s mean=" << mean
                                      << " std=" << stddev
                                      << " count=" << count
                                      << "\n";
                        };
                        std::vector<std::size_t> beat_peaks;
                        std::vector<std::size_t> downbeat_peaks;
                        const float beat_floor = activation_floor;
                        const float downbeat_floor = activation_floor;
                        collect_peaks(result.beat_activation, beat_floor, beat_peaks);
                        collect_peaks(result.downbeat_activation, downbeat_floor, downbeat_peaks);
                        compute_offsets(decoded.beat_frames, beat_peaks, "beat");
                        compute_offsets(decoded.downbeat_frames, downbeat_peaks, "downbeat");
                    }
                    if (config.verbose) {
                        std::cerr << "DBN grid: start=" << start
                                  << " grid_start=" << grid_start
                                  << " strongest_peak=" << strongest_peak
                                  << " strongest_peak_value=" << strongest_peak_value
                                  << " earliest_downbeat_peak=" << earliest_downbeat_peak
                                  << " advance_s=" << config.dbn_grid_start_advance_seconds
                                  << "\n";
                    }
                }
            }

            fill_beats_from_frames_raw(decoded.beat_frames);
            if (config.dbn_project_grid && decoded.beat_frames.size() >= 2 && projected_bpm > 0.0) {
                fill_beats_from_bpm_grid_into(decoded.beat_frames.front(),
                                              projected_bpm,
                                              grid_total_frames,
                                              result.beat_projected_feature_frames,
                                              result.beat_projected_sample_frames,
                                              result.beat_projected_strengths);
                std::vector<std::size_t> projected_frames;
                projected_frames.reserve(result.beat_projected_feature_frames.size());
                for (unsigned long long frame : result.beat_projected_feature_frames) {
                    projected_frames.push_back(static_cast<std::size_t>(frame));
                }
                const std::vector<std::size_t> adjusted_downbeats =
                    apply_latency_to_frames(decoded.downbeat_frames);
                const std::vector<std::size_t> projected_downbeats =
                    align_downbeats_to_beats(projected_frames, adjusted_downbeats);
                result.downbeat_projected_feature_frames.clear();
                result.downbeat_projected_feature_frames.reserve(projected_downbeats.size());
                for (std::size_t frame : projected_downbeats) {
                    result.downbeat_projected_feature_frames.push_back(
                        static_cast<unsigned long long>(frame));
                }
            } else {
                result.beat_projected_feature_frames.clear();
                result.beat_projected_sample_frames.clear();
                result.beat_projected_strengths.clear();
                result.downbeat_projected_feature_frames.clear();
            }
            result.downbeat_feature_frames.clear();
            const std::vector<std::size_t> adjusted_downbeats =
                apply_latency_to_frames(decoded.downbeat_frames);
            result.downbeat_feature_frames.reserve(adjusted_downbeats.size());
            for (std::size_t frame : adjusted_downbeats) {
                result.downbeat_feature_frames.push_back(static_cast<unsigned long long>(frame));
            }
            if (config.profile) {
                std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                          << "ms peaks=" << peaks_ms << "ms\n";
            }
            return result;
        }
    }

    if (!config.use_dbn && config.use_minimal_postprocess) {
        const std::vector<std::size_t> beat_peaks = compute_minimal_peaks(result.beat_activation);
        const std::vector<std::size_t> downbeat_peaks =
            compute_minimal_peaks(result.downbeat_activation);
        fill_beats_from_frames(beat_peaks);
        const std::vector<std::size_t> aligned_downbeats =
            align_downbeats_to_beats(beat_peaks, downbeat_peaks);
        result.downbeat_feature_frames.clear();
        result.downbeat_feature_frames.reserve(aligned_downbeats.size());
        for (std::size_t frame : aligned_downbeats) {
            result.downbeat_feature_frames.push_back(static_cast<unsigned long long>(frame));
        }
        if (config.profile) {
            std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                      << "ms peaks=" << peaks_ms << "ms\n";
        }
        return result;
    }

    auto compute_peaks = [&](const std::vector<float>& activation,
                             float local_min_bpm,
                             float local_max_bpm,
                             float threshold) {
        const auto peaks_start = std::chrono::steady_clock::now();
        const double max_bpm_local = std::max(local_min_bpm + 1.0f, local_max_bpm);
        const double min_bpm_local = std::max(1.0f, local_min_bpm);
        const std::size_t min_interval =
            static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm_local)));
        const std::size_t max_interval =
            static_cast<std::size_t>(std::ceil((60.0 * fps) / min_bpm_local));
        auto peaks = pick_peaks(activation, threshold, min_interval, max_interval);
        const auto peaks_end = std::chrono::steady_clock::now();
        peaks_ms += std::chrono::duration<double, std::milli>(peaks_end - peaks_start).count();
        return peaks;
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

    fill_beats_from_frames(peaks);

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

    if (config.profile) {
        std::cerr << "Timing(postprocess): dbn=" << dbn_ms
                  << "ms peaks=" << peaks_ms << "ms\n";
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
    NSURL* compiled_url = compile_model_if_needed(model_url, &error);
    if (compiled_url) {
        model_url = compiled_url;
    }
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
