#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "coreml_model_utils.h"

#include <algorithm>
#include <string>
#include <vector>

@interface BeatitBundleAnchor : NSObject
@end

@implementation BeatitBundleAnchor
@end

namespace beatit {
namespace detail {

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

} // namespace detail
} // namespace beatit
