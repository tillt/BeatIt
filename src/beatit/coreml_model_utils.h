#pragma once

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "beatit/coreml.h"

#include <cstddef>
#include <vector>

namespace beatit {
namespace detail {

MLModel* load_cached_model(NSURL* model_url,
                           MLModelConfiguration* model_config,
                           NSError** error);

NSURL* compile_model_if_needed(NSURL* model_url, NSError** error);

NSString* resolve_model_path(const CoreMLConfig& config);

bool load_multiarray_from_features(MLMultiArray* array,
                                   const std::vector<float>& features,
                                   std::size_t frames,
                                   std::size_t mel_bins,
                                   CoreMLConfig::InputLayout layout);

CoreMLConfig::InputLayout infer_model_input_layout(MLModel* model,
                                                   const CoreMLConfig& config);

std::vector<float> flatten_output(MLFeatureValue* value);

} // namespace detail
} // namespace beatit
