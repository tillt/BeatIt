//
//  coreml_metadata.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "beatit/coreml.h"

#include "coreml_model_utils.h"

#include <string>

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

namespace beatit {

CoreMLMetadata load_coreml_metadata(const CoreMLConfig& config) {
    CoreMLMetadata metadata;

    NSString* model_path = detail::resolve_model_path(config);
    if (!model_path) {
        return metadata;
    }

    NSURL* model_url = [NSURL fileURLWithPath:model_path];
    NSError* error = nil;
    NSURL* compiled_url = detail::compile_model_if_needed(model_url, &error);
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
