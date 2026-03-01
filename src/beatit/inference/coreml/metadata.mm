//
//  metadata.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "beatit/config.h"

#include "model_utils.h"

#include <algorithm>
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

namespace {

NSDictionary* parse_coreml_metadata_json(NSData* metadata_data) {
    if (!metadata_data) {
        return nil;
    }

    NSError* json_error = nil;
    id json_root = [NSJSONSerialization JSONObjectWithData:metadata_data options:0 error:&json_error];
    if (json_error || ![json_root isKindOfClass:[NSArray class]]) {
        return nil;
    }

    NSArray* entries = static_cast<NSArray*>(json_root);
    if (entries.count == 0) {
        return nil;
    }

    id first_entry = entries.firstObject;
    if (![first_entry isKindOfClass:[NSDictionary class]]) {
        return nil;
    }

    return static_cast<NSDictionary*>(first_entry);
}

NSDictionary* load_package_metadata_via_coremlcompiler(NSURL* source_model_url) {
    if (!source_model_url ||
        ![[source_model_url pathExtension].lowercaseString isEqualToString:@"mlpackage"]) {
        return nil;
    }

    NSString* xcrun_path = @"/usr/bin/xcrun";
    if (![[NSFileManager defaultManager] isExecutableFileAtPath:xcrun_path]) {
        return nil;
    }

    NSTask* task = [[NSTask alloc] init];
    task.launchPath = xcrun_path;
    task.arguments = @[ @"coremlcompiler", @"metadata", source_model_url.path ];

    NSPipe* stdout_pipe = [NSPipe pipe];
    NSPipe* stderr_pipe = [NSPipe pipe];
    task.standardOutput = stdout_pipe;
    task.standardError = stderr_pipe;

    @try {
        [task launch];
        [task waitUntilExit];
    } @catch (NSException*) {
        return nil;
    }

    if (task.terminationStatus != 0) {
        return nil;
    }

    NSData* output_data = [[stdout_pipe fileHandleForReading] readDataToEndOfFile];
    return parse_coreml_metadata_json(output_data);
}

} // namespace

namespace beatit {

CoreMLMetadata load_coreml_metadata(const BeatitConfig& config) {
    CoreMLMetadata metadata;

    NSString* model_path = detail::resolve_model_path(config);
    if (!model_path) {
        return metadata;
    }

    NSURL* source_model_url = [NSURL fileURLWithPath:model_path];
    NSURL* model_url = source_model_url;
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
    if (info) {
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
    }

    const bool needs_json_fallback =
        metadata.author.empty() ||
        metadata.short_description.empty() ||
        metadata.license.empty() ||
        metadata.version.empty();
    if (!needs_json_fallback) {
        return metadata;
    }

    auto load_metadata_dict = ^NSDictionary* {
        NSMutableArray<NSURL*>* candidates = [NSMutableArray array];
        if ([[compiled_url pathExtension].lowercaseString isEqualToString:@"mlmodelc"]) {
            [candidates addObject:[compiled_url URLByAppendingPathComponent:@"metadata.json"]];
        }

        if ([[source_model_url pathExtension].lowercaseString isEqualToString:@"mlpackage"]) {
            NSString* package_path = source_model_url.path;
            NSString* sibling_path = [[package_path stringByDeletingPathExtension]
                stringByAppendingPathExtension:@"mlmodelc"];
            NSURL* sibling_url = [NSURL fileURLWithPath:sibling_path];
            [candidates addObject:[sibling_url URLByAppendingPathComponent:@"metadata.json"]];
        }

        for (NSURL* candidate in candidates) {
            if (!candidate ||
                ![[NSFileManager defaultManager] fileExistsAtPath:candidate.path]) {
                continue;
            }

            NSData* metadata_data = [NSData dataWithContentsOfURL:candidate];
            if (!metadata_data) {
                continue;
            }

            NSDictionary* metadata_dict = parse_coreml_metadata_json(metadata_data);
            if (metadata_dict) {
                return metadata_dict;
            }
        }

        NSDictionary* package_metadata = load_package_metadata_via_coremlcompiler(source_model_url);
        if (package_metadata) {
            return package_metadata;
        }

        return static_cast<NSDictionary*>(nil);
    };

    NSDictionary* metadata_dict = load_metadata_dict();
    if (!metadata_dict) {
        return metadata;
    }

    auto assign_json_string = [&](NSString* key, std::string* target) {
        if (!target->empty()) {
            return;
        }
        id value = [metadata_dict objectForKey:key];
        if ([value isKindOfClass:[NSString class]]) {
            *target = [static_cast<NSString*>(value) UTF8String];
        }
    };

    assign_json_string(@"author", &metadata.author);
    assign_json_string(@"shortDescription", &metadata.short_description);
    assign_json_string(@"license", &metadata.license);
    assign_json_string(@"version", &metadata.version);

    id user_defined = [metadata_dict objectForKey:@"userDefinedMetadata"];
    if ([user_defined isKindOfClass:[NSDictionary class]]) {
        NSDictionary* user_dict = static_cast<NSDictionary*>(user_defined);
        for (id key in user_dict) {
            id value = [user_dict objectForKey:key];
            if ([key isKindOfClass:[NSString class]] && [value isKindOfClass:[NSString class]]) {
                const std::string key_string = [static_cast<NSString*>(key) UTF8String];
                const std::string value_string = [static_cast<NSString*>(value) UTF8String];
                const auto already_present = std::find_if(
                    metadata.user_defined.begin(),
                    metadata.user_defined.end(),
                    [&](const auto& entry) { return entry.first == key_string; });
                if (already_present == metadata.user_defined.end()) {
                    metadata.user_defined.emplace_back(key_string, value_string);
                }
            }
        }
    }

    return metadata;
}

} // namespace beatit
