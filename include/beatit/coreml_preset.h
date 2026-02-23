//
//  coreml_preset.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/coreml.h"

#include <memory>
#include <string>
#include <vector>

namespace beatit {

class CoreMLPreset {
public:
    virtual ~CoreMLPreset() = default;
    virtual const char* name() const = 0;
    virtual void apply(CoreMLConfig& config) const = 0;
};

std::unique_ptr<CoreMLPreset> make_coreml_preset(const std::string& name);
std::vector<std::string> coreml_preset_names();

} // namespace beatit
