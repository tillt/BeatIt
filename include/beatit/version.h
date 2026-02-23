//
//  version.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <string>

namespace beatit {

/// @brief Return the BeatIt version display string.
///
/// Mirrors CLI version output (for example: `v1.2.0` or `v1.2.0+abcd123`).
std::string version_string();

} // namespace beatit
