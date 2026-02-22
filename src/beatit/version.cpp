//
//  version.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/version.h"
#include "beatit_version.hpp"

namespace beatit {

std::string version_string() {
    return BEATIT_VERSION_DISPLAY;
}

} // namespace beatit
