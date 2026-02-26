//
//  logging.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/logging.hpp"

#include "beatit/coreml.h"

namespace beatit {

namespace {

std::atomic<int> g_log_level{static_cast<int>(LogVerbosity::Info)};

} // namespace

void set_log_verbosity(LogVerbosity level) {
    g_log_level.store(static_cast<int>(level), std::memory_order_relaxed);
}

LogVerbosity get_log_verbosity() {
    return static_cast<LogVerbosity>(g_log_level.load(std::memory_order_relaxed));
}

void set_log_verbosity_from_config(const CoreMLConfig& config) {
    LogVerbosity level = config.log_verbosity;
    if (config.profile && static_cast<int>(level) < static_cast<int>(LogVerbosity::Info)) {
        level = LogVerbosity::Info;
    }
    set_log_verbosity(level);
}

} // namespace beatit
