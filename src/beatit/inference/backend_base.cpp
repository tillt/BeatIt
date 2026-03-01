//
//  backend_base.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-03-01.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/inference/backend.h"

namespace beatit {
namespace detail {

bool InferenceBackend::infer_windows(const std::vector<std::vector<float>>& windows,
                                     const BeatitConfig& config,
                                     std::vector<std::vector<float>>* beats,
                                     std::vector<std::vector<float>>* downbeats,
                                     InferenceTiming* timing) {
    if (!beats || !downbeats) {
        return false;
    }

    beats->clear();
    downbeats->clear();
    beats->reserve(windows.size());
    downbeats->reserve(windows.size());

    for (const auto& window : windows) {
        std::vector<float> beat;
        std::vector<float> downbeat;
        if (!infer_window(window, config, &beat, &downbeat, timing)) {
            return false;
        }
        beats->push_back(std::move(beat));
        downbeats->push_back(std::move(downbeat));
    }

    return true;
}

} // namespace detail
} // namespace beatit
