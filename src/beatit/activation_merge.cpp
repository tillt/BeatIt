//
//  activation_merge.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/activation_merge.h"

#include <algorithm>

namespace beatit {
namespace detail {

void trim_activation_to_frames(std::vector<float>* activation, std::size_t frame_count) {
    if (!activation) {
        return;
    }
    if (activation->size() > frame_count) {
        activation->resize(frame_count);
    }
}

void merge_window_activations(std::vector<float>* merged_beat,
                              std::vector<float>* merged_downbeat,
                              std::size_t window_offset,
                              std::size_t frame_count,
                              const std::vector<float>& beat_activation,
                              const std::vector<float>& downbeat_activation,
                              std::size_t border_frames) {
    if (!merged_beat || !merged_downbeat || frame_count == 0) {
        return;
    }

    const std::size_t needed = window_offset + frame_count;
    if (merged_beat->size() < needed) {
        merged_beat->resize(needed, 0.0f);
        merged_downbeat->resize(needed, 0.0f);
    }

    const std::size_t border = std::min(border_frames, frame_count / 2);
    const std::size_t first = border;
    const std::size_t last = frame_count - border;

    for (std::size_t i = first; i < last; ++i) {
        const std::size_t idx = window_offset + i;
        if (i < beat_activation.size()) {
            (*merged_beat)[idx] = std::max((*merged_beat)[idx], beat_activation[i]);
        }
        if (i < downbeat_activation.size()) {
            (*merged_downbeat)[idx] =
                std::max((*merged_downbeat)[idx], downbeat_activation[i]);
        }
    }
}

} // namespace detail
} // namespace beatit
