//
//  activation_merge.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//

#pragma once

#include <cstddef>
#include <vector>

namespace beatit {
namespace detail {

void trim_activation_to_frames(std::vector<float>* activation, std::size_t frame_count);

void merge_window_activations(std::vector<float>* merged_beat,
                              std::vector<float>* merged_downbeat,
                              std::size_t window_offset,
                              std::size_t frame_count,
                              const std::vector<float>& beat_activation,
                              const std::vector<float>& downbeat_activation,
                              std::size_t border_frames);

} // namespace detail
} // namespace beatit
