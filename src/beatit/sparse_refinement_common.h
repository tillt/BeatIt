//
//  sparse_refinement_common.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <vector>

namespace beatit {
namespace detail {

double sparse_median_frame_diff(const std::vector<unsigned long long>& frames);

} // namespace detail
} // namespace beatit
