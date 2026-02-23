//
//  wav.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include <string>
#include <vector>

namespace beatit::detail {

bool write_wav_mono_16(const std::string& path,
                       const std::vector<float>& samples,
                       double sample_rate,
                       std::string* error);

} // namespace beatit::detail
