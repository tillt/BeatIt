//
//  wav.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>

namespace beatit::detail {

bool write_wav_mono_16(const std::string& path,
                       const std::vector<float>& samples,
                       double sample_rate,
                       std::string* error) {
    if (samples.empty() || sample_rate <= 0.0) {
        if (error) {
            *error = "Empty samples or invalid sample rate.";
        }
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        if (error) {
            *error = "Failed to open WAV output.";
        }
        return false;
    }

    const std::uint16_t channels = 1;
    const std::uint16_t bits_per_sample = 16;
    const std::uint32_t sample_rate_u = static_cast<std::uint32_t>(std::lround(sample_rate));
    const std::uint32_t byte_rate = sample_rate_u * channels * (bits_per_sample / 8);
    const std::uint16_t block_align = channels * (bits_per_sample / 8);
    const std::uint32_t data_size =
        static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    const std::uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riff_size), sizeof(riff_size));
    out.write("WAVE", 4);

    out.write("fmt ", 4);
    const std::uint32_t fmt_size = 16;
    const std::uint16_t audio_format = 1;
    out.write(reinterpret_cast<const char*>(&fmt_size), sizeof(fmt_size));
    out.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
    out.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    out.write(reinterpret_cast<const char*>(&sample_rate_u), sizeof(sample_rate_u));
    out.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    out.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
    out.write(reinterpret_cast<const char*>(&bits_per_sample), sizeof(bits_per_sample));

    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

    for (float sample : samples) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const std::int16_t value = static_cast<std::int16_t>(std::lround(clamped * 32767.0f));
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    if (!out.good()) {
        if (error) {
            *error = "Failed to write WAV data.";
        }
        return false;
    }

    return true;
}

} // namespace beatit::detail
