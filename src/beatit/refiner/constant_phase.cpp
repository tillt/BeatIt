//
//  constant_phase.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-26.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "constant_internal.h"

#include <algorithm>
#include <cmath>

namespace beatit {
namespace detail {

double compute_local_peak(const std::vector<float>* activation,
                          unsigned long long frame,
                          std::size_t window) {
    if (!activation || activation->empty()) {
        return 0.0;
    }

    const std::size_t size = activation->size();
    const std::size_t center =
        static_cast<std::size_t>(std::min<unsigned long long>(frame, size - 1));
    const std::size_t start = center >= window ? center - window : 0;
    const std::size_t end = std::min(size, center + window + 1);

    double peak = 0.0;
    for (std::size_t i = start; i < end; ++i) {
        peak = std::max(peak, static_cast<double>((*activation)[i]));
    }
    return peak;
}

bool is_found_peak(const std::vector<unsigned long long>& coarse_beats,
                   std::size_t* coarse_index,
                   unsigned long long frame,
                   unsigned long long tolerance_frames) {
    if (coarse_beats.empty() || !coarse_index) {
        return false;
    }

    std::size_t idx = *coarse_index;
    while (idx + 1 < coarse_beats.size() && coarse_beats[idx + 1] <= frame) {
        ++idx;
    }
    *coarse_index = idx;

    const unsigned long long current = coarse_beats[idx];
    if (frame >= current) {
        if (frame - current <= tolerance_frames) {
            return true;
        }
    } else if (current - frame <= tolerance_frames) {
        return true;
    }

    if (idx + 1 < coarse_beats.size()) {
        const unsigned long long next = coarse_beats[idx + 1];
        if (next >= frame && next - frame <= tolerance_frames) {
            return true;
        }
    }

    return false;
}

double score_phase_shift(const std::vector<unsigned long long>& beat_frames,
                         const std::vector<float>* energy,
                         long long shift_frames,
                         std::size_t total_frames) {
    if (!energy || energy->empty()) {
        return 0.0;
    }

    double score = 0.0;
    const std::size_t size = energy->size();
    for (unsigned long long frame : beat_frames) {
        const long long shifted = static_cast<long long>(frame) + shift_frames;
        if (shifted < 0) {
            continue;
        }
        const std::size_t idx = static_cast<std::size_t>(shifted);
        if (idx >= total_frames || idx >= size) {
            continue;
        }
        score += static_cast<double>((*energy)[idx]);
    }

    return score;
}

long long choose_half_beat_shift(const std::vector<unsigned long long>& beat_frames,
                                 const std::vector<float>* energy,
                                 long long half_beat_frames,
                                 std::size_t total_frames,
                                 double margin) {
    if (half_beat_frames <= 0 || !energy || energy->empty()) {
        return 0;
    }

    const double base_score = score_phase_shift(beat_frames, energy, 0, total_frames);
    const double plus_score = score_phase_shift(beat_frames, energy, half_beat_frames, total_frames);
    const double minus_score = score_phase_shift(beat_frames, energy, -half_beat_frames, total_frames);

    double best_score = base_score;
    long long best_shift = 0;
    if (plus_score > best_score) {
        best_score = plus_score;
        best_shift = half_beat_frames;
    }
    if (minus_score > best_score) {
        best_score = minus_score;
        best_shift = -half_beat_frames;
    }

    if (best_shift != 0 && base_score > 0.0) {
        const double improvement = (best_score - base_score) / base_score;
        if (improvement < margin) {
            return 0;
        }
    }

    return best_shift;
}

std::size_t choose_downbeat_phase(const std::vector<unsigned long long>& beat_frames,
                                  const std::vector<float>* downbeat_activation,
                                  const std::vector<float>* phase_energy,
                                  double low_freq_weight) {
    if (beat_frames.empty()) {
        return 0;
    }

    constexpr std::size_t kPeakWindow = 2;

    double scores[4] = {0.0, 0.0, 0.0, 0.0};

    float max_downbeat = 0.0f;
    if (downbeat_activation) {
        for (float value : *downbeat_activation) {
            max_downbeat = std::max(max_downbeat, value);
        }
    }

    float max_phase = 0.0f;
    if (phase_energy) {
        for (float value : *phase_energy) {
            max_phase = std::max(max_phase, value);
        }
    }

    for (std::size_t i = 0; i < beat_frames.size(); ++i) {
        double score = 0.0;

        if (downbeat_activation && !downbeat_activation->empty() && max_downbeat > 0.0f) {
            const std::size_t frame =
                static_cast<std::size_t>(
                    std::min<unsigned long long>(beat_frames[i], downbeat_activation->size() - 1));
            score += static_cast<double>((*downbeat_activation)[frame]) / max_downbeat;
        }

        if (phase_energy && !phase_energy->empty() && max_phase > 0.0f) {
            const double peak =
                compute_local_peak(phase_energy, beat_frames[i], kPeakWindow) / max_phase;
            score += low_freq_weight * peak;
        }

        scores[i % 4] += score;
    }

    std::size_t best = 0;
    double best_score = scores[0];
    for (std::size_t i = 1; i < 4; ++i) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best = i;
        }
    }

    return best;
}

std::size_t last_active_frame(const std::vector<float>* energy,
                              std::size_t total_frames,
                              double active_energy_ratio) {
    if (!energy || energy->empty()) {
        return total_frames;
    }

    float max_value = 0.0f;
    for (float value : *energy) {
        if (value > max_value) {
            max_value = value;
        }
    }

    if (max_value <= 0.0f) {
        return total_frames;
    }

    const float threshold = static_cast<float>(max_value * active_energy_ratio);
    for (std::size_t i = energy->size(); i > 0; --i) {
        if ((*energy)[i - 1] >= threshold) {
            return std::min(total_frames, i);
        }
    }

    return total_frames;
}

std::size_t first_active_frame(const std::vector<float>* energy,
                               std::size_t total_frames,
                               double active_energy_ratio) {
    if (!energy || energy->empty()) {
        return 0;
    }

    float max_value = 0.0f;
    for (float value : *energy) {
        if (value > max_value) {
            max_value = value;
        }
    }

    if (max_value <= 0.0f) {
        return 0;
    }

    const float threshold = static_cast<float>(max_value * active_energy_ratio);
    for (std::size_t i = 0; i < energy->size(); ++i) {
        if ((*energy)[i] >= threshold) {
            return std::min(total_frames, i);
        }
    }

    return 0;
}

} // namespace detail
} // namespace beatit
