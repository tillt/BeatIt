//
//  constant.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "constant_internal.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

ConstantBeatResult refine_constant_beats(const std::vector<unsigned long long>& beat_feature_frames,
                                         std::size_t total_frames,
                                         const BeatitConfig& config,
                                         double sample_rate,
                                         std::size_t initial_silence_frames,
                                         const ConstantBeatRefinerConfig& refiner_config,
                                         const std::vector<float>* beat_activation,
                                         const std::vector<float>* downbeat_activation,
                                         const std::vector<float>* phase_energy) {
    ConstantBeatResult result;
    if (beat_feature_frames.size() < 2 || total_frames == 0) {
        return result;
    }

    const double frame_rate =
        static_cast<double>(config.sample_rate) / static_cast<double>(config.hop_size);
    const std::vector<detail::BeatConstRegion> regions =
        detail::retrieve_constant_regions(beat_feature_frames, frame_rate, refiner_config);
    if (regions.size() < 2) {
        return result;
    }

    const std::vector<float>* energy = phase_energy ? phase_energy : beat_activation;
    std::size_t silence_frames = initial_silence_frames;
    if (refiner_config.auto_active_trim && silence_frames == 0) {
        silence_frames =
            detail::first_active_frame(energy, total_frames, refiner_config.active_energy_ratio);
    }

    const std::size_t active_frames =
        detail::last_active_frame(energy, total_frames, refiner_config.active_energy_ratio);
    result.active_start_frame = silence_frames;
    result.active_end_frame = active_frames;

    long long first_beat_frame = 0;
    const double const_bpm =
        detail::make_constant_bpm(regions, frame_rate, refiner_config, &first_beat_frame, silence_frames);
    if (const_bpm <= 0.0) {
        return result;
    }

    const long long adjusted_first =
        detail::adjust_phase(first_beat_frame, const_bpm, beat_feature_frames, frame_rate, refiner_config);
    const double beat_length = 60.0 * frame_rate / const_bpm;

    long long phase_shift_frames = 0;
    if (refiner_config.use_half_beat_correction) {
        const long long half_beat_frames = static_cast<long long>(std::llround(beat_length * 0.5));
        phase_shift_frames =
            detail::choose_half_beat_shift(beat_feature_frames,
                                           energy,
                                           half_beat_frames,
                                           active_frames,
                                           refiner_config.half_beat_score_margin);
    }

    const long long shifted_first = adjusted_first + phase_shift_frames;
    double next_beat_frame = static_cast<double>(shifted_first);
    bool fake_first = shifted_first < 0;
    if (fake_first) {
        next_beat_frame = 0.0;
    }

    result.bpm = static_cast<float>(const_bpm);
    result.first_beat_feature_frame = shifted_first;
    result.phase_shift_frames = phase_shift_frames;

    std::size_t beat_count_assumption =
        static_cast<std::size_t>((static_cast<double>(active_frames) + (beat_length - 1.0)) / beat_length);

    // Align to a 4/4 grid for the marker scheme.
    beat_count_assumption = (beat_count_assumption >> 2) << 2;

    const unsigned long long tolerance_frames =
        static_cast<unsigned long long>(std::max(1.0, std::round(refiner_config.max_phase_error_seconds * frame_rate)));
    std::size_t coarse_index = 0;

    std::size_t beat_index = 0;
    bool use_model_downbeat = refiner_config.use_downbeat_anchor;
    const std::vector<float>* downbeat_for_phase = downbeat_activation;
    if (use_model_downbeat && refiner_config.auto_downbeat_fallback) {
        if (!downbeat_activation || downbeat_activation->empty()) {
            use_model_downbeat = false;
            downbeat_for_phase = nullptr;
        } else {
            float max_downbeat = 0.0f;
            for (float value : *downbeat_activation) {
                max_downbeat = std::max(max_downbeat, value);
            }
            if (max_downbeat <= 0.0f) {
                use_model_downbeat = false;
                downbeat_for_phase = nullptr;
            } else {
                std::size_t covered = 0;
                for (unsigned long long frame : beat_feature_frames) {
                    if (frame >= downbeat_activation->size()) {
                        continue;
                    }
                    const float value = (*downbeat_activation)[static_cast<std::size_t>(frame)];
                    if (value >= max_downbeat * refiner_config.downbeat_min_strength_ratio) {
                        covered++;
                    }
                }
                if (!beat_feature_frames.empty()) {
                    result.downbeat_coverage =
                        static_cast<double>(covered) /
                        static_cast<double>(beat_feature_frames.size());
                }
                if (result.downbeat_coverage < refiner_config.downbeat_min_coverage) {
                    use_model_downbeat = false;
                    downbeat_for_phase = nullptr;
                }
            }
        }
    }
    const std::size_t downbeat_phase =
        detail::choose_downbeat_phase(beat_feature_frames,
                                      downbeat_for_phase,
                                      phase_energy,
                                      refiner_config.low_freq_weight);
    result.downbeat_phase = downbeat_phase;
    result.used_model_downbeat = use_model_downbeat;

    // Segment markers are defined in bars (4 beats) and aligned to the bar phase.
    unsigned long intro_bar_count = 32U;
    unsigned long buildup_bar_count = 64U;
    unsigned long teardown_bar_count;
    unsigned long outro_bar_count;

    const std::size_t bar_count_assumption = beat_count_assumption / 4;
    if (bar_count_assumption < 3 * buildup_bar_count) {
        buildup_bar_count >>= 1;
        intro_bar_count >>= 1;
    }

    outro_bar_count = intro_bar_count;
    teardown_bar_count = buildup_bar_count;

    const std::size_t intro_beats_starting_at =
        downbeat_phase + (static_cast<std::size_t>(intro_bar_count) * 4);
    const std::size_t buildup_beats_starting_at =
        downbeat_phase + (static_cast<std::size_t>(buildup_bar_count) * 4);
    const std::size_t teardown_beats_starting_at =
        downbeat_phase + (bar_count_assumption - teardown_bar_count) * 4;
    const std::size_t outro_beats_starting_at =
        downbeat_phase + (bar_count_assumption - outro_bar_count) * 4;
    // Clamp to real audio activity so padding does not create synthetic beats.
    while (next_beat_frame < static_cast<double>(active_frames)) {
        const auto frame = static_cast<unsigned long long>(std::llround(next_beat_frame));
        result.beat_feature_frames.push_back(frame);
        if (beat_index % 4 == downbeat_phase) {
            result.downbeat_feature_frames.push_back(frame);
        }

        BeatEvent event;
        event.bpm = const_bpm;
        event.index = beat_index;
        event.style = BeatEventStyleBeat;
        if (beat_index % 4 == downbeat_phase) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleBar);
        }
        if (detail::is_found_peak(beat_feature_frames, &coarse_index, frame, tolerance_frames)) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleFound);
        }

        if (beat_index == 0) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleMarkStart);
        } else if (beat_index == intro_beats_starting_at) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleMarkIntro);
        } else if (beat_index == buildup_beats_starting_at) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleMarkBuildup);
        } else if (beat_index == teardown_beats_starting_at) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleMarkTeardown);
        } else if (beat_index == outro_beats_starting_at) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleMarkOutro);
        }

        if (beat_index >= outro_beats_starting_at) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleAlarmOutro);
        } else if (beat_index >= teardown_beats_starting_at) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleAlarmTeardown);
        } else if (beat_index < intro_beats_starting_at) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleAlarmIntro);
        } else if (beat_index < buildup_beats_starting_at) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleAlarmBuildup);
        }

        if (fake_first) {
            next_beat_frame = static_cast<double>(shifted_first) + beat_length;
            fake_first = false;
        } else {
            next_beat_frame += beat_length;
        }
        beat_index++;

        const double hop_scale = sample_rate / static_cast<double>(config.sample_rate);
        const double sample_pos = static_cast<double>(frame * config.hop_size) * hop_scale;
        event.frame = static_cast<unsigned long long>(std::llround(sample_pos));

        if (beat_activation && frame < beat_activation->size()) {
            event.energy = static_cast<double>((*beat_activation)[frame]);
            event.peak = detail::compute_local_peak(beat_activation, frame, 2);
        }

        result.beat_events.push_back(event);
    }

    if (!result.beat_events.empty()) {
        BeatEvent& last_event = result.beat_events.back();
        last_event.style =
            static_cast<BeatEventStyle>(last_event.style | BeatEventStyleMarkEnd);
    }

    result.total_count = result.beat_events.size();
    if (result.total_count > 0) {
        std::size_t current_missing = 0;
        std::size_t total_missing = 0;
        std::size_t missing_runs = 0;
        for (const BeatEvent& event : result.beat_events) {
            const bool found = (event.style & BeatEventStyleFound) == BeatEventStyleFound;
            if (found) {
                result.found_count++;
                if (current_missing > 0) {
                    result.max_missing_run = std::max(result.max_missing_run, current_missing);
                    total_missing += current_missing;
                    missing_runs++;
                    current_missing = 0;
                }
            } else {
                current_missing++;
            }
        }
        if (current_missing > 0) {
            result.max_missing_run = std::max(result.max_missing_run, current_missing);
            total_missing += current_missing;
            missing_runs++;
        }
        result.found_ratio =
            static_cast<double>(result.found_count) / static_cast<double>(result.total_count);
        if (missing_runs > 0) {
            result.avg_missing_run =
                static_cast<double>(total_missing) / static_cast<double>(missing_runs);
        }
    }

    result.beat_sample_frames.reserve(result.beat_feature_frames.size());
    result.beat_strengths.reserve(result.beat_feature_frames.size());
    for (const BeatEvent& event : result.beat_events) {
        result.beat_sample_frames.push_back(event.frame);
        result.beat_strengths.push_back(static_cast<float>(event.peak));
    }

    return result;
}

} // namespace beatit
