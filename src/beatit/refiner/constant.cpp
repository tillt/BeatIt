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
    const detail::DownbeatPhaseSource downbeat_source =
        detail::choose_downbeat_phase_source(beat_feature_frames,
                                             downbeat_activation,
                                             refiner_config);
    const std::size_t downbeat_phase =
        detail::choose_downbeat_phase(beat_feature_frames,
                                      downbeat_source.activation,
                                      phase_energy,
                                      refiner_config.low_freq_weight);
    result.downbeat_phase = downbeat_phase;
    result.used_model_downbeat = downbeat_source.use_model;
    result.downbeat_coverage = downbeat_source.coverage;

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

        detail::apply_marker_style(event,
                                   beat_index,
                                   intro_beats_starting_at,
                                   buildup_beats_starting_at,
                                   teardown_beats_starting_at,
                                   outro_beats_starting_at);
        detail::apply_alarm_style(event,
                                  beat_index,
                                  intro_beats_starting_at,
                                  buildup_beats_starting_at,
                                  teardown_beats_starting_at,
                                  outro_beats_starting_at);

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

    detail::summarize_found_stats(result);
    detail::fill_event_exports(result);

    return result;
}

} // namespace detail
} // namespace beatit
