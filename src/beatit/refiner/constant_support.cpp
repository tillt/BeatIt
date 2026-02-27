//
//  constant_support.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "constant_internal.h"

#include <algorithm>

namespace beatit {
namespace detail {

namespace {

float max_activation_value(const std::vector<float>* activation) {
    if (!activation || activation->empty()) {
        return 0.0f;
    }
    float max_value = 0.0f;
    for (float value : *activation) {
        max_value = std::max(max_value, value);
    }
    return max_value;
}

} // namespace

DownbeatPhaseSource choose_downbeat_phase_source(const std::vector<unsigned long long>& beat_feature_frames,
                                                 const std::vector<float>* downbeat_activation,
                                                 const ConstantBeatRefinerConfig& refiner_config) {
    DownbeatPhaseSource source;
    source.use_model = refiner_config.use_downbeat_anchor;
    source.activation = downbeat_activation;

    if (!source.use_model || !refiner_config.auto_downbeat_fallback) {
        return source;
    }

    const float max_downbeat = max_activation_value(downbeat_activation);
    if (!downbeat_activation || downbeat_activation->empty() || max_downbeat <= 0.0f) {
        source.use_model = false;
        source.activation = nullptr;
        return source;
    }

    std::size_t covered = 0;
    for (unsigned long long frame : beat_feature_frames) {
        if (frame >= downbeat_activation->size()) {
            continue;
        }
        const float value = (*downbeat_activation)[static_cast<std::size_t>(frame)];
        if (value >= max_downbeat * refiner_config.downbeat_min_strength_ratio) {
            ++covered;
        }
    }

    if (!beat_feature_frames.empty()) {
        source.coverage =
            static_cast<double>(covered) / static_cast<double>(beat_feature_frames.size());
    }
    if (source.coverage < refiner_config.downbeat_min_coverage) {
        source.use_model = false;
        source.activation = nullptr;
    }

    return source;
}

void apply_marker_style(BeatEvent& event,
                        std::size_t beat_index,
                        std::size_t intro_beats_starting_at,
                        std::size_t buildup_beats_starting_at,
                        std::size_t teardown_beats_starting_at,
                        std::size_t outro_beats_starting_at) {
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
}

void apply_alarm_style(BeatEvent& event,
                       std::size_t beat_index,
                       std::size_t intro_beats_starting_at,
                       std::size_t buildup_beats_starting_at,
                       std::size_t teardown_beats_starting_at,
                       std::size_t outro_beats_starting_at) {
    if (beat_index >= outro_beats_starting_at) {
        event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleAlarmOutro);
    } else if (beat_index >= teardown_beats_starting_at) {
        event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleAlarmTeardown);
    } else if (beat_index < intro_beats_starting_at) {
        event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleAlarmIntro);
    } else if (beat_index < buildup_beats_starting_at) {
        event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleAlarmBuildup);
    }
}

void summarize_found_stats(ConstantBeatResult& result) {
    result.total_count = result.beat_events.size();
    if (result.total_count == 0) {
        return;
    }

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

void fill_event_exports(ConstantBeatResult& result) {
    result.beat_sample_frames.reserve(result.beat_feature_frames.size());
    result.beat_strengths.reserve(result.beat_feature_frames.size());
    for (const BeatEvent& event : result.beat_events) {
        result.beat_sample_frames.push_back(event.frame);
        result.beat_strengths.push_back(static_cast<float>(event.peak));
    }
}

} // namespace detail
} // namespace beatit
