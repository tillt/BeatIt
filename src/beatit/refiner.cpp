//
//  refiner.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/refiner.h"

#include "refiner_constant_internal.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

namespace beatit {
namespace {

const unsigned long long kBeatEventMaskMarkers =
    BeatEventStyleMarkIntro |
    BeatEventStyleMarkBuildup |
    BeatEventStyleMarkTeardown |
    BeatEventStyleMarkOutro |
    BeatEventStyleMarkStart |
    BeatEventStyleMarkEnd |
    BeatEventStyleMarkChapter;

} // namespace

const unsigned long long BeatEventMaskMarkers = kBeatEventMaskMarkers;

std::string beat_event_csv_header() {
    return "index,frame,bpm,energy,peak,style";
}

std::string beat_event_csv_preamble(std::size_t downbeat_phase,
                                    long long phase_shift_frames,
                                    bool used_model_downbeat,
                                    double downbeat_coverage,
                                    std::size_t active_start_frame,
                                    std::size_t active_end_frame,
                                    std::size_t found_count,
                                    std::size_t total_count,
                                    std::size_t max_missing_run,
                                    double avg_missing_run) {
    std::ostringstream out;
    out << "# downbeat_phase=" << downbeat_phase
        << " phase_shift_frames=" << phase_shift_frames;
    out << " model_downbeat=" << (used_model_downbeat ? 1 : 0)
        << " downbeat_coverage=" << std::setprecision(4) << downbeat_coverage;
    out << " active_start=" << active_start_frame
        << " active_end=" << active_end_frame;
    if (total_count > 0) {
        out.setf(std::ios::fixed);
        out << " found=" << found_count
            << " total=" << total_count
            << " found_ratio=" << std::setprecision(4)
            << (static_cast<double>(found_count) / static_cast<double>(total_count))
            << " max_missing_run=" << max_missing_run
            << " avg_missing_run=" << std::setprecision(2) << avg_missing_run;
    }
    return out.str();
}

std::string beat_event_to_csv(const BeatEvent& event) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out << event.index << ","
        << event.frame << ","
        << std::setprecision(6) << event.bpm << ","
        << event.energy << ","
        << event.peak << ","
        << static_cast<unsigned long long>(event.style);
    return out.str();
}

void write_beat_events_csv(std::ostream& out,
                           const std::vector<BeatEvent>& events,
                           bool include_header,
                           std::size_t downbeat_phase,
                           long long phase_shift_frames,
                           bool used_model_downbeat,
                           double downbeat_coverage,
                           std::size_t active_start_frame,
                           std::size_t active_end_frame,
                           std::size_t found_count,
                           std::size_t total_count,
                           std::size_t max_missing_run,
                           double avg_missing_run) {
    out << beat_event_csv_preamble(downbeat_phase,
                                   phase_shift_frames,
                                   used_model_downbeat,
                                   downbeat_coverage,
                                   active_start_frame,
                                   active_end_frame,
                                   found_count,
                                   total_count,
                                   max_missing_run,
                                   avg_missing_run)
        << "\n";
    if (include_header) {
        out << beat_event_csv_header() << "\n";
    }

    for (const BeatEvent& event : events) {
        out << beat_event_to_csv(event) << "\n";
    }
}

std::vector<BeatEvent> build_shakespear_markers(const std::vector<unsigned long long>& beat_feature_frames,
                                                const std::vector<unsigned long long>& beat_sample_frames,
                                                const std::vector<unsigned long long>& downbeat_feature_frames,
                                                const std::vector<float>* beat_activation,
                                                double bpm,
                                                double sample_rate,
                                                const CoreMLConfig& config) {
    std::vector<BeatEvent> events;
    if (beat_feature_frames.empty()) {
        return events;
    }

    const std::size_t beat_count_assumption = (beat_feature_frames.size() >> 2) << 2;
    const std::size_t bar_count_assumption = beat_count_assumption / 4;

    std::size_t downbeat_phase = 0;
    if (!downbeat_feature_frames.empty()) {
        for (std::size_t i = 0; i < beat_feature_frames.size(); ++i) {
            const unsigned long long frame = beat_feature_frames[i];
            if (std::find(downbeat_feature_frames.begin(),
                          downbeat_feature_frames.end(),
                          frame) != downbeat_feature_frames.end()) {
                downbeat_phase = i % 4;
                break;
            }
        }
    }

    unsigned long intro_bar_count = 32U;
    unsigned long buildup_bar_count = 64U;
    unsigned long teardown_bar_count;
    unsigned long outro_bar_count;

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

    const double hop_scale = sample_rate / static_cast<double>(config.sample_rate);

    events.reserve(beat_feature_frames.size());
    for (std::size_t beat_index = 0; beat_index < beat_feature_frames.size(); ++beat_index) {
        const unsigned long long frame = beat_feature_frames[beat_index];
        BeatEvent event;
        event.bpm = bpm;
        event.index = beat_index;
        event.style = BeatEventStyleBeat;
        if (beat_index % 4 == downbeat_phase) {
            event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleBar);
        }
        event.style = static_cast<BeatEventStyle>(event.style | BeatEventStyleFound);

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

        if (beat_index < beat_sample_frames.size()) {
            event.frame = beat_sample_frames[beat_index];
        } else {
            const double sample_pos = static_cast<double>(frame * config.hop_size) * hop_scale;
            event.frame = static_cast<unsigned long long>(std::llround(sample_pos));
        }

        if (beat_activation && frame < beat_activation->size()) {
            event.energy = static_cast<double>((*beat_activation)[frame]);
            event.peak = detail::compute_local_peak(beat_activation, frame, 2);
        }

        events.push_back(event);
    }

    if (!events.empty()) {
        BeatEvent& last_event = events.back();
        last_event.style =
            static_cast<BeatEventStyle>(last_event.style | BeatEventStyleMarkEnd);
    }

    return events;
}

} // namespace beatit
