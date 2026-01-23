//
//  refiner.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/refiner.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

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

struct BeatConstRegion {
    std::size_t first_beat_frame = 0;
    double beat_length = 0.0;
};

double round_bpm_within_range(double min_bpm, double center_bpm, double max_bpm) {
    double snap_bpm = std::round(center_bpm);
    if (snap_bpm > min_bpm && snap_bpm < max_bpm) {
        return snap_bpm;
    }

    const double round_bpm_width = max_bpm - min_bpm;
    if (round_bpm_width > 0.5) {
        if (center_bpm < 85.0) {
            return std::round(center_bpm * 2.0) / 2.0;
        }
        if (center_bpm > 127.0) {
            return std::round(center_bpm / 3.0 * 2.0) * 3.0 / 2.0;
        }
    }

    if (round_bpm_width > (1.0 / 12.0)) {
        return std::round(center_bpm * 12.0) / 12.0;
    }

    snap_bpm = std::round(center_bpm * 12.0) / 12.0;
    if (snap_bpm > min_bpm && snap_bpm < max_bpm) {
        return snap_bpm;
    }
    return center_bpm;
}

std::vector<BeatConstRegion> retrieve_constant_regions(const std::vector<unsigned long long>& coarse_beats,
                                                       double frame_rate,
                                                       const ConstantBeatRefinerConfig& config) {
    std::vector<BeatConstRegion> regions;
    if (coarse_beats.empty()) {
        return regions;
    }

    const double max_error = config.max_phase_error_seconds * frame_rate;
    const double max_error_sum = config.max_phase_error_sum_seconds * frame_rate;

    const auto border_error = [&](std::size_t window_start,
                                  std::size_t window_end,
                                  double mean_length) -> double {
        if (window_end <= window_start + 2) {
            return 0.0;
        }
        const double left_len =
            static_cast<double>(coarse_beats[window_start + 1] - coarse_beats[window_start]);
        const double right_len =
            static_cast<double>(coarse_beats[window_end] - coarse_beats[window_end - 1]);
        return std::fabs(left_len + right_len - (2.0 * mean_length));
    };

    std::size_t window_start = 0;
    std::size_t window_end = coarse_beats.size() - 1;

    while (window_start < coarse_beats.size() - 1) {
        const double mean_length =
            static_cast<double>(coarse_beats[window_end] - coarse_beats[window_start]) /
            static_cast<double>(window_end - window_start);

        std::size_t outliers = 0;
        double expected = static_cast<double>(coarse_beats[window_start]);
        double drift = 0.0;
        std::size_t probe = window_start + 1;

        for (; probe <= window_end; ++probe) {
            expected += mean_length;
            const double error = expected - static_cast<double>(coarse_beats[probe]);
            drift += error;
            if (std::fabs(drift) > max_error_sum) {
                break;
            }
            if (std::fabs(error) > max_error) {
                outliers++;
                if (outliers > config.max_outliers || probe == window_start + 1) {
                    break;
                }
            }
        }

        if (probe > window_end && border_error(window_start, window_end, mean_length) <= max_error / 2.0) {
            regions.push_back({static_cast<std::size_t>(coarse_beats[window_start]), mean_length});
            window_start = window_end;
            window_end = coarse_beats.size() - 1;
            continue;
        }

        if (window_end == 0) {
            break;
        }
        window_end--;
    }

    regions.push_back({static_cast<std::size_t>(coarse_beats.back()), 0.0});
    return regions;
}

double make_constant_bpm(const std::vector<BeatConstRegion>& regions,
                         double frame_rate,
                         const ConstantBeatRefinerConfig& config,
                         long long* out_first_beat,
                         std::size_t initial_silence_frames) {
    if (regions.size() < 2) {
        return 0.0;
    }

    int mid_region_index = 0;
    double longest_region_length = 0.0;
    double longest_region_beat_length = 0.0;
    int longest_region_number_of_beats = 0;

    for (std::size_t i = 0; i + 1 < regions.size(); ++i) {
        const double length =
            static_cast<double>(regions[i + 1].first_beat_frame - regions[i].first_beat_frame);
        if (regions[i].beat_length <= 0.0) {
            continue;
        }
        const int beat_count = static_cast<int>((length / regions[i].beat_length) + 0.5);
        if (beat_count > longest_region_number_of_beats) {
            longest_region_length = length;
            longest_region_beat_length = regions[i].beat_length;
            longest_region_number_of_beats = beat_count;
            mid_region_index = static_cast<int>(i);
        }
    }

    if (longest_region_length == 0.0) {
        return 0.0;
    }

    double longest_region_beat_length_min =
        longest_region_beat_length -
        ((config.max_phase_error_seconds * frame_rate) / longest_region_number_of_beats);
    double longest_region_beat_length_max =
        longest_region_beat_length +
        ((config.max_phase_error_seconds * frame_rate) / longest_region_number_of_beats);

    int start_region_index = mid_region_index;

    for (int i = 0; i < mid_region_index; ++i) {
        const double length =
            static_cast<double>(regions[i + 1].first_beat_frame - regions[i].first_beat_frame);
        const int number_of_beats = static_cast<int>((length / regions[i].beat_length) + 0.5);
        if (number_of_beats < static_cast<int>(config.min_region_beats)) {
            continue;
        }
        const double this_region_beat_length_min =
            regions[i].beat_length -
            ((config.max_phase_error_seconds * frame_rate) / number_of_beats);
        const double this_region_beat_length_max =
            regions[i].beat_length +
            ((config.max_phase_error_seconds * frame_rate) / number_of_beats);
        if (longest_region_beat_length > this_region_beat_length_min &&
            longest_region_beat_length < this_region_beat_length_max) {
            const double new_longest_region_length =
                static_cast<double>(regions[mid_region_index + 1].first_beat_frame -
                                    regions[i].first_beat_frame);

            const double beat_length_min =
                std::max(longest_region_beat_length_min, this_region_beat_length_min);
            const double beat_length_max =
                std::min(longest_region_beat_length_max, this_region_beat_length_max);

            const int max_number_of_beats =
                static_cast<int>(std::round(new_longest_region_length / beat_length_min));
            const int min_number_of_beats =
                static_cast<int>(std::round(new_longest_region_length / beat_length_max));
            if (min_number_of_beats != max_number_of_beats) {
                continue;
            }
            const int number_of_beats = min_number_of_beats;
            const double new_beat_length = new_longest_region_length / number_of_beats;
            if (new_beat_length > longest_region_beat_length_min &&
                new_beat_length < longest_region_beat_length_max) {
                longest_region_length = new_longest_region_length;
                longest_region_beat_length = new_beat_length;
                longest_region_number_of_beats = number_of_beats;
                longest_region_beat_length_min =
                    longest_region_beat_length -
                    ((config.max_phase_error_seconds * frame_rate) / longest_region_number_of_beats);
                longest_region_beat_length_max =
                    longest_region_beat_length +
                    ((config.max_phase_error_seconds * frame_rate) / longest_region_number_of_beats);
                start_region_index = i;
                break;
            }
        }
    }

    for (std::size_t i = regions.size() - 2; i > static_cast<std::size_t>(mid_region_index); --i) {
        const double length =
            static_cast<double>(regions[i + 1].first_beat_frame - regions[i].first_beat_frame);
        const int number_of_beats = static_cast<int>((length / regions[i].beat_length) + 0.5);
        if (number_of_beats < static_cast<int>(config.min_region_beats)) {
            continue;
        }
        const double this_region_beat_length_min =
            regions[i].beat_length -
            ((config.max_phase_error_seconds * frame_rate) / number_of_beats);
        const double this_region_beat_length_max =
            regions[i].beat_length +
            ((config.max_phase_error_seconds * frame_rate) / number_of_beats);
        if (longest_region_beat_length > this_region_beat_length_min &&
            longest_region_beat_length < this_region_beat_length_max) {
            const double new_longest_region_length =
                static_cast<double>(regions[i + 1].first_beat_frame -
                                    regions[start_region_index].first_beat_frame);
            const double beat_length_min =
                std::max(longest_region_beat_length_min, this_region_beat_length_min);
            const double beat_length_max =
                std::min(longest_region_beat_length_max, this_region_beat_length_max);
            const int max_number_of_beats =
                static_cast<int>(std::round(new_longest_region_length / beat_length_min));
            const int min_number_of_beats =
                static_cast<int>(std::round(new_longest_region_length / beat_length_max));
            if (min_number_of_beats != max_number_of_beats) {
                continue;
            }
            const int number_of_beats = min_number_of_beats;
            const double new_beat_length = new_longest_region_length / number_of_beats;
            if (new_beat_length > longest_region_beat_length_min &&
                new_beat_length < longest_region_beat_length_max) {
                longest_region_length = new_longest_region_length;
                longest_region_beat_length = new_beat_length;
                longest_region_number_of_beats = number_of_beats;
                break;
            }
        }
    }

    const double min_round_bpm = 60.0 * frame_rate / longest_region_beat_length_max;
    const double max_round_bpm = 60.0 * frame_rate / longest_region_beat_length_min;
    const double center_bpm = 60.0 * frame_rate / longest_region_beat_length;
    const double round_bpm = round_bpm_within_range(min_round_bpm, center_bpm, max_round_bpm);

    if (out_first_beat) {
        const double rounded_beat_length = 60.0 * frame_rate / round_bpm;
        const std::size_t first_measured_good_beat_frame =
            regions[static_cast<std::size_t>(start_region_index)].first_beat_frame;

        double possible_first_beat_offset =
            std::fmod(static_cast<double>(first_measured_good_beat_frame), rounded_beat_length);
        const double delta = rounded_beat_length - possible_first_beat_offset;
        const double error_threshold = rounded_beat_length / 4.0;
        if (delta < error_threshold) {
            possible_first_beat_offset -= rounded_beat_length;
        }

        const double skip_silence_frames =
            std::floor(static_cast<double>(initial_silence_frames) / rounded_beat_length) *
            rounded_beat_length;
        if (skip_silence_frames > 0.0) {
            possible_first_beat_offset += skip_silence_frames;
        }

        *out_first_beat = static_cast<long long>(std::llround(possible_first_beat_offset));
    }

    return round_bpm;
}

long long adjust_phase(long long first_beat,
                       double bpm,
                       const std::vector<unsigned long long>& coarse_beats,
                       double frame_rate,
                       const ConstantBeatRefinerConfig& config) {
    if (coarse_beats.empty() || bpm <= 0.0) {
        return first_beat;
    }

    const double beat_length = 60.0 * frame_rate / bpm;
    const double start_offset = std::fmod(static_cast<double>(first_beat), beat_length);
    double offset_adjust = 0.0;
    double offset_adjust_count = 0.0;

    for (unsigned long long frame : coarse_beats) {
        double offset = std::fmod(static_cast<double>(frame) - start_offset, beat_length);
        if (offset > beat_length / 2.0) {
            offset -= beat_length;
        }
        if (std::fabs(offset) < (config.max_phase_error_seconds * frame_rate)) {
            offset_adjust += offset;
            offset_adjust_count += 1.0;
        }
    }

    if (offset_adjust_count == 0.0) {
        return first_beat;
    }

    offset_adjust /= offset_adjust_count;
    return static_cast<long long>(std::llround(static_cast<double>(first_beat) + offset_adjust));
}

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

ConstantBeatResult refine_constant_beats(const std::vector<unsigned long long>& beat_feature_frames,
                                         std::size_t total_frames,
                                         const CoreMLConfig& config,
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
    const std::vector<BeatConstRegion> regions =
        retrieve_constant_regions(beat_feature_frames, frame_rate, refiner_config);
    if (regions.size() < 2) {
        return result;
    }

    const std::vector<float>* energy = phase_energy ? phase_energy : beat_activation;
    std::size_t silence_frames = initial_silence_frames;
    if (refiner_config.auto_active_trim && silence_frames == 0) {
        silence_frames =
            first_active_frame(energy, total_frames, refiner_config.active_energy_ratio);
    }

    const std::size_t active_frames =
        last_active_frame(energy, total_frames, refiner_config.active_energy_ratio);
    result.active_start_frame = silence_frames;
    result.active_end_frame = active_frames;

    long long first_beat_frame = 0;
    const double const_bpm =
        make_constant_bpm(regions, frame_rate, refiner_config, &first_beat_frame, silence_frames);
    if (const_bpm <= 0.0) {
        return result;
    }

    const long long adjusted_first =
        adjust_phase(first_beat_frame, const_bpm, beat_feature_frames, frame_rate, refiner_config);
    const double beat_length = 60.0 * frame_rate / const_bpm;

    long long phase_shift_frames = 0;
    if (refiner_config.use_half_beat_correction) {
        const long long half_beat_frames = static_cast<long long>(std::llround(beat_length * 0.5));
        phase_shift_frames =
            choose_half_beat_shift(beat_feature_frames,
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
        choose_downbeat_phase(beat_feature_frames,
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
        if (is_found_peak(beat_feature_frames, &coarse_index, frame, tolerance_frames)) {
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
            event.peak = compute_local_peak(beat_activation, frame, 2);
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
