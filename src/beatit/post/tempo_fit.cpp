//
//  tempo_fit.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/tempo_fit.h"

#include "beatit/dbn/calmdad.h"
#include "beatit/logging.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace beatit::detail {

namespace {

struct ModeCandidate {
    double bpm = 0.0;
    double conf = 0.0;
    double mode = 1.0;
};

std::pair<double, double> estimate_window_tempo(const std::vector<std::size_t>& beats,
                                                double fps) {
    if (beats.size() < 8) {
        return {0.0, 0.0};
    }

    std::vector<double> intervals;
    intervals.reserve(beats.size() - 1);
    for (std::size_t i = 1; i < beats.size(); ++i) {
        if (beats[i] > beats[i - 1]) {
            intervals.push_back(static_cast<double>(beats[i] - beats[i - 1]));
        }
    }
    if (intervals.size() < 4) {
        return {0.0, 0.0};
    }

    const std::size_t mid = intervals.size() / 2;
    std::nth_element(intervals.begin(),
                     intervals.begin() + static_cast<std::ptrdiff_t>(mid),
                     intervals.end());
    const double median_interval = intervals[mid];
    if (!(median_interval > 0.0)) {
        return {0.0, 0.0};
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    for (double value : intervals) {
        sum += value;
        sum_sq += value * value;
    }
    const double n = static_cast<double>(intervals.size());
    const double mean = sum / n;
    const double variance = std::max(0.0, (sum_sq / n) - (mean * mean));
    const double stdev = std::sqrt(variance);
    const double cv = mean > 0.0 ? (stdev / mean) : 1.0;
    const double confidence = (1.0 / (1.0 + cv)) * std::min(1.0, n / 32.0);

    return {(60.0 * fps) / median_interval, confidence};
}

std::vector<ModeCandidate> expand_mode_candidates(double bpm,
                                                  double conf,
                                                  float min_bpm,
                                                  float max_bpm) {
    std::vector<ModeCandidate> out;
    if (!(bpm > 0.0) || !(conf > 0.0)) {
        return out;
    }

    static const double kModes[] = {0.5, 1.0, 2.0, 3.0, (2.0 / 3.0), 1.5};
    for (double mode : kModes) {
        const double candidate_bpm = bpm * mode;
        if (candidate_bpm >= min_bpm && candidate_bpm <= max_bpm) {
            out.push_back({candidate_bpm, conf, mode});
        }
    }
    return out;
}

std::pair<ModeCandidate, ModeCandidate> match_mode_candidates(
    const std::vector<ModeCandidate>& intro_modes,
    const std::vector<ModeCandidate>& outro_modes) {
    double best_err = std::numeric_limits<double>::infinity();
    ModeCandidate best_intro{};
    ModeCandidate best_outro{};

    for (const auto& intro : intro_modes) {
        for (const auto& outro : outro_modes) {
            const double mean = 0.5 * (intro.bpm + outro.bpm);
            if (!(mean > 0.0)) {
                continue;
            }
            const double rel = std::abs(intro.bpm - outro.bpm) / mean;
            const double weight = std::max(1e-6, intro.conf * outro.conf);
            const double score = rel / weight;
            if (score < best_err) {
                best_err = score;
                best_intro = intro;
                best_outro = outro;
            }
        }
    }

    return {best_intro, best_outro};
}

} // namespace

double bpm_from_linear_fit(const std::vector<std::size_t>& beats, double fps) {
    if (beats.size() < 4 || fps <= 0.0) {
        return 0.0;
    }
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    const double n = static_cast<double>(beats.size());
    for (std::size_t i = 0; i < beats.size(); ++i) {
        const double x = static_cast<double>(i);
        const double y = static_cast<double>(beats[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    const double denom = (n * sum_xx - sum_x * sum_x);
    if (std::abs(denom) < 1e-9) {
        return 0.0;
    }
    const double slope = (n * sum_xy - sum_x * sum_y) / denom;
    if (slope <= 0.0) {
        return 0.0;
    }
    return (60.0 * fps) / slope;
}

double bpm_from_global_fit(const CoreMLResult& result,
                           const BeatitConfig& config,
                           const CalmdadDecoder& calmdad_decoder,
                           double fps,
                           float min_bpm,
                           float max_bpm,
                           std::size_t used_frames) {
    if (used_frames == 0 || fps <= 0.0) {
        return 0.0;
    }

    double bpm_from_global_fit = 0.0;

    const std::size_t requested_window_frames = static_cast<std::size_t>(
        std::round(config.dbn_window_seconds * fps));
    const std::size_t min_window_frames = static_cast<std::size_t>(std::round(15.0 * fps));
    const std::size_t window_frames = std::min<std::size_t>(
        used_frames, std::max<std::size_t>(requested_window_frames, min_window_frames));
    if (window_frames == 0) {
        return 0.0;
    }

    const std::size_t max_start = used_frames - window_frames;
    const std::size_t intro_start = std::min<std::size_t>(
        static_cast<std::size_t>(
            std::round(std::max(0.0, config.dbn_tempo_anchor_intro_seconds) * fps)),
        max_start);
    std::size_t outro_end = used_frames;
    const std::size_t outro_offset_frames = static_cast<std::size_t>(
        std::round(std::max(0.0, config.dbn_tempo_anchor_outro_offset_seconds) * fps));
    if (outro_offset_frames < used_frames) {
        outro_end = used_frames - outro_offset_frames;
    }
    if (outro_end < window_frames) {
        outro_end = window_frames;
    }
    if (outro_end > used_frames) {
        outro_end = used_frames;
    }
    const std::size_t outro_start =
        (outro_end > window_frames) ? (outro_end - window_frames) : 0;
    const auto intro = std::pair<std::size_t, std::size_t>{
        intro_start, std::min<std::size_t>(used_frames, intro_start + window_frames)};
    const auto outro = std::pair<std::size_t, std::size_t>{outro_start, outro_end};
    const std::size_t intro_mid = intro.first + ((intro.second - intro.first) / 2);
    const std::size_t outro_mid = outro.first + ((outro.second - outro.first) / 2);
    const std::size_t anchor_gap_frames =
        (intro_mid > outro_mid) ? (intro_mid - outro_mid) : (outro_mid - intro_mid);
    const std::size_t min_anchor_gap_frames = std::max<std::size_t>(
        window_frames, static_cast<std::size_t>(std::round(45.0 * fps)));
    const bool anchors_separated = anchor_gap_frames >= min_anchor_gap_frames;

    auto decode_window_beats = [&](const std::pair<std::size_t, std::size_t>& w) {
        std::vector<std::size_t> beats;
        if (w.second <= w.first || w.second > result.beat_activation.size()) {
            return beats;
        }
        std::vector<float> w_beat(result.beat_activation.begin() + w.first,
                                  result.beat_activation.begin() + w.second);
        std::vector<float> w_downbeat;
        if (!result.downbeat_activation.empty() &&
            w.second <= result.downbeat_activation.size()) {
            w_downbeat.assign(result.downbeat_activation.begin() + w.first,
                              result.downbeat_activation.begin() + w.second);
        }
        DBNDecodeResult tmp = calmdad_decoder.decode({
            w_beat,
            w_downbeat,
            fps,
            min_bpm,
            max_bpm,
            config.dbn_bpm_step,
        });
        beats.reserve(tmp.beat_frames.size());
        for (std::size_t frame : tmp.beat_frames) {
            beats.push_back(frame + w.first);
        }
        return beats;
    };

    std::vector<std::size_t> intro_beats;
    std::vector<std::size_t> outro_beats;
    if (anchors_separated) {
        intro_beats = decode_window_beats(intro);
        outro_beats = decode_window_beats(outro);
    }
    const bool intro_valid = intro_beats.size() >= 8;
    const bool outro_valid = outro_beats.size() >= 8;
    if (config.dbn_trace) {
        auto print_window = [&](const char* name,
                                const std::pair<std::size_t, std::size_t>& w,
                                std::size_t beat_count) {
            const double start_s = static_cast<double>(w.first) / fps;
            const double end_s = static_cast<double>(w.second) / fps;
            BEATIT_LOG_DEBUG("DBN tempo window " << name
                             << ": frames=" << w.first << "-" << w.second
                             << " (" << start_s << "s-" << end_s << "s)"
                             << " beats=" << beat_count);
        };
        print_window("intro", intro, intro_beats.size());
        print_window("outro", outro, outro_beats.size());
        BEATIT_LOG_DEBUG("DBN tempo anchors: gap_frames=" << anchor_gap_frames
                         << " min_gap_frames=" << min_anchor_gap_frames
                         << " separated=" << (anchors_separated ? "true" : "false"));
    }

    if (intro_valid && outro_valid && anchors_separated) {
        const auto [intro_bpm, intro_conf] = estimate_window_tempo(intro_beats, fps);
        const auto [outro_bpm, outro_conf] = estimate_window_tempo(outro_beats, fps);
        const auto intro_modes = expand_mode_candidates(intro_bpm, intro_conf, min_bpm, max_bpm);
        const auto outro_modes = expand_mode_candidates(outro_bpm, outro_conf, min_bpm, max_bpm);
        const auto [best_intro, best_outro] = match_mode_candidates(intro_modes, outro_modes);
        const double agree_rel = (best_intro.bpm > 0.0 && best_outro.bpm > 0.0)
            ? (std::abs(best_intro.bpm - best_outro.bpm) /
               (0.5 * (best_intro.bpm + best_outro.bpm)))
            : std::numeric_limits<double>::infinity();
        if (agree_rel <= 0.025) {
            const double conf_sum = best_intro.conf + best_outro.conf;
            if (conf_sum > 0.0) {
                bpm_from_global_fit =
                    ((best_intro.bpm * best_intro.conf) +
                     (best_outro.bpm * best_outro.conf)) /
                    conf_sum;
            }
        }
        if (config.dbn_trace) {
            BEATIT_LOG_DEBUG("DBN tempo anchors: intro_bpm=" << intro_bpm
                             << " intro_conf=" << intro_conf
                             << " outro_bpm=" << outro_bpm
                             << " outro_conf=" << outro_conf
                             << " mode_match_intro=" << best_intro.mode
                             << " mode_match_outro=" << best_outro.mode
                             << " agree_rel=" << agree_rel
                             << " fused_bpm=" << bpm_from_global_fit);
        }
    }

    return bpm_from_global_fit;
}

} // namespace beatit::detail
