//
//  eureka_window_alignment_test.mm
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "beatit/stream.h"
#include "coreml_test_config.h"
#include "window_alignment_test_utils.h"

namespace {

constexpr std::size_t kEdgeWindowBeats = 64;
constexpr std::size_t kAlternationWindowBeats = 24;
constexpr double kTargetBpm = 120.209;
constexpr double kMaxBpmError = 0.05;
constexpr std::size_t kExpectedBeatCount = 719;
constexpr std::size_t kExpectedDownbeatCount = 30;
constexpr unsigned long long kExpectedFirstDownbeatFeatureFrame = 0ULL;
constexpr unsigned long long kExpectedFirstDownbeatSampleFrame = 241ULL;
constexpr unsigned long long kFirstDownbeatFeatureFrameTolerance = 1ULL;
constexpr double kFirstDownbeatSampleToleranceMs = 10.0;
constexpr double kMaxOffsetSlopeMsPerBeat = 0.04;
constexpr double kMaxUnwrappedSlopeMsPerBeat = 0.06;
constexpr double kMaxStartEndDeltaMs = 65.0;
constexpr double kMaxStartEndDeltaBeats = 0.12;
constexpr double kMaxUnwrappedStartEndDeltaBeats = 0.13;
constexpr double kMaxOddEvenMedianGapMs = 11.0;
constexpr double kMaxIntroMedianAbsOffsetMs = 52.0;
constexpr std::size_t kTempoEdgeIntervals = 64;
constexpr double kMaxTempoEdgeBpmDelta = 0.0005;
constexpr std::size_t kDriftProbeCount = 24;
constexpr std::size_t kEventProbeCount = 16;
constexpr double kOneBeatSignatureMinBeats = 0.70;
constexpr double kOneBeatSignatureMaxBeats = 1.30;

using namespace beatit::tests::window_alignment;

} // namespace

int main() {
    std::string source_model_path = beatit::tests::resolve_beatthis_coreml_model_path();
    if (source_model_path.empty()) {
        std::cerr << "SKIP: BeatThis CoreML model missing (set BEATIT_COREML_MODEL_PATH).\n";
        return 77;
    }

    std::string model_error;
    std::string model_path = compile_model_if_needed(source_model_path, &model_error);
    if (model_path.empty()) {
        std::cerr << "SKIP: Failed to prepare CoreML model: " << model_error << "\n";
        return 77;
    }

    std::filesystem::path test_root = std::filesystem::current_path();
#if defined(BEATIT_TEST_DATA_DIR)
    test_root = BEATIT_TEST_DATA_DIR;
#endif
    const std::filesystem::path audio_path = test_root / "training" / "eureka.wav";
    if (!std::filesystem::exists(audio_path)) {
        std::cerr << "SKIP: missing " << audio_path.string() << "\n";
        return 77;
    }

    beatit::BeatitConfig config;
    beatit::tests::apply_beatthis_coreml_test_config(config);
    config.model_path = model_path;
    config.use_dbn = true;
    config.max_analysis_seconds = 60.0;
    config.dbn_window_start_seconds = 0.0;
#if defined(BEATIT_TEST_SPARSE_PROBE_MODE)
    config.sparse_probe_mode = true;
#endif
    if (const char* trace = std::getenv("BEATIT_WINDOW_TRACE")) {
        if (trace[0] != '\0' && trace[0] != '0') {
            config.log_verbosity = beatit::LogVerbosity::Debug;
            config.dbn_trace = true;
            config.profile = true;
        }
    }
    if (const char* force_cpu = std::getenv("BEATIT_TEST_CPU_ONLY")) {
        if (force_cpu[0] != '\0' && force_cpu[0] != '0') {
            config.compute_units = beatit::BeatitConfig::ComputeUnits::CPUOnly;
        }
    }

    std::vector<float> mono;
    double sample_rate = 0.0;
    std::string decode_error;
    if (!decode_audio_mono(audio_path.string(), &mono, &sample_rate, &decode_error)) {
        std::cerr << "Eureka alignment test failed: decode error: " << decode_error << "\n";
        return 1;
    }

    beatit::BeatitStream stream(sample_rate, config, true);
    double start_s = 0.0;
    double duration_s = 0.0;
    if (!stream.request_analysis_window(&start_s, &duration_s)) {
        std::cerr << "Eureka alignment test failed: request_analysis_window returned false.\n";
        return 1;
    }

    const double total_duration_s = static_cast<double>(mono.size()) / sample_rate;
    auto provider =
        [&](double start_seconds, double duration_seconds, std::vector<float>* out_samples) -> std::size_t {
            if (!out_samples) {
                return 0;
            }
            out_samples->clear();
            if (sample_rate <= 0.0 || mono.empty()) {
                return 0;
            }
            const std::size_t begin = static_cast<std::size_t>(
                std::llround(std::max(0.0, start_seconds) * sample_rate));
            const std::size_t count = static_cast<std::size_t>(
                std::llround(std::max(0.0, duration_seconds) * sample_rate));
            const std::size_t end = std::min(mono.size(), begin + count);
            if (begin >= end) {
                return 0;
            }
            out_samples->assign(mono.begin() + static_cast<long>(begin),
                                mono.begin() + static_cast<long>(end));
            return out_samples->size();
        };

    beatit::AnalysisResult result =
        stream.analyze_window(start_s, duration_s, total_duration_s, provider);

    if (result.coreml_beat_events.size() < kEdgeWindowBeats) {
        std::cerr << "Eureka alignment test failed: too few beat events: "
                  << result.coreml_beat_events.size() << "\n";
        return 1;
    }
    if (result.coreml_beat_events.size() != kExpectedBeatCount) {
        std::cerr << "Eureka alignment test failed: beat event count "
                  << result.coreml_beat_events.size() << " != " << kExpectedBeatCount << ".\n";
        return 1;
    }
    const bool opening_bar_complete = first_bar_is_complete_four_four(result);
    if (!bars_repeat_every_four_beats(result)) {
        std::cerr << "Eureka alignment test failed: bar markers are not consistently every 4 beats.\n";
        return 1;
    }
    if (!(result.estimated_bpm > 0.0)) {
        std::cerr << "Eureka alignment test failed: non-positive BPM.\n";
        return 1;
    }
    if (std::fabs(result.estimated_bpm - kTargetBpm) > kMaxBpmError) {
        std::cerr << "Eureka alignment test failed: estimated BPM "
                  << result.estimated_bpm << " outside ["
                  << (kTargetBpm - kMaxBpmError) << ","
                  << (kTargetBpm + kMaxBpmError) << "].\n";
        return 1;
    }

    std::vector<unsigned long long> beat_frames;
    std::vector<unsigned long long> beat_styles;
    beat_frames.reserve(result.coreml_beat_events.size());
    beat_styles.reserve(result.coreml_beat_events.size());
    for (const auto& event : result.coreml_beat_events) {
        beat_frames.push_back(event.frame);
        beat_styles.push_back(event.style);
    }
    if (result.coreml_downbeat_feature_frames.empty()) {
        std::cerr << "Eureka alignment test failed: missing downbeat feature frames.\n";
        return 1;
    }
    if (result.coreml_downbeat_feature_frames.size() != kExpectedDownbeatCount) {
        std::cerr << "Eureka alignment test failed: downbeat count "
                  << result.coreml_downbeat_feature_frames.size() << " != "
                  << kExpectedDownbeatCount << ".\n";
        return 1;
    }
    unsigned long long first_downbeat_feature_ff = 0;
    if (!first_downbeat_feature_frame(result, &first_downbeat_feature_ff)) {
        std::cerr << "Eureka alignment test failed: missing first downbeat feature frame.\n";
        return 1;
    }
    const auto downbeat_feature_delta =
        (first_downbeat_feature_ff > kExpectedFirstDownbeatFeatureFrame)
            ? (first_downbeat_feature_ff - kExpectedFirstDownbeatFeatureFrame)
            : (kExpectedFirstDownbeatFeatureFrame - first_downbeat_feature_ff);
    if (downbeat_feature_delta > kFirstDownbeatFeatureFrameTolerance) {
        const auto lower_feature_frame =
            (kExpectedFirstDownbeatFeatureFrame > kFirstDownbeatFeatureFrameTolerance)
                ? (kExpectedFirstDownbeatFeatureFrame - kFirstDownbeatFeatureFrameTolerance)
                : 0ULL;
        std::cerr << "Eureka alignment test failed: first downbeat feature frame "
                  << first_downbeat_feature_ff << " outside ["
                  << lower_feature_frame << ","
                  << (kExpectedFirstDownbeatFeatureFrame + kFirstDownbeatFeatureFrameTolerance)
                  << "].\n";
        return 1;
    }
    unsigned long long first_downbeat_sample_sf = 0;
    if (!first_downbeat_sample_frame(result, &first_downbeat_sample_sf)) {
        std::cerr << "Eureka alignment test failed: missing first downbeat sample frame.\n";
        return 1;
    }
    const auto downbeat_sample_tolerance_frames = static_cast<unsigned long long>(
        std::llround((kFirstDownbeatSampleToleranceMs / 1000.0) * sample_rate));
    const auto downbeat_sample_delta =
        (first_downbeat_sample_sf > kExpectedFirstDownbeatSampleFrame)
            ? (first_downbeat_sample_sf - kExpectedFirstDownbeatSampleFrame)
            : (kExpectedFirstDownbeatSampleFrame - first_downbeat_sample_sf);
    if (downbeat_sample_delta > downbeat_sample_tolerance_frames) {
        const auto lower_sample_frame =
            (kExpectedFirstDownbeatSampleFrame > downbeat_sample_tolerance_frames)
                ? (kExpectedFirstDownbeatSampleFrame - downbeat_sample_tolerance_frames)
                : 0ULL;
        std::cerr << "Eureka alignment test failed: first downbeat sample frame "
                  << first_downbeat_sample_sf << " outside ["
                  << lower_sample_frame << ","
                  << (kExpectedFirstDownbeatSampleFrame + downbeat_sample_tolerance_frames)
                  << "] (±" << kFirstDownbeatSampleToleranceMs << "ms).\n";
        return 1;
    }
    if (!result.coreml_beat_projected_sample_frames.empty() &&
        result.coreml_beat_projected_sample_frames.size() != kExpectedBeatCount) {
        std::cerr << "Eureka alignment test failed: projected beat count "
                  << result.coreml_beat_projected_sample_frames.size() << " != "
                  << kExpectedBeatCount << ".\n";
        return 1;
    }

    const std::vector<double> offsets_ms =
        compute_strong_peak_offsets_ms(beat_frames, mono, sample_rate, result.estimated_bpm);
    if (offsets_ms.size() < kEdgeWindowBeats) {
        std::cerr << "Eureka alignment test failed: too few offsets.\n";
        return 1;
    }

    std::vector<double> first(offsets_ms.begin(),
                              offsets_ms.begin() + static_cast<long>(kEdgeWindowBeats));
    const std::size_t middle_start =
        (offsets_ms.size() - kEdgeWindowBeats) / 2;
    const std::size_t between_start = middle_start / 2;
    std::vector<double> between(offsets_ms.begin() + static_cast<long>(between_start),
                                offsets_ms.begin() + static_cast<long>(between_start + kEdgeWindowBeats));
    std::vector<double> middle(offsets_ms.begin() + static_cast<long>(middle_start),
                               offsets_ms.begin() + static_cast<long>(middle_start + kEdgeWindowBeats));
    std::vector<double> last(offsets_ms.end() - static_cast<long>(kEdgeWindowBeats),
                             offsets_ms.end());
    const double start_median_ms = median(first);
    std::vector<double> first_abs = first;
    for (double& v : first_abs) {
        v = std::fabs(v);
    }
    const double start_median_abs_ms = median(first_abs);
    const double between_median_ms = median(between);
    std::vector<double> between_abs = between;
    for (double& v : between_abs) {
        v = std::fabs(v);
    }
    const double between_median_abs_ms = median(between_abs);
    const double middle_median_ms = median(middle);
    std::vector<double> middle_abs = middle;
    for (double& v : middle_abs) {
        v = std::fabs(v);
    }
    const double middle_median_abs_ms = median(middle_abs);
    const double end_median_ms = median(last);
    const double start_end_delta_ms = end_median_ms - start_median_ms;
    const double ms_per_beat = result.estimated_bpm > 0.0
                                   ? (60000.0 / result.estimated_bpm)
                                   : 0.0;
    const double start_end_delta_beats =
        ms_per_beat > 0.0 ? (std::fabs(start_end_delta_ms) / ms_per_beat) : 0.0;
    const double beat_period_ms =
        (result.estimated_bpm > 0.0f) ? (60000.0 / result.estimated_bpm) : 500.0;
    const double slope_ms_per_beat = robust_linear_slope(offsets_ms, beat_period_ms);
    const std::vector<double> unwrapped_offsets_ms =
        unwrap_periodic_offsets_ms(offsets_ms, beat_period_ms);
    const double unwrapped_slope_ms_per_beat =
        robust_linear_slope(unwrapped_offsets_ms, beat_period_ms);

    const std::size_t alt_n = std::min<std::size_t>(kAlternationWindowBeats, offsets_ms.size());
    std::vector<double> odd;
    std::vector<double> even;
    odd.reserve(alt_n / 2);
    even.reserve((alt_n + 1) / 2);
    for (std::size_t i = 0; i < alt_n; ++i) {
        if ((i % 2) == 0) {
            even.push_back(offsets_ms[i]);
        } else {
            odd.push_back(offsets_ms[i]);
        }
    }
    const double odd_even_gap_ms = std::fabs(median(even) - median(odd));
    const double early_interval_s =
        median_interval_seconds(beat_frames, sample_rate, kTempoEdgeIntervals, false);
    const double late_interval_s =
        median_interval_seconds(beat_frames, sample_rate, kTempoEdgeIntervals, true);
    const double early_bpm = early_interval_s > 0.0 ? (60.0 / early_interval_s) : 0.0;
    const double late_bpm = late_interval_s > 0.0 ? (60.0 / late_interval_s) : 0.0;
    const double edge_bpm_delta = std::fabs(early_bpm - late_bpm);
    std::vector<double> first_unwrapped(unwrapped_offsets_ms.begin(),
                                        unwrapped_offsets_ms.begin() + static_cast<long>(kEdgeWindowBeats));
    std::vector<double> last_unwrapped(unwrapped_offsets_ms.end() - static_cast<long>(kEdgeWindowBeats),
                                       unwrapped_offsets_ms.end());
    const double unwrapped_start_median_ms = median(first_unwrapped);
    const double unwrapped_end_median_ms = median(last_unwrapped);
    const double unwrapped_start_end_delta_ms =
        unwrapped_end_median_ms - unwrapped_start_median_ms;
    const double unwrapped_start_end_delta_beats =
        ms_per_beat > 0.0 ? (std::fabs(unwrapped_start_end_delta_ms) / ms_per_beat) : 0.0;
    const double phase_hot_threshold_ms = std::max(35.0, beat_period_ms * 0.22);
    const bool edges_calm =
        std::fabs(start_median_ms) <= (phase_hot_threshold_ms * 0.5) &&
        std::fabs(end_median_ms) <= (phase_hot_threshold_ms * 0.5);
    const bool between_hot = between_median_abs_ms > phase_hot_threshold_ms;
    const bool middle_hot = middle_median_abs_ms > phase_hot_threshold_ms;
    const bool one_beat_linear_signature =
        unwrapped_start_end_delta_beats > kOneBeatSignatureMinBeats &&
        unwrapped_start_end_delta_beats < kOneBeatSignatureMaxBeats;
    const bool wrapped_middle_signature = edges_calm && (between_hot || middle_hot);

    std::cout << "Eureka drift probe: beats=" << result.coreml_beat_events.size()
              << " bpm=" << result.estimated_bpm
              << " start_med=" << start_median_ms << "ms"
              << " between_med=" << between_median_ms << "ms"
              << " middle_med=" << middle_median_ms << "ms"
              << " end_med=" << end_median_ms << "ms"
              << " delta=" << start_end_delta_ms << "ms"
              << " slope=" << slope_ms_per_beat << "ms/beat"
              << " unwrapped_start_med=" << unwrapped_start_median_ms << "ms"
              << " unwrapped_end_med=" << unwrapped_end_median_ms << "ms"
              << " unwrapped_delta=" << unwrapped_start_end_delta_ms << "ms"
              << " unwrapped_slope=" << unwrapped_slope_ms_per_beat << "ms/beat"
              << " probes=" << offsets_ms.size()
              << " first24_ms=" << format_double_slice(offsets_ms, kDriftProbeCount, false, 2)
              << " between24_ms=" << format_double_slice(between, kDriftProbeCount, false, 2)
              << " middle24_ms=" << format_double_slice(middle, kDriftProbeCount, false, 2)
              << " last24_ms=" << format_double_slice(offsets_ms, kDriftProbeCount, true, 2)
              << "\n";

    if (const char* dump = std::getenv("BEATIT_EUREKA_DUMP_EVENTS")) {
        if (dump[0] != '\0' && dump[0] != '0') {
            std::cout << "Eureka event probe: first_frames="
                      << format_slice(beat_frames, kEventProbeCount, false)
                      << " first_styles="
                      << format_slice(beat_styles, kEventProbeCount, false)
                      << " last_frames="
                      << format_slice(beat_frames, kEventProbeCount, true)
                      << " last_styles="
                      << format_slice(beat_styles, kEventProbeCount, true)
                      << "\n";
            std::cout << "Eureka downbeat probe: first="
                      << format_slice(result.coreml_downbeat_feature_frames, kEventProbeCount, false)
                      << " last="
                      << format_slice(result.coreml_downbeat_feature_frames, kEventProbeCount, true)
                      << "\n";
            unsigned long long first_downbeat_feature_ff = 0;
            unsigned long long first_downbeat_sample_sf = 0;
            const bool has_feature = first_downbeat_feature_frame(result, &first_downbeat_feature_ff);
            const bool has_sample = first_downbeat_sample_frame(result, &first_downbeat_sample_sf);
            std::cout << "Eureka downbeat timing probe: feature_frame="
                      << (has_feature ? std::to_string(first_downbeat_feature_ff) : std::string("none"))
                      << " sample_frame="
                      << (has_sample ? std::to_string(first_downbeat_sample_sf) : std::string("none"))
                      << "\n";
        }
    }

    std::cout << "Eureka alignment metrics: bpm=" << result.estimated_bpm
              << " beat_events=" << result.coreml_beat_events.size()
              << " downbeats=" << result.coreml_downbeat_feature_frames.size()
              << " projected_beats=" << result.coreml_beat_projected_sample_frames.size()
              << " start_median_ms=" << start_median_ms
              << " start_median_abs_ms=" << start_median_abs_ms
              << " between_median_ms=" << between_median_ms
              << " between_median_abs_ms=" << between_median_abs_ms
              << " middle_median_ms=" << middle_median_ms
              << " middle_median_abs_ms=" << middle_median_abs_ms
              << " end_median_ms=" << end_median_ms
              << " delta_ms=" << start_end_delta_ms
              << " delta_beats=" << start_end_delta_beats
              << " slope_ms_per_beat=" << slope_ms_per_beat
              << " unwrapped_delta_ms=" << unwrapped_start_end_delta_ms
              << " unwrapped_delta_beats=" << unwrapped_start_end_delta_beats
              << " unwrapped_slope_ms_per_beat=" << unwrapped_slope_ms_per_beat
              << " odd_even_gap_ms=" << odd_even_gap_ms
              << " early_bpm=" << early_bpm
              << " late_bpm=" << late_bpm
              << " edge_bpm_delta=" << edge_bpm_delta
              << " one_beat_linear_signature=" << (one_beat_linear_signature ? 1 : 0)
              << " wrapped_middle_signature=" << (wrapped_middle_signature ? 1 : 0)
              << "\n";

    if (!opening_bar_complete) {
        std::cerr << "Eureka alignment test failed: opening bar is not complete 4/4 "
                     "(expected first downbeats at beat indices 0 and 4).\n";
        return 1;
    }
    if (start_median_abs_ms > kMaxIntroMedianAbsOffsetMs) {
        std::cerr << "Eureka alignment test failed: intro median abs offset "
                  << start_median_abs_ms << "ms > " << kMaxIntroMedianAbsOffsetMs
                  << "ms\n";
        return 1;
    }
    if (std::fabs(slope_ms_per_beat) > kMaxOffsetSlopeMsPerBeat) {
        std::cerr << "Eureka alignment test failed: slope " << slope_ms_per_beat
                  << "ms/beat > " << kMaxOffsetSlopeMsPerBeat << "\n";
        return 1;
    }
    if (std::fabs(unwrapped_slope_ms_per_beat) > kMaxUnwrappedSlopeMsPerBeat) {
        std::cerr << "Eureka alignment test failed: unwrapped slope "
                  << unwrapped_slope_ms_per_beat << "ms/beat > "
                  << kMaxUnwrappedSlopeMsPerBeat << "\n";
        return 1;
    }
    if (std::fabs(start_end_delta_ms) > kMaxStartEndDeltaMs) {
        std::cerr << "Eureka alignment test failed: start/end delta " << start_end_delta_ms
                  << "ms > " << kMaxStartEndDeltaMs << "ms\n";
        return 1;
    }
    if (start_end_delta_beats > kMaxStartEndDeltaBeats) {
        std::cerr << "Eureka alignment test failed: start/end delta "
                  << start_end_delta_beats << " beats > "
                  << kMaxStartEndDeltaBeats << " beats\n";
        return 1;
    }
    if (unwrapped_start_end_delta_beats > kMaxUnwrappedStartEndDeltaBeats) {
        std::cerr << "Eureka alignment test failed: unwrapped start/end delta "
                  << unwrapped_start_end_delta_beats << " beats > "
                  << kMaxUnwrappedStartEndDeltaBeats << " beats\n";
        return 1;
    }
    if (odd_even_gap_ms > kMaxOddEvenMedianGapMs) {
        std::cerr << "Eureka alignment test failed: odd/even median gap "
                  << odd_even_gap_ms << "ms > " << kMaxOddEvenMedianGapMs << "ms\n";
        return 1;
    }
    if (!(early_bpm > 0.0) || !(late_bpm > 0.0)) {
        std::cerr << "Eureka alignment test failed: invalid edge BPM estimate.\n";
        return 1;
    }
    if (edge_bpm_delta > kMaxTempoEdgeBpmDelta) {
        std::cerr << "Eureka alignment test failed: edge BPM delta " << edge_bpm_delta
                  << " > " << kMaxTempoEdgeBpmDelta << "\n";
        return 1;
    }
    if (one_beat_linear_signature) {
        std::cerr << "Eureka alignment test failed: one-beat linear signature detected.\n";
        return 1;
    }
    if (wrapped_middle_signature) {
        std::cerr << "Eureka alignment test failed: wrapped middle signature detected.\n";
        return 1;
    }

    return 0;
}
