#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "beatit/stream.h"
#include "coreml_test_config.h"
#include "window_alignment_test_utils.h"

namespace {

constexpr std::size_t kEdgeWindowBeats = 64;
constexpr std::size_t kAlternationWindowBeats = 24;
constexpr double kTargetBpm = 126.0;
constexpr double kMaxBpmError = 2.0;
constexpr std::size_t kMinBeatCount = 100;
constexpr std::size_t kMinDownbeatCount = 20;
constexpr double kMaxOffsetSlopeMsPerBeat = 0.04;
constexpr double kMaxStartEndDeltaMs = 45.0;
constexpr double kMaxStartEndDeltaBeats = 0.08;
constexpr double kMaxOddEvenMedianGapMs = 15.0;
constexpr double kMaxIntroMedianAbsOffsetMs = 25.0;
constexpr std::size_t kTempoEdgeIntervals = 64;
constexpr double kMaxTempoEdgeBpmDelta = 0.01;

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
    const std::filesystem::path audio_path = test_root / "training" / "vamp.wav";
    if (!std::filesystem::exists(audio_path)) {
        std::cerr << "SKIP: missing " << audio_path.string() << "\n";
        return 77;
    }

    beatit::CoreMLConfig config;
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
            config.verbose = true;
            config.dbn_trace = true;
            config.profile = true;
        }
    }
    if (const char* force_cpu = std::getenv("BEATIT_TEST_CPU_ONLY")) {
        if (force_cpu[0] != '\0' && force_cpu[0] != '0') {
            config.compute_units = beatit::CoreMLConfig::ComputeUnits::CPUOnly;
        }
    }

    std::vector<float> mono;
    double sample_rate = 0.0;
    std::string decode_error;
    if (!decode_audio_mono(audio_path.string(), &mono, &sample_rate, &decode_error)) {
        std::cerr << "Vamp alignment test failed: decode error: " << decode_error << "\n";
        return 1;
    }

    beatit::BeatitStream stream(sample_rate, config, true);
    double start_s = 0.0;
    double duration_s = 0.0;
    if (!stream.request_analysis_window(&start_s, &duration_s)) {
        std::cerr << "Vamp alignment test failed: request_analysis_window returned false.\n";
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
        std::cerr << "Vamp alignment test failed: too few beat events: "
                  << result.coreml_beat_events.size() << "\n";
        return 1;
    }
    if (result.coreml_beat_events.size() < kMinBeatCount) {
        std::cerr << "Vamp alignment test failed: beat event count "
                  << result.coreml_beat_events.size() << " < " << kMinBeatCount << ".\n";
        return 1;
    }
    if (!first_bar_is_complete_four_four(result)) {
        std::cerr << "Vamp alignment test failed: opening bar is not complete 4/4 "
                     "(expected first downbeats at beat indices 0 and 4).\n";
        return 1;
    }
    if (!bars_repeat_every_four_beats(result)) {
        std::cerr << "Vamp alignment test failed: bar markers are not consistently every 4 beats.\n";
        return 1;
    }
    if (!(result.estimated_bpm > 0.0)) {
        std::cerr << "Vamp alignment test failed: non-positive BPM.\n";
        return 1;
    }
    if (std::fabs(result.estimated_bpm - kTargetBpm) > kMaxBpmError) {
        std::cerr << "Vamp alignment test failed: estimated BPM "
                  << result.estimated_bpm << " outside [" << (kTargetBpm - kMaxBpmError)
                  << "," << (kTargetBpm + kMaxBpmError) << "].\n";
        return 1;
    }

    std::vector<unsigned long long> beat_frames;
    beat_frames.reserve(result.coreml_beat_events.size());
    for (const auto& event : result.coreml_beat_events) {
        beat_frames.push_back(event.frame);
    }
    if (result.coreml_downbeat_feature_frames.empty()) {
        std::cerr << "Vamp alignment test failed: missing downbeat feature frames.\n";
        return 1;
    }
    if (result.coreml_downbeat_feature_frames.size() < kMinDownbeatCount) {
        std::cerr << "Vamp alignment test failed: downbeat count "
                  << result.coreml_downbeat_feature_frames.size() << " < "
                  << kMinDownbeatCount << ".\n";
        return 1;
    }
    if (!result.coreml_beat_projected_sample_frames.empty() &&
        result.coreml_beat_projected_sample_frames.size() < kMinBeatCount) {
        std::cerr << "Vamp alignment test failed: projected beat count "
                  << result.coreml_beat_projected_sample_frames.size() << " < "
                  << kMinBeatCount << ".\n";
        return 1;
    }

    const std::vector<double> offsets_ms =
        compute_strong_peak_offsets_ms(beat_frames, mono, sample_rate, result.estimated_bpm);
    if (offsets_ms.size() < kEdgeWindowBeats) {
        std::cerr << "Vamp alignment test failed: too few offsets.\n";
        return 1;
    }

    std::vector<double> first(offsets_ms.begin(),
                              offsets_ms.begin() + static_cast<long>(kEdgeWindowBeats));
    std::vector<double> last(offsets_ms.end() - static_cast<long>(kEdgeWindowBeats),
                             offsets_ms.end());
    const double start_median_ms = median(first);
    std::vector<double> first_abs = first;
    for (double& v : first_abs) {
        v = std::fabs(v);
    }
    const double start_median_abs_ms = median(first_abs);
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

    std::cout << "Vamp alignment metrics: bpm=" << result.estimated_bpm
              << " beat_events=" << result.coreml_beat_events.size()
              << " downbeats=" << result.coreml_downbeat_feature_frames.size()
              << " projected_beats=" << result.coreml_beat_projected_sample_frames.size()
              << " start_median_ms=" << start_median_ms
              << " start_median_abs_ms=" << start_median_abs_ms
              << " end_median_ms=" << end_median_ms
              << " delta_ms=" << start_end_delta_ms
              << " delta_beats=" << start_end_delta_beats
              << " slope_ms_per_beat=" << slope_ms_per_beat
              << " odd_even_gap_ms=" << odd_even_gap_ms
              << " early_bpm=" << early_bpm
              << " late_bpm=" << late_bpm
              << " edge_bpm_delta=" << edge_bpm_delta
              << "\n";

    if (start_median_abs_ms > kMaxIntroMedianAbsOffsetMs) {
        std::cerr << "Vamp alignment test failed: intro median abs offset "
                  << start_median_abs_ms << "ms > " << kMaxIntroMedianAbsOffsetMs
                  << "ms\n";
        return 1;
    }
    if (std::fabs(slope_ms_per_beat) > kMaxOffsetSlopeMsPerBeat) {
        std::cerr << "Vamp alignment test failed: slope " << slope_ms_per_beat
                  << "ms/beat > " << kMaxOffsetSlopeMsPerBeat << "\n";
        return 1;
    }
    if (std::fabs(start_end_delta_ms) > kMaxStartEndDeltaMs) {
        std::cerr << "Vamp alignment test failed: start/end delta " << start_end_delta_ms
                  << "ms > " << kMaxStartEndDeltaMs << "ms\n";
        return 1;
    }
    if (start_end_delta_beats > kMaxStartEndDeltaBeats) {
        std::cerr << "Vamp alignment test failed: start/end delta "
                  << start_end_delta_beats << " beats > "
                  << kMaxStartEndDeltaBeats << " beats\n";
        return 1;
    }
    if (odd_even_gap_ms > kMaxOddEvenMedianGapMs) {
        std::cerr << "Vamp alignment test failed: odd/even median gap "
                  << odd_even_gap_ms << "ms > " << kMaxOddEvenMedianGapMs << "ms\n";
        return 1;
    }
    if (!(early_bpm > 0.0) || !(late_bpm > 0.0)) {
        std::cerr << "Vamp alignment test failed: invalid edge BPM estimate.\n";
        return 1;
    }
    if (edge_bpm_delta > kMaxTempoEdgeBpmDelta) {
        std::cerr << "Vamp alignment test failed: edge BPM delta " << edge_bpm_delta
                  << " > " << kMaxTempoEdgeBpmDelta << "\n";
        return 1;
    }

    return 0;
}
