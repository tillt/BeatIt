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

namespace {

constexpr std::size_t kEdgeWindowBeats = 64;
constexpr std::size_t kAlternationWindowBeats = 24;
constexpr double kTargetBpm = 126.0;
constexpr double kMaxBpmError = 2.0;
constexpr std::size_t kMinBeatCount = 100;
constexpr std::size_t kMinDownbeatCount = 20;
constexpr double kMaxOffsetSlopeMsPerBeat = 0.08;
constexpr double kMaxStartEndDeltaMs = 90.0;
constexpr double kMaxStartEndDeltaBeats = 0.16;
constexpr double kMaxOddEvenMedianGapMs = 40.0;
constexpr double kMaxIntroMedianAbsOffsetMs = 85.0;
constexpr std::size_t kTempoEdgeIntervals = 64;
constexpr double kMaxTempoEdgeBpmDelta = 0.05;

std::string compile_model_if_needed(const std::string& path, std::string* error) {
    NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:ns_path];
    if (!url) {
        if (error) {
            *error = "Failed to create model URL.";
        }
        return {};
    }

    NSString* ext = url.pathExtension.lowercaseString;
    if (![ext isEqualToString:@"mlpackage"] && ![ext isEqualToString:@"mlmodel"]) {
        return path;
    }

    std::string tmp_dir;
    try {
        tmp_dir = (std::filesystem::current_path() / "coreml_tmp").string();
        std::filesystem::create_directories(tmp_dir);
        setenv("TMPDIR", tmp_dir.c_str(), 1);
    } catch (const std::exception&) {
        if (error) {
            *error = "Failed to prepare temporary directory for CoreML compile.";
        }
        return {};
    }

    NSError* compile_error = nil;
    NSURL* compiled_url = [MLModel compileModelAtURL:url error:&compile_error];
    if (!compiled_url || compile_error) {
        if (error) {
            std::string message = "Failed to compile CoreML model.";
            if (compile_error) {
                message += " ";
                message += compile_error.localizedDescription.UTF8String;
            }
            *error = message;
        }
        return {};
    }

    return compiled_url.path.UTF8String ? compiled_url.path.UTF8String : std::string();
}

bool decode_audio_mono(const std::string& path,
                       std::vector<float>* mono_out,
                       double* sample_rate_out,
                       std::string* error) {
    if (!mono_out) {
        if (error) {
            *error = "Output pointer is null.";
        }
        return false;
    }

    @autoreleasepool {
        NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:ns_path];
        NSError* open_error = nil;
        AVAudioFile* file = [[AVAudioFile alloc] initForReading:url error:&open_error];
        if (!file || open_error) {
            if (error) {
                *error = "AVAudioFile failed to open.";
            }
            return false;
        }

        AVAudioFormat* input_format = file.processingFormat;
        const double sample_rate = input_format.sampleRate > 0.0 ? input_format.sampleRate : 44100.0;
        const AVAudioChannelCount channels = input_format.channelCount > 0 ? input_format.channelCount : 1;
        AVAudioFormat* render_format =
            [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                              sampleRate:sample_rate
                                                channels:channels
                                             interleaved:NO];

        AVAudioConverter* converter = [[AVAudioConverter alloc] initFromFormat:input_format
                                                                       toFormat:render_format];
        converter.sampleRateConverterQuality = AVAudioQualityMax;
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering;

        const AVAudioFrameCount chunk_frames = 4096;
        AVAudioPCMBuffer* input_buffer =
            [[AVAudioPCMBuffer alloc] initWithPCMFormat:input_format frameCapacity:chunk_frames];
        AVAudioPCMBuffer* output_buffer =
            [[AVAudioPCMBuffer alloc] initWithPCMFormat:render_format frameCapacity:chunk_frames];

        mono_out->clear();
        __block BOOL done_reading = NO;
        __block NSError* read_error = nil;
        while (!done_reading) {
            input_buffer.frameLength = 0;
            output_buffer.frameLength = 0;

            AVAudioConverterInputBlock input_block =
                ^AVAudioBuffer* _Nullable(AVAudioPacketCount, AVAudioConverterInputStatus* out_status) {
                    NSError* local_error = nil;
                    if (![file readIntoBuffer:input_buffer error:&local_error] ||
                        input_buffer.frameLength == 0) {
                        if (local_error) {
                            read_error = local_error;
                        }
                        *out_status = AVAudioConverterInputStatus_EndOfStream;
                        done_reading = YES;
                        return nil;
                    }
                    if (local_error) {
                        read_error = local_error;
                        done_reading = YES;
                        *out_status = AVAudioConverterInputStatus_EndOfStream;
                        return nil;
                    }
                    *out_status = AVAudioConverterInputStatus_HaveData;
                    return input_buffer;
                };

            NSError* convert_error = nil;
            AVAudioConverterOutputStatus status =
                [converter convertToBuffer:output_buffer
                                     error:&convert_error
                       withInputFromBlock:input_block];
            if (status == AVAudioConverterOutputStatus_Error || convert_error) {
                if (error) {
                    std::string message = "AVAudioConverter failed.";
                    if (convert_error) {
                        message += " ";
                        message += convert_error.localizedDescription.UTF8String;
                    }
                    *error = message;
                }
                return false;
            }
            if (status == AVAudioConverterOutputStatus_EndOfStream) {
                done_reading = YES;
            }
            if (status != AVAudioConverterOutputStatus_HaveData ||
                output_buffer.frameLength == 0) {
                continue;
            }

            const AVAudioFrameCount frames = output_buffer.frameLength;
            const std::size_t base = mono_out->size();
            mono_out->resize(base + frames, 0.0f);
            for (AVAudioChannelCount ch = 0; ch < channels; ++ch) {
                const float* channel_data = output_buffer.floatChannelData[ch];
                for (AVAudioFrameCount i = 0; i < frames; ++i) {
                    (*mono_out)[base + i] += channel_data[i];
                }
            }
            const float scale = 1.0f / static_cast<float>(channels);
            for (AVAudioFrameCount i = 0; i < frames; ++i) {
                (*mono_out)[base + i] *= scale;
            }
        }

        if (read_error) {
            if (error) {
                *error = read_error.localizedDescription.UTF8String;
            }
            return false;
        }

        if (sample_rate_out) {
            *sample_rate_out = sample_rate;
        }
    }

    return true;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    auto mid = values.begin() + static_cast<long>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

double linear_slope(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double n = static_cast<double>(values.size());
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double x = static_cast<double>(i);
        const double y = values[i];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double den = n * sxx - sx * sx;
    if (std::fabs(den) < 1e-12) {
        return 0.0;
    }
    return (n * sxy - sx * sy) / den;
}

bool is_bar_event(const beatit::BeatEvent& event) {
    return (static_cast<unsigned long long>(event.style) &
            static_cast<unsigned long long>(beatit::BeatEventStyleBar)) != 0ULL;
}

bool first_bar_is_complete_four_four(const beatit::AnalysisResult& result) {
    std::vector<std::size_t> bar_indices;
    bar_indices.reserve(result.coreml_beat_events.size() / 4);
    for (std::size_t i = 0; i < result.coreml_beat_events.size(); ++i) {
        if (is_bar_event(result.coreml_beat_events[i])) {
            bar_indices.push_back(i);
        }
    }
    if (bar_indices.size() < 2) {
        return false;
    }
    return bar_indices[0] == 0 && bar_indices[1] == 4;
}

bool bars_repeat_every_four_beats(const beatit::AnalysisResult& result) {
    std::vector<std::size_t> bar_indices;
    bar_indices.reserve(result.coreml_beat_events.size() / 4);
    for (std::size_t i = 0; i < result.coreml_beat_events.size(); ++i) {
        if (is_bar_event(result.coreml_beat_events[i])) {
            bar_indices.push_back(i);
        }
    }
    if (bar_indices.size() < 2) {
        return false;
    }
    for (std::size_t i = 1; i < bar_indices.size(); ++i) {
        if ((bar_indices[i] - bar_indices[i - 1]) != 4) {
            return false;
        }
    }
    return true;
}

double median_interval_seconds(const std::vector<unsigned long long>& beat_frames,
                               double sample_rate,
                               std::size_t interval_count,
                               bool from_end) {
    if (beat_frames.size() < 3 || sample_rate <= 0.0 || interval_count == 0) {
        return 0.0;
    }
    std::vector<double> intervals;
    intervals.reserve(beat_frames.size() - 1);
    for (std::size_t i = 1; i < beat_frames.size(); ++i) {
        if (beat_frames[i] > beat_frames[i - 1]) {
            intervals.push_back(
                static_cast<double>(beat_frames[i] - beat_frames[i - 1]) / sample_rate);
        }
    }
    if (intervals.empty()) {
        return 0.0;
    }

    const std::size_t n = std::min(interval_count, intervals.size());
    std::vector<double> slice;
    if (from_end) {
        slice.assign(intervals.end() - static_cast<long>(n), intervals.end());
    } else {
        slice.assign(intervals.begin(), intervals.begin() + static_cast<long>(n));
    }
    return median(slice);
}

std::vector<double> compute_strong_peak_offsets_ms(const std::vector<unsigned long long>& beat_frames,
                                                   const std::vector<float>& mono,
                                                   double sample_rate,
                                                   double bpm) {
    std::vector<double> offsets;
    if (beat_frames.empty() || mono.empty() || sample_rate <= 0.0 || bpm <= 0.0) {
        return offsets;
    }

    const std::size_t radius = static_cast<std::size_t>(
        std::llround(sample_rate * (60.0 / bpm) * 0.6));
    offsets.reserve(beat_frames.size());

    for (unsigned long long frame_ull : beat_frames) {
        const std::size_t frame =
            static_cast<std::size_t>(std::min<unsigned long long>(frame_ull, mono.size() - 1));
        const std::size_t start = frame > radius ? frame - radius : 0;
        const std::size_t end = std::min(mono.size() - 1, frame + radius);
        if (end <= start + 2) {
            offsets.push_back(0.0);
            continue;
        }

        float window_max = 0.0f;
        for (std::size_t i = start; i <= end; ++i) {
            window_max = std::max(window_max, std::fabs(mono[i]));
        }
        const float threshold = window_max * 0.6f;

        std::size_t best_peak = frame;
        float best_value = 0.0f;
        for (std::size_t i = start + 1; i < end; ++i) {
            const float left = std::fabs(mono[i - 1]);
            const float value = std::fabs(mono[i]);
            const float right = std::fabs(mono[i + 1]);
            if (value < threshold) {
                continue;
            }
            if (value >= left && value > right && value > best_value) {
                best_value = value;
                best_peak = i;
            }
        }

        if (best_value <= 0.0f) {
            float max_value = 0.0f;
            for (std::size_t i = start; i <= end; ++i) {
                const float value = std::fabs(mono[i]);
                if (value > max_value) {
                    max_value = value;
                    best_peak = i;
                }
            }
        }

        const double delta_frames = static_cast<double>(static_cast<long long>(best_peak) -
                                                        static_cast<long long>(frame));
        offsets.push_back((delta_frames * 1000.0) / sample_rate);
    }
    return offsets;
}

}  // namespace

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
    const std::filesystem::path audio_path = test_root / "training" / "enigma.wav";
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
        std::cerr << "Enigma alignment test failed: decode error: " << decode_error << "\n";
        return 1;
    }

    beatit::BeatitStream stream(sample_rate, config, true);
    double start_s = 0.0;
    double duration_s = 0.0;
    if (!stream.request_analysis_window(&start_s, &duration_s)) {
        std::cerr << "Enigma alignment test failed: request_analysis_window returned false.\n";
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
        std::cerr << "Enigma alignment test failed: too few beat events: "
                  << result.coreml_beat_events.size() << "\n";
        return 1;
    }
    if (result.coreml_beat_events.size() < kMinBeatCount) {
        std::cerr << "Enigma alignment test failed: beat event count "
                  << result.coreml_beat_events.size() << " < " << kMinBeatCount << ".\n";
        return 1;
    }
    if (!first_bar_is_complete_four_four(result)) {
        std::cerr << "Enigma alignment test failed: opening bar is not complete 4/4 "
                     "(expected first downbeats at beat indices 0 and 4).\n";
        return 1;
    }
    if (!bars_repeat_every_four_beats(result)) {
        std::cerr << "Enigma alignment test failed: bar markers are not consistently every 4 beats.\n";
        return 1;
    }
    if (!(result.estimated_bpm > 0.0)) {
        std::cerr << "Enigma alignment test failed: non-positive BPM.\n";
        return 1;
    }
    if (std::fabs(result.estimated_bpm - kTargetBpm) > kMaxBpmError) {
        std::cerr << "Enigma alignment test failed: estimated BPM "
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
        std::cerr << "Enigma alignment test failed: missing downbeat feature frames.\n";
        return 1;
    }
    if (result.coreml_downbeat_feature_frames.size() < kMinDownbeatCount) {
        std::cerr << "Enigma alignment test failed: downbeat count "
                  << result.coreml_downbeat_feature_frames.size() << " < "
                  << kMinDownbeatCount << ".\n";
        return 1;
    }
    if (!result.coreml_beat_projected_sample_frames.empty() &&
        result.coreml_beat_projected_sample_frames.size() < kMinBeatCount) {
        std::cerr << "Enigma alignment test failed: projected beat count "
                  << result.coreml_beat_projected_sample_frames.size() << " < "
                  << kMinBeatCount << ".\n";
        return 1;
    }

    const std::vector<double> offsets_ms =
        compute_strong_peak_offsets_ms(beat_frames, mono, sample_rate, result.estimated_bpm);
    if (offsets_ms.size() < kEdgeWindowBeats) {
        std::cerr << "Enigma alignment test failed: too few offsets.\n";
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
    const double slope_ms_per_beat = linear_slope(offsets_ms);

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

    std::cout << "Enigma alignment metrics: bpm=" << result.estimated_bpm
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
        std::cerr << "Enigma alignment test failed: intro median abs offset "
                  << start_median_abs_ms << "ms > " << kMaxIntroMedianAbsOffsetMs
                  << "ms\n";
        return 1;
    }
    if (std::fabs(slope_ms_per_beat) > kMaxOffsetSlopeMsPerBeat) {
        std::cerr << "Enigma alignment test failed: slope " << slope_ms_per_beat
                  << "ms/beat > " << kMaxOffsetSlopeMsPerBeat << "\n";
        return 1;
    }
    if (std::fabs(start_end_delta_ms) > kMaxStartEndDeltaMs) {
        std::cerr << "Enigma alignment test failed: start/end delta " << start_end_delta_ms
                  << "ms > " << kMaxStartEndDeltaMs << "ms\n";
        return 1;
    }
    if (start_end_delta_beats > kMaxStartEndDeltaBeats) {
        std::cerr << "Enigma alignment test failed: start/end delta "
                  << start_end_delta_beats << " beats > "
                  << kMaxStartEndDeltaBeats << " beats\n";
        return 1;
    }
    if (odd_even_gap_ms > kMaxOddEvenMedianGapMs) {
        std::cerr << "Enigma alignment test failed: odd/even median gap "
                  << odd_even_gap_ms << "ms > " << kMaxOddEvenMedianGapMs << "ms\n";
        return 1;
    }
    if (!(early_bpm > 0.0) || !(late_bpm > 0.0)) {
        std::cerr << "Enigma alignment test failed: invalid edge BPM estimate.\n";
        return 1;
    }
    if (edge_bpm_delta > kMaxTempoEdgeBpmDelta) {
        std::cerr << "Enigma alignment test failed: edge BPM delta " << edge_bpm_delta
                  << " > " << kMaxTempoEdgeBpmDelta << "\n";
        return 1;
    }

    return 0;
}
