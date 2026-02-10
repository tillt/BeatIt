#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "beatit/refiner.h"
#include "beatit/stream.h"
#include "coreml_test_config.h"

namespace {

constexpr double kTargetBpm = 110.0;
constexpr double kMaxIntervalStdMilliseconds = 1.0;
constexpr double kMaxPeakOffsetMeanAbsMilliseconds = 25.0;
constexpr double kMaxPeakOffsetSlopeMsPerBeat = 0.05;

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

bool stream_audio_to_beatit(const std::string& path,
                            const beatit::CoreMLConfig& config,
                            beatit::AnalysisResult* result,
                            double* input_sample_rate,
                            std::vector<float>* decoded_mono,
                            std::string* error) {
    if (!result) {
        if (error) {
            *error = "Result pointer is null.";
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

        beatit::BeatitStream stream(sample_rate, config, true);
        std::vector<float> mono;
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
            mono.assign(frames, 0.0f);
            for (AVAudioChannelCount ch = 0; ch < channels; ++ch) {
                const float* channel_data = output_buffer.floatChannelData[ch];
                for (AVAudioFrameCount i = 0; i < frames; ++i) {
                    mono[i] += channel_data[i];
                }
            }
            const float scale = 1.0f / static_cast<float>(channels);
            for (float& sample : mono) {
                sample *= scale;
            }

            if (decoded_mono) {
                decoded_mono->insert(decoded_mono->end(), mono.begin(), mono.end());
            }
            stream.push(mono.data(), mono.size());
        }

        if (read_error) {
            if (error) {
                *error = read_error.localizedDescription.UTF8String;
            }
            return false;
        }

        *result = stream.finalize();
        if (input_sample_rate) {
            *input_sample_rate = sample_rate;
        }
    }

    return true;
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum / static_cast<double>(values.size());
}

double stddev(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double m = mean(values);
    double sum_sq = 0.0;
    for (double v : values) {
        const double d = v - m;
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq / static_cast<double>(values.size()));
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

bool first_downbeat_sample_frame(const beatit::AnalysisResult& result,
                                 unsigned long long* frame) {
    if (!frame) {
        return false;
    }
    for (const auto& event : result.coreml_beat_events) {
        if (is_bar_event(event)) {
            *frame = event.frame;
            return true;
        }
    }
    return false;
}

void write_index_series_csv(const std::filesystem::path& path,
                            const char* header,
                            const std::vector<unsigned long long>& values) {
    std::ofstream out(path);
    out << header << "\n";
    for (std::size_t i = 0; i < values.size(); ++i) {
        out << i << "," << values[i] << "\n";
    }
}

void write_activation_csv(const std::filesystem::path& path,
                          const std::vector<float>& beat,
                          const std::vector<float>& downbeat,
                          const std::vector<float>& phase) {
    std::ofstream out(path);
    out << "frame,beat_activation,downbeat_activation,phase_energy\n";
    const std::size_t count = std::max(beat.size(), std::max(downbeat.size(), phase.size()));
    for (std::size_t i = 0; i < count; ++i) {
        out << i << ",";
        if (i < beat.size()) {
            out << std::setprecision(8) << beat[i];
        }
        out << ",";
        if (i < downbeat.size()) {
            out << std::setprecision(8) << downbeat[i];
        }
        out << ",";
        if (i < phase.size()) {
            out << std::setprecision(8) << phase[i];
        }
        out << "\n";
    }
}

void write_summary(const std::filesystem::path& path,
                   const beatit::AnalysisResult& result,
                   double sample_rate,
                   double interval_std_ms,
                   double peak_offset_mean_abs_ms,
                   double peak_offset_slope_ms_per_beat) {
    std::ofstream out(path);
    out << std::fixed << std::setprecision(6);
    out << "estimated_bpm=" << result.estimated_bpm << "\n";
    out << "interval_std_ms=" << interval_std_ms << "\n";
    out << "peak_offset_mean_abs_ms=" << peak_offset_mean_abs_ms << "\n";
    out << "peak_offset_slope_ms_per_beat=" << peak_offset_slope_ms_per_beat << "\n";
    out << "sample_rate=" << sample_rate << "\n";
    out << "beat_sample_count=" << result.coreml_beat_sample_frames.size() << "\n";
    out << "beat_feature_count=" << result.coreml_beat_feature_frames.size() << "\n";
    out << "downbeat_feature_count=" << result.coreml_downbeat_feature_frames.size() << "\n";
    out << "beat_events_count=" << result.coreml_beat_events.size() << "\n";
}

struct PeakOffsetRow {
    std::size_t index = 0;
    std::size_t beat_frame = 0;
    std::size_t nearest_peak_frame = 0;
    double beat_time = 0.0;
    double peak_time = 0.0;
    double offset_ms = 0.0;
    float peak_abs = 0.0f;
    std::size_t window_start = 0;
    std::size_t window_end = 0;
};

std::vector<PeakOffsetRow> compute_beat_peak_offsets(const std::vector<unsigned long long>& beat_sample_frames,
                                                     const std::vector<float>& samples,
                                                     double sample_rate,
                                                     double bpm_for_window) {
    std::vector<PeakOffsetRow> rows;
    if (beat_sample_frames.empty() || samples.empty() || sample_rate <= 0.0 || bpm_for_window <= 0.0) {
        return rows;
    }

    const double beat_period_s = 60.0 / bpm_for_window;
    const std::size_t radius = static_cast<std::size_t>(std::llround(sample_rate * beat_period_s * 0.6));

    const auto sample_abs = [&](std::size_t idx) -> float {
        return std::fabs(samples[idx]);
    };

    rows.reserve(beat_sample_frames.size());
    for (std::size_t i = 0; i < beat_sample_frames.size(); ++i) {
        const std::size_t beat_frame =
            static_cast<std::size_t>(std::min<unsigned long long>(beat_sample_frames[i], samples.size() - 1));
        const std::size_t start = beat_frame > radius ? beat_frame - radius : 0;
        const std::size_t end = std::min(samples.size() - 1, beat_frame + radius);

        float window_max = 0.0f;
        std::size_t window_max_idx = beat_frame;
        for (std::size_t p = start; p <= end; ++p) {
            const float a = sample_abs(p);
            if (a > window_max) {
                window_max = a;
                window_max_idx = p;
            }
        }

        const float threshold = window_max * 0.2f;
        std::size_t best_peak = window_max_idx;
        std::size_t best_distance = std::numeric_limits<std::size_t>::max();

        if (end > start + 1) {
            for (std::size_t p = start + 1; p < end; ++p) {
                const float left = sample_abs(p - 1);
                const float cur = sample_abs(p);
                const float right = sample_abs(p + 1);
                if (cur < threshold) {
                    continue;
                }
                if (cur >= left && cur > right) {
                    const std::size_t distance = p > beat_frame ? p - beat_frame : beat_frame - p;
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_peak = p;
                    }
                }
            }
        }

        PeakOffsetRow row;
        row.index = i;
        row.beat_frame = beat_frame;
        row.nearest_peak_frame = best_peak;
        row.beat_time = static_cast<double>(beat_frame) / sample_rate;
        row.peak_time = static_cast<double>(best_peak) / sample_rate;
        row.offset_ms = (row.peak_time - row.beat_time) * 1000.0;
        row.peak_abs = sample_abs(best_peak);
        row.window_start = start;
        row.window_end = end;
        rows.push_back(row);
    }

    return rows;
}

void write_beat_peak_offsets_csv(const std::filesystem::path& path,
                                 const std::vector<PeakOffsetRow>& rows) {
    std::ofstream out(path);
    out << "index,beat_frame,beat_time_s,nearest_peak_frame,peak_time_s,offset_ms,peak_abs,window_start,window_end\n";
    for (const auto& row : rows) {
        out << row.index << ","
            << row.beat_frame << ","
            << std::setprecision(8) << row.beat_time << ","
            << row.nearest_peak_frame << ","
            << std::setprecision(8) << row.peak_time << ","
            << std::setprecision(8) << row.offset_ms << ","
            << std::setprecision(8) << row.peak_abs << ","
            << row.window_start << ","
            << row.window_end << "\n";
    }
}

std::filesystem::path dump_dir_for_run() {
    if (const char* env = std::getenv("BEATIT_TEST_DUMP_DIR")) {
        if (env[0] != '\0') {
            return std::filesystem::path(env);
        }
    }
    return std::filesystem::current_path() / "logs" / "manucho_alignment_dump";
}

void write_complete_dump(const beatit::AnalysisResult& result,
                         const std::vector<unsigned long long>& grid_sample_frames,
                         const std::vector<float>& decoded_mono,
                         double sample_rate,
                         double interval_std_ms,
                         double peak_offset_mean_abs_ms,
                         double peak_offset_slope_ms_per_beat,
                         std::string* error,
                         std::filesystem::path* out_dir) {
    try {
        const std::filesystem::path dump_dir = dump_dir_for_run();
        std::filesystem::create_directories(dump_dir);

        write_index_series_csv(dump_dir / "beat_sample_frames.csv",
                               "index,sample_frame",
                               result.coreml_beat_sample_frames);
        write_index_series_csv(dump_dir / "beat_feature_frames.csv",
                               "index,feature_frame",
                               result.coreml_beat_feature_frames);
        write_index_series_csv(dump_dir / "downbeat_feature_frames.csv",
                               "index,feature_frame",
                               result.coreml_downbeat_feature_frames);
        write_index_series_csv(dump_dir / "projected_beat_sample_frames.csv",
                               "index,sample_frame",
                               result.coreml_beat_projected_sample_frames);
        write_index_series_csv(dump_dir / "projected_beat_feature_frames.csv",
                               "index,feature_frame",
                               result.coreml_beat_projected_feature_frames);
        write_index_series_csv(dump_dir / "projected_downbeat_feature_frames.csv",
                               "index,feature_frame",
                               result.coreml_downbeat_projected_feature_frames);

        {
            std::ofstream out(dump_dir / "beat_events.csv");
            beatit::write_beat_events_csv(out, result.coreml_beat_events, true);
        }

        write_activation_csv(dump_dir / "activations.csv",
                             result.coreml_beat_activation,
                             result.coreml_downbeat_activation,
                             result.coreml_phase_energy);

        write_summary(dump_dir / "summary.txt",
                      result,
                      sample_rate,
                      interval_std_ms,
                      peak_offset_mean_abs_ms,
                      peak_offset_slope_ms_per_beat);
        const double window_bpm = result.estimated_bpm > 0.0 ? result.estimated_bpm : kTargetBpm;
        const std::vector<PeakOffsetRow> peak_rows =
            compute_beat_peak_offsets(grid_sample_frames,
                                      decoded_mono,
                                      sample_rate,
                                      window_bpm);
        write_beat_peak_offsets_csv(dump_dir / "beat_to_peak_offsets.csv", peak_rows);

        if (out_dir) {
            *out_dir = dump_dir;
        }
    } catch (const std::exception& ex) {
        if (error) {
            *error = ex.what();
        }
    }
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
    const std::filesystem::path audio_path = test_root / "training" / "manucho.wav";
    if (!std::filesystem::exists(audio_path)) {
        std::cerr << "SKIP: missing " << audio_path.string() << "\n";
        return 77;
    }

    beatit::CoreMLConfig config;
    beatit::tests::apply_beatthis_coreml_test_config(config);
    config.model_path = model_path;
    config.use_dbn = true;
    config.prepend_silence_seconds = 0.0;
    if (const char* force_cpu = std::getenv("BEATIT_TEST_CPU_ONLY")) {
        if (force_cpu[0] != '\0' && force_cpu[0] != '0') {
            config.compute_units = beatit::CoreMLConfig::ComputeUnits::CPUOnly;
        }
    }

    beatit::AnalysisResult result;
    std::string error;
    double sample_rate = 0.0;
    std::vector<float> decoded_mono;
    decoded_mono.reserve(44100 * 60 * 8);
    if (!stream_audio_to_beatit(audio_path.string(),
                                config,
                                &result,
                                &sample_rate,
                                &decoded_mono,
                                &error)) {
        std::cerr << "Manucho alignment test failed: " << error << "\n";
        return 1;
    }
    if (result.coreml_beat_sample_frames.empty()) {
        std::cerr << "Manucho alignment test failed: missing beat sample frames.\n";
        return 1;
    }
    unsigned long long first_downbeat_frame = 0;
    if (!first_downbeat_sample_frame(result, &first_downbeat_frame)) {
        std::cerr << "Manucho alignment test failed: missing downbeat beat events.\n";
        return 1;
    }

    if (!(result.estimated_bpm > 0.0)) {
        std::cerr << "Manucho alignment test failed: non-positive BPM.\n";
        return 1;
    }

    std::vector<unsigned long long> grid_sample_frames;
    grid_sample_frames.reserve(result.coreml_beat_events.size());
    for (const auto& event : result.coreml_beat_events) {
        grid_sample_frames.push_back(event.frame);
    }
    if (grid_sample_frames.size() < 2) {
        std::cerr << "Manucho alignment test failed: insufficient beat event frames.\n";
        return 1;
    }

    std::vector<double> intervals_ms;
    intervals_ms.reserve(grid_sample_frames.size());
    for (std::size_t i = 1; i < grid_sample_frames.size(); ++i) {
        const double delta_frames =
            static_cast<double>(grid_sample_frames[i] - grid_sample_frames[i - 1]);
        intervals_ms.push_back((delta_frames / sample_rate) * 1000.0);
    }
    const double interval_std_ms = stddev(intervals_ms);

    const std::vector<PeakOffsetRow> peak_rows =
        compute_beat_peak_offsets(grid_sample_frames,
                                  decoded_mono,
                                  sample_rate,
                                  result.estimated_bpm);
    std::vector<double> offset_ms;
    offset_ms.reserve(peak_rows.size());
    for (const auto& row : peak_rows) {
        offset_ms.push_back(row.offset_ms);
    }
    std::vector<double> abs_offset_ms;
    abs_offset_ms.reserve(offset_ms.size());
    for (double v : offset_ms) {
        abs_offset_ms.push_back(std::fabs(v));
    }
    const double peak_offset_mean_abs_ms = mean(abs_offset_ms);
    const double peak_offset_slope_ms_per_beat = linear_slope(offset_ms);

    std::string dump_error;
    std::filesystem::path dump_dir;
    write_complete_dump(result,
                        grid_sample_frames,
                        decoded_mono,
                        sample_rate,
                        interval_std_ms,
                        peak_offset_mean_abs_ms,
                        peak_offset_slope_ms_per_beat,
                        &dump_error,
                        &dump_dir);
    if (!dump_error.empty()) {
        std::cerr << "Manucho alignment test warning: failed to write dump: "
                  << dump_error << "\n";
    } else {
        std::cout << "Manucho alignment dump: " << dump_dir.string() << "\n";
    }

    if (interval_std_ms > kMaxIntervalStdMilliseconds) {
        std::cerr << "Manucho alignment test failed: interval std "
                  << interval_std_ms << "ms > " << kMaxIntervalStdMilliseconds << "ms\n";
        return 1;
    }
    if (peak_offset_mean_abs_ms > kMaxPeakOffsetMeanAbsMilliseconds) {
        std::cerr << "Manucho alignment test failed: peak offset mean abs "
                  << peak_offset_mean_abs_ms << "ms > "
                  << kMaxPeakOffsetMeanAbsMilliseconds << "ms\n";
        return 1;
    }
    if (std::fabs(peak_offset_slope_ms_per_beat) > kMaxPeakOffsetSlopeMsPerBeat) {
        std::cerr << "Manucho alignment test failed: peak offset slope "
                  << peak_offset_slope_ms_per_beat << "ms/beat > "
                  << kMaxPeakOffsetSlopeMsPerBeat << "ms/beat\n";
        return 1;
    }

    std::cout << "Manucho alignment test passed. "
              << "bpm=" << result.estimated_bpm
              << " first_downbeat_frame=" << first_downbeat_frame
              << " interval_std_ms=" << interval_std_ms
              << " peak_offset_mean_abs_ms=" << peak_offset_mean_abs_ms
              << " peak_offset_slope_ms_per_beat=" << peak_offset_slope_ms_per_beat
              << "\n";
    return 0;
}
