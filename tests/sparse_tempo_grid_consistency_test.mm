#import <AVFoundation/AVFoundation.h>
#import <CoreML/CoreML.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "beatit/analysis.h"
#include "beatit/coreml_preset.h"
#include "coreml_test_config.h"

namespace {

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

bool decode_audio_mono_limited(const std::string& path,
                               double max_seconds,
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
        const std::size_t max_frames = max_seconds > 0.0
                                           ? static_cast<std::size_t>(std::llround(max_seconds * sample_rate))
                                           : 0;

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
            std::size_t frames_to_copy = static_cast<std::size_t>(frames);
            if (max_frames > 0 && mono_out->size() < max_frames) {
                frames_to_copy = std::min<std::size_t>(frames_to_copy, max_frames - mono_out->size());
            }

            const std::size_t base = mono_out->size();
            mono_out->resize(base + frames_to_copy, 0.0f);
            for (AVAudioChannelCount ch = 0; ch < channels; ++ch) {
                const float* channel_data = output_buffer.floatChannelData[ch];
                for (std::size_t i = 0; i < frames_to_copy; ++i) {
                    (*mono_out)[base + i] += channel_data[i];
                }
            }
            const float scale = 1.0f / static_cast<float>(channels);
            for (std::size_t i = 0; i < frames_to_copy; ++i) {
                (*mono_out)[base + i] *= scale;
            }

            if (max_frames > 0 && mono_out->size() >= max_frames) {
                break;
            }
        }

        if (read_error) {
            if (error) {
                *error = read_error.localizedDescription.UTF8String
                             ? read_error.localizedDescription.UTF8String
                             : "Unknown AVAudioFile read error.";
            }
            return false;
        }

        if (sample_rate_out) {
            *sample_rate_out = sample_rate;
        }
    }

    return true;
}

void write_pcm16_wav(const std::filesystem::path& path,
                     const std::vector<float>& samples,
                     int sample_rate) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return;
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * channels * (bits_per_sample / 8);
    const uint16_t block_align = channels * (bits_per_sample / 8);
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riff_size), sizeof(riff_size));
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    const uint32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), sizeof(fmt_size));
    const uint16_t audio_format = 1;
    out.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
    out.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    const uint32_t sr_u32 = static_cast<uint32_t>(sample_rate);
    out.write(reinterpret_cast<const char*>(&sr_u32), sizeof(sr_u32));
    out.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    out.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
    out.write(reinterpret_cast<const char*>(&bits_per_sample), sizeof(bits_per_sample));
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

    for (float sample : samples) {
        const float clamped = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t pcm = static_cast<int16_t>(std::lrint(clamped * 32767.0f));
        out.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
    }
}

std::vector<float> make_click_track(double sample_rate,
                                    double bpm,
                                    double seconds,
                                    double pulse_ms,
                                    float amp,
                                    double active_start_s,
                                    double active_end_s) {
    const std::size_t total_samples =
        static_cast<std::size_t>(std::ceil(sample_rate * seconds));
    const std::size_t beat_period =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::round(sample_rate * 60.0 / bpm)));
    const std::size_t pulse_width =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::round(sample_rate * pulse_ms * 0.001)));
    std::vector<float> samples(total_samples, 0.0f);

    const std::size_t active_start =
        static_cast<std::size_t>(std::max(0.0, active_start_s) * sample_rate);
    const std::size_t active_end =
        std::min<std::size_t>(total_samples,
                              static_cast<std::size_t>(std::max(0.0, active_end_s) * sample_rate));

    for (std::size_t i = active_start; i < active_end; ++i) {
        const std::size_t phase = (i - active_start) % beat_period;
        if (phase < pulse_width) {
            const float taper = 1.0f - static_cast<float>(phase) /
                                           static_cast<float>(pulse_width);
            samples[i] = amp * taper;
        }
    }

    return samples;
}

std::vector<float> make_tremolo_sine(double sample_rate,
                                     double carrier_hz,
                                     double bpm_mod,
                                     double seconds,
                                     float amp) {
    const std::size_t total_samples =
        static_cast<std::size_t>(std::ceil(sample_rate * seconds));
    std::vector<float> samples(total_samples, 0.0f);
    const double w_carrier = 2.0 * M_PI * carrier_hz;
    const double w_mod = 2.0 * M_PI * (bpm_mod / 60.0);
    for (std::size_t i = 0; i < total_samples; ++i) {
        const double t = static_cast<double>(i) / sample_rate;
        const double env = 0.5 * (1.0 + std::sin(w_mod * t));
        samples[i] = amp * static_cast<float>(std::sin(w_carrier * t) * env);
    }
    return samples;
}

struct Scenario {
    std::string name;
    std::string description;
    std::vector<float> samples;
};

struct ScenarioExpectation {
    bool expect_no_tempo = false;
    bool expect_no_projected_grid = false;
    std::size_t max_beats = std::numeric_limits<std::size_t>::max();
    std::size_t max_events = std::numeric_limits<std::size_t>::max();
};

std::size_t grid_count(const beatit::AnalysisResult& result) {
    const std::size_t events = result.coreml_beat_events.size();
    const std::size_t projected = result.coreml_beat_projected_sample_frames.size();
    const std::size_t beats = result.coreml_beat_sample_frames.size();
    return std::max(events, std::max(projected, beats));
}

bool env_flag_enabled(const char* name) {
    if (const char* value = std::getenv(name)) {
        return value[0] != '\0' && value[0] != '0';
    }
    return false;
}

std::size_t env_size_t_or(const char* name,
                          std::size_t fallback,
                          std::size_t min_value,
                          std::size_t max_value) {
    if (const char* value = std::getenv(name)) {
        if (value[0] == '\0') {
            return fallback;
        }
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end && *end == '\0') {
            const std::size_t result = static_cast<std::size_t>(parsed);
            return std::min(max_value, std::max(min_value, result));
        }
    }
    return fallback;
}

uint64_t env_u64_or(const char* name, uint64_t fallback) {
    if (const char* value = std::getenv(name)) {
        if (value[0] == '\0') {
            return fallback;
        }
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end && *end == '\0') {
            return static_cast<uint64_t>(parsed);
        }
    }
    return fallback;
}

double env_double_or(const char* name,
                     double fallback,
                     double min_value,
                     double max_value) {
    if (const char* value = std::getenv(name)) {
        if (value[0] == '\0') {
            return fallback;
        }
        char* end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end && *end == '\0' && std::isfinite(parsed)) {
            return std::min(max_value, std::max(min_value, parsed));
        }
    }
    return fallback;
}

void add_in_place(std::vector<float>* dst, const std::vector<float>& src, float gain) {
    if (!dst) {
        return;
    }
    if (dst->size() < src.size()) {
        dst->resize(src.size(), 0.0f);
    }
    for (std::size_t i = 0; i < src.size(); ++i) {
        (*dst)[i] += src[i] * gain;
    }
}

void clamp_audio(std::vector<float>* samples) {
    if (!samples) {
        return;
    }
    for (float& sample : *samples) {
        sample = std::max(-1.0f, std::min(1.0f, sample));
    }
}

Scenario make_random_scenario(std::size_t index,
                              std::mt19937_64* rng,
                              double sample_rate,
                              double duration_seconds) {
    Scenario scenario;
    if (!rng) {
        scenario.name = "invalid_rng";
        scenario.description = "invalid_rng";
        return scenario;
    }

    std::uniform_real_distribution<double> bpm_dist(70.0, 180.0);
    std::uniform_real_distribution<double> pulse_dist(2.0, 20.0);
    std::uniform_real_distribution<double> amp_dist(0.02, 1.0);
    std::uniform_real_distribution<double> frac_dist(0.0, 1.0);
    std::uniform_real_distribution<double> hz_dist(40.0, 240.0);
    std::uniform_int_distribution<int> kind_dist(0, 5);

    const int kind = kind_dist(*rng);
    const double bpm = bpm_dist(*rng);
    const double pulse_ms = pulse_dist(*rng);
    const float amp = static_cast<float>(amp_dist(*rng));
    const double intro_frac = std::min(0.5, frac_dist(*rng) * 0.5);
    const double outro_frac = std::min(0.5, frac_dist(*rng) * 0.5);
    const double intro_end = duration_seconds * intro_frac;
    const double outro_start = duration_seconds * (1.0 - outro_frac);

    std::ostringstream name;
    name << "sweep_" << index << "_k" << kind;
    scenario.name = name.str();

    std::vector<float> samples(static_cast<std::size_t>(std::ceil(sample_rate * duration_seconds)), 0.0f);

    if (kind == 0) {
        samples = make_click_track(sample_rate, bpm, duration_seconds, pulse_ms, amp, 0.0, duration_seconds);
        scenario.description = "steady_click bpm=" + std::to_string(bpm) +
                               " pulse_ms=" + std::to_string(pulse_ms) +
                               " amp=" + std::to_string(amp);
    } else if (kind == 1) {
        samples = make_click_track(sample_rate, bpm, duration_seconds, pulse_ms, amp, 0.0, intro_end);
        scenario.description = "intro_click bpm=" + std::to_string(bpm) +
                               " active=[0," + std::to_string(intro_end) + "]";
    } else if (kind == 2) {
        samples = make_click_track(sample_rate, bpm, duration_seconds, pulse_ms, amp, outro_start, duration_seconds);
        scenario.description = "outro_click bpm=" + std::to_string(bpm) +
                               " active=[" + std::to_string(outro_start) + "," +
                               std::to_string(duration_seconds) + "]";
    } else if (kind == 3) {
        const double ratio = (frac_dist(*rng) < 0.5) ? 0.5 : 2.0;
        const double secondary_bpm = std::max(40.0, bpm * ratio);
        samples = make_click_track(sample_rate, bpm, duration_seconds, pulse_ms, amp, 0.0, duration_seconds);
        const std::vector<float> secondary =
            make_click_track(sample_rate, secondary_bpm, duration_seconds, pulse_ms * 0.75, amp * 0.35f, 0.0, duration_seconds);
        add_in_place(&samples, secondary, 1.0f);
        scenario.description = "dual_click bpm_a=" + std::to_string(bpm) +
                               " bpm_b=" + std::to_string(secondary_bpm);
    } else if (kind == 4) {
        samples = make_tremolo_sine(sample_rate, hz_dist(*rng), bpm, duration_seconds, amp * 0.5f);
        const std::vector<float> clicks =
            make_click_track(sample_rate, bpm, duration_seconds, pulse_ms, amp * 0.15f, 0.0, duration_seconds);
        add_in_place(&samples, clicks, 1.0f);
        scenario.description = "tremolo_plus_click bpm=" + std::to_string(bpm);
    } else {
        samples = make_click_track(sample_rate, bpm, duration_seconds, pulse_ms, amp * 0.2f, 0.0, duration_seconds);
        std::normal_distribution<float> noise_dist(0.0f, amp * 0.03f);
        for (float& sample : samples) {
            sample += noise_dist(*rng);
        }
        scenario.description = "weak_click_with_noise bpm=" + std::to_string(bpm) +
                               " amp=" + std::to_string(amp * 0.2f);
    }

    clamp_audio(&samples);
    scenario.samples = std::move(samples);
    return scenario;
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

    beatit::CoreMLConfig config;
    if (auto preset = beatit::make_coreml_preset("beatthis")) {
        preset->apply(config);
    }
    config.model_path = model_path;
    config.sparse_probe_mode = true;
    config.use_dbn = true;
    config.dbn_project_grid = true;
    config.prepend_silence_seconds = 0.0;
    if (const char* force_cpu = std::getenv("BEATIT_TEST_CPU_ONLY")) {
        if (force_cpu[0] != '\0' && force_cpu[0] != '0') {
            config.compute_units = beatit::CoreMLConfig::ComputeUnits::CPUOnly;
        }
    }

    constexpr double kSampleRate = 44100.0;
    const double duration_seconds =
        env_double_or("BEATIT_SYNTH_SWEEP_DURATION", 45.0, 20.0, 180.0);
    const bool sweep_flag = env_flag_enabled("BEATIT_SYNTH_SWEEP");
    std::size_t sweep_cases =
        env_size_t_or("BEATIT_SYNTH_SWEEP_CASES", 0, 0, 5000);
    if (sweep_flag && sweep_cases == 0) {
        sweep_cases = 64;
    }
    const uint64_t sweep_seed = env_u64_or("BEATIT_SYNTH_SWEEP_SEED", 1337ULL);
    std::mt19937_64 sweep_rng(sweep_seed);
    const char* real_audio_env = std::getenv("BEATIT_SYNTH_REAL_AUDIO_PATH");
    const std::string real_audio_path = real_audio_env ? std::string(real_audio_env) : std::string();
    const double real_audio_max_seconds =
        env_double_or("BEATIT_SYNTH_REAL_AUDIO_MAX_SECONDS", 0.0, 0.0, 3600.0);

    std::vector<Scenario> scenarios;
    scenarios.push_back({"steady_click_110",
                         "steady_click_110",
                         make_click_track(kSampleRate, 110.0, duration_seconds, 8.0, 0.9f, 0.0, duration_seconds)});
    scenarios.push_back({"intro_only_click_110",
                         "intro_only_click_110",
                         make_click_track(kSampleRate, 110.0, duration_seconds, 8.0, 0.9f, 0.0, 12.0)});
    scenarios.push_back({"outro_only_click_110",
                         "outro_only_click_110",
                         make_click_track(kSampleRate, 110.0, duration_seconds, 8.0, 0.9f, 33.0, duration_seconds)});
    scenarios.push_back({"weak_click_110",
                         "weak_click_110",
                         make_click_track(kSampleRate, 110.0, duration_seconds, 5.0, 0.05f, 0.0, duration_seconds)});
    scenarios.push_back({"tremolo_sine_110",
                         "tremolo_sine_110",
                         make_tremolo_sine(kSampleRate, 80.0, 110.0, duration_seconds, 0.2f)});

    if (sweep_cases > 0) {
        for (std::size_t i = 0; i < sweep_cases; ++i) {
            scenarios.push_back(make_random_scenario(i, &sweep_rng, kSampleRate, duration_seconds));
        }
    }

    if (!real_audio_path.empty()) {
        std::vector<float> real_audio_samples;
        double real_audio_rate = 0.0;
        std::string decode_error;
        if (!decode_audio_mono_limited(real_audio_path,
                                       real_audio_max_seconds,
                                       &real_audio_samples,
                                       &real_audio_rate,
                                       &decode_error)) {
            std::cerr << "Sparse synthetic test failed: could not decode BEATIT_SYNTH_REAL_AUDIO_PATH="
                      << real_audio_path << " error=" << decode_error << "\n";
            return 1;
        }
        if (real_audio_samples.empty()) {
            std::cerr << "Sparse synthetic test failed: decoded empty audio from "
                      << real_audio_path << "\n";
            return 1;
        }
        if (std::abs(real_audio_rate - kSampleRate) > 1.0) {
            std::cerr << "Sparse synthetic test failed: expected decoded rate near "
                      << kSampleRate << "Hz, got " << real_audio_rate << "Hz.\n";
            return 1;
        }

        const std::filesystem::path p(real_audio_path);
        const std::string stem = p.stem().string().empty() ? "external_audio" : p.stem().string();
        std::ostringstream description;
        description << "real_audio path=" << real_audio_path;
        if (real_audio_max_seconds > 0.0) {
            description << " max_s=" << real_audio_max_seconds;
        }
        scenarios.push_back({"real_" + stem, description.str(), std::move(real_audio_samples)});
    }

    const std::filesystem::path test_root =
#ifdef BEATIT_TEST_DATA_DIR
        std::filesystem::path(BEATIT_TEST_DATA_DIR);
#else
        std::filesystem::current_path();
#endif
    const std::filesystem::path echoes_path = test_root / "training" / "echoes.wav";
    if (!std::filesystem::exists(echoes_path)) {
        std::cerr << "Sparse synthetic test failed: missing required non-rhythmic fixture: "
                  << echoes_path.string() << "\n";
        return 1;
    }
    {
        std::vector<float> echoes_samples;
        double echoes_rate = 0.0;
        std::string decode_error;
        if (!decode_audio_mono_limited(echoes_path.string(),
                                       0.0,
                                       &echoes_samples,
                                       &echoes_rate,
                                       &decode_error)) {
            std::cerr << "Sparse synthetic test failed: could not decode echoes fixture: "
                      << decode_error << "\n";
            return 1;
        }
        if (echoes_samples.empty()) {
            std::cerr << "Sparse synthetic test failed: decoded empty echoes fixture.\n";
            return 1;
        }
        if (std::abs(echoes_rate - kSampleRate) > 1.0) {
            std::cerr << "Sparse synthetic test failed: echoes fixture rate mismatch; expected "
                      << kSampleRate << "Hz, got " << echoes_rate << "Hz.\n";
            return 1;
        }
        scenarios.push_back({"echoes_nonrhythmic",
                             "ambient_nonrhythmic_regression path=" + echoes_path.string(),
                             std::move(echoes_samples)});
    }

    std::cout << "Sparse synthetic corpus: baseline=5 sweep=" << sweep_cases
              << " seed=" << sweep_seed
              << " duration_s=" << duration_seconds
              << " real_audio=" << (real_audio_path.empty() ? "0" : "1")
              << "\n";

    bool found_mismatch = false;
    std::string failing_name;
    std::string failing_description;
    std::string failing_reason;
    beatit::AnalysisResult failing_result;
    std::vector<float> failing_samples;

    for (const auto& scenario : scenarios) {
        const beatit::AnalysisResult result = beatit::analyze(scenario.samples,
                                                              kSampleRate,
                                                              config);
        const std::size_t total_grid = grid_count(result);
        std::cout << "Sparse synthetic: " << scenario.name
                  << " desc=" << scenario.description
                  << " bpm=" << result.estimated_bpm
                  << " grid=" << total_grid
                  << " beats=" << result.coreml_beat_sample_frames.size()
                  << " projected=" << result.coreml_beat_projected_sample_frames.size()
                  << " events=" << result.coreml_beat_events.size()
                  << "\n";

        const ScenarioExpectation expectation = [&]() {
            ScenarioExpectation e;
            if (scenario.name == "echoes_nonrhythmic") {
                e.expect_no_tempo = true;
                e.expect_no_projected_grid = true;
                e.max_beats = 1;
                e.max_events = 1;
            }
            return e;
        }();

        if (expectation.expect_no_tempo && result.estimated_bpm > 0.0f) {
            found_mismatch = true;
            failing_name = scenario.name;
            failing_description = scenario.description;
            failing_reason = "expected_no_tempo";
            failing_result = result;
            failing_samples = scenario.samples;
            break;
        }

        if (expectation.expect_no_projected_grid &&
            !result.coreml_beat_projected_sample_frames.empty()) {
            found_mismatch = true;
            failing_name = scenario.name;
            failing_description = scenario.description;
            failing_reason = "expected_no_projected_grid";
            failing_result = result;
            failing_samples = scenario.samples;
            break;
        }

        if (result.coreml_beat_sample_frames.size() > expectation.max_beats) {
            found_mismatch = true;
            failing_name = scenario.name;
            failing_description = scenario.description;
            failing_reason = "beat_count_exceeds_expected";
            failing_result = result;
            failing_samples = scenario.samples;
            break;
        }

        if (result.coreml_beat_events.size() > expectation.max_events) {
            found_mismatch = true;
            failing_name = scenario.name;
            failing_description = scenario.description;
            failing_reason = "event_count_exceeds_expected";
            failing_result = result;
            failing_samples = scenario.samples;
            break;
        }

        if (result.estimated_bpm > 0.0f && total_grid == 0) {
            found_mismatch = true;
            failing_name = scenario.name;
            failing_description = scenario.description;
            failing_reason = "tempo_without_user_grid";
            failing_result = result;
            failing_samples = scenario.samples;
            break;
        }
    }

    if (found_mismatch) {
        const std::filesystem::path dump_dir =
            std::filesystem::current_path() / "logs" / "sparse_tempo_without_grid";
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        const std::filesystem::path wav_path = dump_dir / (failing_name + ".wav");
        const std::filesystem::path meta_path = dump_dir / (failing_name + ".txt");
        write_pcm16_wav(wav_path,
                        failing_samples,
                        static_cast<int>(kSampleRate));
        {
            std::ofstream meta(meta_path);
            if (meta) {
                meta << "name=" << failing_name << "\n";
                meta << "description=" << failing_description << "\n";
                meta << "reason=" << failing_reason << "\n";
                meta << "seed=" << sweep_seed << "\n";
                meta << "duration_s=" << duration_seconds << "\n";
                meta << "sweep_cases=" << sweep_cases << "\n";
                meta << "estimated_bpm=" << failing_result.estimated_bpm << "\n";
                meta << "events=" << failing_result.coreml_beat_events.size() << "\n";
                meta << "beat_sample_frames=" << failing_result.coreml_beat_sample_frames.size() << "\n";
                meta << "projected_sample_frames=" << failing_result.coreml_beat_projected_sample_frames.size() << "\n";
            }
        }
        std::cerr << "Sparse tempo/grid consistency test failed: scenario '" << failing_name
                  << "' (" << failing_description << ")"
                  << " reason=" << failing_reason
                  << "' returned estimated_bpm=" << failing_result.estimated_bpm
                  << ". Wrote repro audio to "
                  << wav_path.string() << "\n";
        return 1;
    }

    std::cout << "Sparse tempo/grid consistency test passed.\n";
    return 0;
}
