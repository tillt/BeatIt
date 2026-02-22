//
//  stream.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-01-17.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#pragma once

#include "beatit/analysis.h"
#include "beatit/coreml.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace beatit {
namespace detail {
class InferenceBackend;
}

class BeatitStream {
public:
    BeatitStream(double sample_rate,
                 const CoreMLConfig& coreml_config,
                 bool enable_coreml = true);
    ~BeatitStream();

    using SampleProvider =
        std::function<std::size_t(double start_seconds,
                                  double duration_seconds,
                                  std::vector<float>* out_samples)>;

    void push(const float* samples, std::size_t count);
    void push(const std::vector<float>& samples) { push(samples.data(), samples.size()); }

    // Returns the preferred analysis window (start + duration in seconds).
    // If false, the caller should stream the full file.
    bool request_analysis_window(double* start_seconds, double* duration_seconds) const;

    // Analyze a single window provided by the caller. This is synchronous and should
    // be run off the main thread. If total_duration_seconds > 0, the projected grid
    // will span the full duration even though only a window is analyzed.
    AnalysisResult analyze_window(double start_seconds,
                                  double duration_seconds,
                                  double total_duration_seconds,
                                  const SampleProvider& provider);

    AnalysisResult finalize();

private:
    void reset_state(bool reset_tempo_anchor = false);
    void process_coreml_windows();
    void process_torch_windows();
    void accumulate_phase_energy(std::size_t begin_sample, std::size_t end_sample);

    double sample_rate_ = 0.0;
    CoreMLConfig coreml_config_;
    bool coreml_enabled_ = true;
    std::unique_ptr<detail::InferenceBackend> inference_backend_;

    struct LinearResampler {
        double ratio = 1.0;
        double src_index = 0.0;
        std::vector<float> buffer;

        void push(const float* input, std::size_t count, std::vector<float>& output);
    } resampler_;

    std::vector<float> resampled_buffer_;
    std::size_t resampled_offset_ = 0;
    std::size_t coreml_frame_offset_ = 0;
    std::vector<float> coreml_beat_activation_;
    std::vector<float> coreml_downbeat_activation_;
    std::vector<float> coreml_phase_energy_;
    std::size_t total_input_samples_ = 0;
    std::size_t total_seen_samples_ = 0;
    std::size_t prepend_samples_ = 0;
    bool prepend_done_ = false;
    std::size_t last_active_sample_ = 0;
    bool has_active_sample_ = false;
    double tempo_reference_bpm_ = 0.0;
    bool tempo_reference_valid_ = false;
    double phase_energy_alpha_ = 0.0;
    double phase_energy_state_ = 0.0;
    double phase_energy_sum_sq_ = 0.0;
    std::size_t phase_energy_sample_count_ = 0;

    struct PerfStats {
        double resample_ms = 0.0;
        double process_ms = 0.0;
        double mel_ms = 0.0;
        double torch_forward_ms = 0.0;
        double window_infer_ms = 0.0;
        std::size_t window_count = 0;
        double finalize_infer_ms = 0.0;
        double postprocess_ms = 0.0;
        double marker_ms = 0.0;
        double finalize_ms = 0.0;
    } perf_;
};

} // namespace beatit
