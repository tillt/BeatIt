//
//  core.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/analysis.h"
#include "beatit/analysis/torch_backend.h"
#include "beatit/config.h"
#include "beatit/logging.hpp"
#include "beatit/stream.h"

#include "internal.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace beatit {

AnalysisResult analyze(const std::vector<float>& samples,
                       double sample_rate,
                       const BeatitConfig& config) {
    set_log_verbosity_from_config(config);

    AnalysisResult result;
    if (samples.empty() || sample_rate <= 0.0) {
        return result;
    }

    if (config.backend == BeatitConfig::Backend::BeatThisExternal) {
        return analyze_with_beatthis(samples, sample_rate, config);
    }

    if (config.sparse_probe_mode) {
        BeatitStream stream(sample_rate, config, true);
        double start_seconds = 0.0;
        double duration_seconds = 0.0;
        if (!stream.request_analysis_window(&start_seconds, &duration_seconds)) {
            stream.push(samples.data(), samples.size());
            return stream.finalize();
        }

        const double total_duration_seconds = static_cast<double>(samples.size()) / sample_rate;
        auto provider =
            [&](double start_s, double duration_s, std::vector<float>* out_samples) -> std::size_t {
                if (!out_samples || sample_rate <= 0.0 || samples.empty()) {
                    return 0;
                }
                out_samples->clear();
                const double clamped_start = std::max(0.0, start_s);
                const double clamped_duration = std::max(0.0, duration_s);
                const double begin_d = std::floor(clamped_start * sample_rate);
                const double end_d = std::ceil((clamped_start + clamped_duration) * sample_rate);
                const std::size_t begin = static_cast<std::size_t>(std::max(0.0, begin_d));
                const std::size_t end = std::min(
                    samples.size(),
                    static_cast<std::size_t>(std::max(0.0, end_d)));
                if (begin >= end) {
                    return 0;
                }
                out_samples->assign(samples.begin() + static_cast<long>(begin),
                                    samples.begin() + static_cast<long>(end));
                return out_samples->size();
            };
        return stream.analyze_window(start_seconds,
                                     duration_seconds,
                                     total_duration_seconds,
                                     provider);
    }

    BeatitConfig base_config = config;
    base_config.tempo_window_percent = 0.0f;
    base_config.prefer_double_time = false;

    const std::size_t last_active_frame =
        estimate_last_active_frame(samples, sample_rate, config);

    std::vector<float> phase_energy = compute_phase_energy(samples, sample_rate, config);

    auto run_pipeline = [&](const std::vector<float>& beat_activation,
                            const std::vector<float>& downbeat_activation,
                            const std::vector<unsigned long long>& beat_sample_frames) {
        const float peaks_bpm =
            estimate_bpm_from_activation(beat_activation, config, sample_rate);
        const float autocorr_bpm =
            estimate_bpm_from_activation_autocorr(beat_activation, config, sample_rate);
        const float comb_bpm =
            estimate_bpm_from_activation_comb(beat_activation, config, sample_rate);
        const float beats_bpm = estimate_bpm_from_beats(beat_sample_frames, sample_rate);

        const float reference_bpm =
            choose_candidate_bpm(peaks_bpm, autocorr_bpm, comb_bpm, beats_bpm);

        return postprocess_coreml_activations(beat_activation,
                                              downbeat_activation,
                                              &phase_energy,
                                              config,
                                              sample_rate,
                                              reference_bpm,
                                              last_active_frame);
    };

    if (config.backend == BeatitConfig::Backend::Torch) {
        CoreMLResult raw = analyze_with_torch_activations(samples, sample_rate, base_config);
        CoreMLResult final_result = run_pipeline(raw.beat_activation,
                                                 raw.downbeat_activation,
                                                 raw.beat_sample_frames);
        assign_coreml_result(result,
                             std::move(final_result),
                             std::move(phase_energy),
                             sample_rate,
                             config);
        return result;
    }

    CoreMLResult base = analyze_with_coreml(samples, sample_rate, base_config, 0.0f);
    CoreMLResult final_result = run_pipeline(base.beat_activation,
                                             base.downbeat_activation,
                                             base.beat_sample_frames);
    assign_coreml_result(result,
                         std::move(final_result),
                         std::move(phase_energy),
                         sample_rate,
                         config);
    return result;
}

} // namespace beatit
