//
//  dbn_run.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/dbn_run.h"

#include "beatit/post/dbn_apply.h"
#include "beatit/post/helpers.h"
#include "beatit/post/result_ops.h"
#include "beatit/post/tempo_fit.h"
#include "beatit/post/window.h"
#include "beatit/logging.hpp"
#include "beatit/dbn/beatit.h"
#include "beatit/dbn/calmdad.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace beatit::detail {

namespace {

struct DBNQualityMetrics {
    bool valid = false;
    double qpar = 0.0;
    double qkur = 0.0;
    double qmax = 0.0;
    std::size_t min_lag = 0;
    std::size_t max_lag = 0;
};

DBNQualityMetrics calculate_dbn_quality_metrics(const std::vector<float>& activation,
                                                double fps,
                                                float min_bpm,
                                                float max_bpm) {
    DBNQualityMetrics metrics;
    if (activation.size() < 16 || fps <= 0.0) {
        return metrics;
    }

    const double min_bpm_q = std::max(1.0, static_cast<double>(min_bpm));
    const double max_bpm_q = std::max(min_bpm_q + 1.0, static_cast<double>(max_bpm));
    metrics.min_lag =
        static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm_q)));
    const std::size_t max_lag =
        static_cast<std::size_t>(std::max<double>(metrics.min_lag + 1,
                                                  std::ceil((60.0 * fps) / min_bpm_q)));
    metrics.max_lag = std::min<std::size_t>(max_lag, activation.size() - 1);
    if (metrics.max_lag <= metrics.min_lag) {
        return metrics;
    }

    std::vector<double> salience;
    salience.reserve(metrics.max_lag - metrics.min_lag + 1);
    for (std::size_t lag = metrics.min_lag; lag <= metrics.max_lag; ++lag) {
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t i = lag; i < activation.size(); ++i) {
            sum += static_cast<double>(activation[i]) *
                   static_cast<double>(activation[i - lag]);
            ++count;
        }
        salience.push_back((count > 0) ? (sum / static_cast<double>(count)) : 0.0);
    }

    double mean = std::accumulate(salience.begin(), salience.end(), 0.0);
    mean /= static_cast<double>(salience.size());

    double variance = 0.0;
    for (double value : salience) {
        const double delta = value - mean;
        variance += delta * delta;
        metrics.qmax = std::max(metrics.qmax, value);
    }
    variance /= static_cast<double>(salience.size());

    const double rms = std::sqrt(variance + mean * mean);
    if (variance > 1e-12) {
        double m4 = 0.0;
        for (double value : salience) {
            const double delta = value - mean;
            m4 += delta * delta * delta * delta;
        }
        m4 /= static_cast<double>(salience.size());
        metrics.qkur = m4 / (variance * variance);
    }

    metrics.qpar = (rms > 1e-12) ? (metrics.qmax / rms) : 0.0;
    metrics.valid = true;
    return metrics;
}

struct DBNWindowSelection {
    std::size_t start = 0;
    std::size_t end = 0;
    bool use_window = false;
    bool selected_by_energy = false;
    std::vector<float> beat_slice;
    std::vector<float> downbeat_slice;
};

DBNWindowSelection select_dbn_processing_window(const CoreMLResult& result,
                                                const std::vector<float>* phase_energy,
                                                const BeatitConfig& config,
                                                double fps,
                                                float min_bpm,
                                                float max_bpm) {
    DBNWindowSelection selection;
    selection.end = result.beat_activation.size();

    const float window_peak_threshold =
        std::max(config.activation_threshold, config.dbn_activation_floor);
    std::pair<std::size_t, std::size_t> window{0, selection.end};
    const bool phase_energy_ok =
        phase_energy && !phase_energy->empty() && phase_energy->size() >= selection.end;

    if (phase_energy_ok) {
        window = detail::select_dbn_window_energy(*phase_energy,
                                                  config.dbn_window_seconds,
                                                  false,
                                                  fps);
        selection.selected_by_energy = true;
    } else {
        window = detail::select_dbn_window(result.beat_activation,
                                           config.dbn_window_seconds,
                                           true,
                                           min_bpm,
                                           max_bpm,
                                           window_peak_threshold,
                                           fps);
    }

    selection.start = window.first;
    selection.end = window.second;
    selection.use_window = (selection.start > 0 || selection.end < result.beat_activation.size());
    if (!selection.use_window) {
        return selection;
    }

    selection.beat_slice.assign(result.beat_activation.begin() + selection.start,
                                result.beat_activation.begin() + selection.end);
    if (!result.downbeat_activation.empty()) {
        selection.downbeat_slice.assign(result.downbeat_activation.begin() + selection.start,
                                        result.downbeat_activation.begin() + selection.end);
    }

    BEATIT_LOG_DEBUG("DBN window: start=" << selection.start
                     << " end=" << selection.end
                     << " frames=" << (selection.end - selection.start)
                     << " (" << ((selection.end - selection.start) / fps) << "s)"
                     << " selector="
                     << (selection.selected_by_energy ? "best-energy-phase" : "tempo")
                     << " energy=" << (selection.selected_by_energy ? "phase" : "beat"));
    return selection;
}

} // namespace

bool run_dbn_postprocess(const DBNRunRequest& request) {
    CoreMLResult& result = request.result;
    const std::vector<float>* phase_energy = request.phase_energy;
    const BeatitConfig& config = request.config;
    const double sample_rate = request.sample_rate;
    const float reference_bpm = request.reference_bpm;
    const std::size_t grid_total_frames = request.grid_total_frames;
    float min_bpm = request.min_bpm;
    float max_bpm = request.max_bpm;
    const double fps = request.fps;
    const double hop_scale = request.hop_scale;
    const std::size_t analysis_latency_frames = request.analysis_latency_frames;
    const double analysis_latency_frames_f = request.analysis_latency_frames_f;
    const double peaks_ms = request.peaks_ms;
    double& dbn_ms = request.dbn_ms;

    const std::size_t used_frames = result.beat_activation.size();
    if (used_frames == 0) {
        return false;
    }

    constexpr std::size_t kRefineWindow = 2;
    const float hard_min_bpm = std::max(1.0f, config.min_bpm);
    const float hard_max_bpm = std::max(hard_min_bpm + 1.0f, config.max_bpm);
    auto clamp_bpm_range = [&](float& min_value, float& max_value) {
        min_value = std::max(hard_min_bpm, min_value);
        max_value = std::min(hard_max_bpm, max_value);
        if (max_value <= min_value) {
            min_value = hard_min_bpm;
            max_value = hard_max_bpm;
        }
    };

    const DBNWindowSelection window_selection =
        select_dbn_processing_window(result, phase_energy, config, fps, min_bpm, max_bpm);
    const std::size_t window_start = window_selection.start;
    const bool use_window = window_selection.use_window;
    const std::vector<float>& beat_slice = window_selection.beat_slice;
    const std::vector<float>& downbeat_slice = window_selection.downbeat_slice;

    double quality_qpar = 0.0;
    double quality_qkur = 0.0;
    bool quality_valid = false;
    auto process_quality_gate = [&] {
        const std::vector<float>& quality_src =
            use_window ? beat_slice : result.beat_activation;
        const DBNQualityMetrics metrics =
            calculate_dbn_quality_metrics(quality_src, fps, config.min_bpm, config.max_bpm);
        quality_qpar = metrics.qpar;
        quality_qkur = metrics.qkur;
        quality_valid = metrics.valid;
        if (config.dbn_trace && metrics.valid) {
            BEATIT_LOG_DEBUG("DBN quality: qpar=" << quality_qpar
                             << " qmax=" << metrics.qmax
                             << " qkur=" << quality_qkur
                             << " lags=[" << metrics.min_lag << "," << metrics.max_lag << "]"
                             << " frames=" << quality_src.size());
        }
    };

    DBNDecodeResult decoded;
    const CalmdadDecoder calmdad_decoder(config);
    auto process_decode = [&] {
        if (config.dbn_mode == BeatitConfig::DBNMode::Calmdad) {
            if (config.dbn_tempo_prior_weight > 0.0f) {
                const double tolerance =
                    std::max(0.0, static_cast<double>(config.dbn_interval_tolerance));
                const double min_interval_frames =
                    std::max(1.0, (60.0 * fps) / std::max(1.0f, max_bpm)) * (1.0 - tolerance);
                const double max_interval_frames =
                    std::max(1.0, (60.0 * fps) / std::max(1.0f, min_bpm)) * (1.0 + tolerance);
                const std::size_t peak_min_interval =
                    static_cast<std::size_t>(std::max(1.0, std::floor(min_interval_frames)));
                const std::size_t peak_max_interval =
                    static_cast<std::size_t>(std::max<double>(peak_min_interval,
                                                              std::ceil(max_interval_frames)));
                const float peak_threshold =
                    std::max(config.activation_threshold, config.dbn_activation_floor);

                const std::vector<float>& prior_src =
                    use_window ? beat_slice : result.beat_activation;
                std::vector<std::size_t> prior_peaks =
                    pick_peaks(prior_src, peak_threshold, peak_min_interval, peak_max_interval);
                const double prior_interval = median_interval_frames(prior_peaks);
                if (prior_interval > 1.0) {
                    const double prior_bpm = (60.0 * fps) / prior_interval;
                    const double window_pct = config.tempo_window_percent > 0.0f
                        ? (static_cast<double>(config.tempo_window_percent) / 100.0)
                        : 0.10;
                    min_bpm = static_cast<float>(prior_bpm * (1.0 - window_pct));
                    max_bpm = static_cast<float>(prior_bpm * (1.0 + window_pct));
                    clamp_bpm_range(min_bpm, max_bpm);
                    BEATIT_LOG_DEBUG("DBN calmdad prior: bpm=" << prior_bpm
                                     << " peaks=" << prior_peaks.size()
                                     << " window_pct=" << window_pct
                                     << " clamp=[" << min_bpm << "," << max_bpm << "]");
                } else {
                    BEATIT_LOG_DEBUG("DBN calmdad prior: insufficient peaks for clamp");
                }
            }
                decoded = calmdad_decoder.decode({
                    use_window ? beat_slice : result.beat_activation,
                    use_window ? downbeat_slice : result.downbeat_activation,
                    fps,
                    min_bpm,
                    max_bpm,
                    config.dbn_bpm_step,
                });
        } else {
            decoded = decode_dbn_beats_beatit(use_window ? beat_slice : result.beat_activation,
                                              use_window ? downbeat_slice : result.downbeat_activation,
                                              fps,
                                              min_bpm,
                                              max_bpm,
                                              config,
                                              reference_bpm);
            }
    };

    if (fps > 0.0) {
        process_quality_gate();
    }

    const auto dbn_start = std::chrono::steady_clock::now();
    process_decode();
    const auto dbn_end = std::chrono::steady_clock::now();
    dbn_ms += std::chrono::duration<double, std::milli>(dbn_end - dbn_start).count();

    if (!decoded.beat_frames.empty()) {
        const DBNProcessingContext processing_ctx{
            config,
            calmdad_decoder,
            sample_rate,
            fps,
            hop_scale,
            analysis_latency_frames,
            analysis_latency_frames_f,
            kRefineWindow,
        };
        const DBNWindowContext window_ctx{
            used_frames,
            use_window,
            window_start,
            beat_slice,
            downbeat_slice,
        };
        const DBNBpmContext bpm_ctx{
            reference_bpm,
            grid_total_frames,
            min_bpm,
            max_bpm,
        };
        const DBNQualityContext quality_ctx{
            quality_valid,
            quality_qkur,
        };
        const DBNDecodedPostprocessContext decoded_context{
            processing_ctx,
            window_ctx,
            bpm_ctx,
            quality_ctx,
        };
        const bool ok = run_dbn_decoded_postprocess(result, decoded, decoded_context);
        if (ok && config.profile) {
            BEATIT_LOG_INFO("Timing(postprocess): dbn=" << dbn_ms
                            << "ms peaks=" << peaks_ms << "ms");
        }
        return ok;
    }

    return false;
}

} // namespace beatit::detail
