//
//  dbn_run.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/post/dbn_run.h"

#include "beatit/post/dbn_apply.h"
#include "beatit/post/window.h"
#include "beatit/logging.hpp"
#include "beatit/dbn/beatit.h"
#include "beatit/dbn/calmdad.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
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

DBNQualityMetrics evaluate_dbn_quality(const CoreMLResult& result,
                                       const DBNWindowSelection& window_selection,
                                       const BeatitConfig& config,
                                       double fps) {
    const std::vector<float>& activation =
        window_selection.use_window ? window_selection.beat_slice : result.beat_activation;
    const DBNQualityMetrics metrics =
        calculate_dbn_quality_metrics(activation, fps, config.min_bpm, config.max_bpm);

    if (config.dbn_trace && metrics.valid) {
        BEATIT_LOG_DEBUG("DBN quality: qpar=" << metrics.qpar
                         << " qmax=" << metrics.qmax
                         << " qkur=" << metrics.qkur
                         << " lags=[" << metrics.min_lag << "," << metrics.max_lag << "]"
                         << " frames=" << activation.size());
    }

    return metrics;
}

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

struct DBNDecodeOutcome {
    DBNDecodeResult decoded;
    float min_bpm = 0.0f;
    float max_bpm = 0.0f;
};

DBNDecodeOutcome run_dbn_decode(const CoreMLResult& result,
                                const DBNWindowSelection& window_selection,
                                const BeatitConfig& config,
                                const CalmdadDecoder& calmdad_decoder,
                                double fps,
                                float min_bpm,
                                float max_bpm) {
    DBNDecodeOutcome outcome;
    outcome.min_bpm = min_bpm;
    outcome.max_bpm = max_bpm;

    const std::vector<float>& beat_activation =
        window_selection.use_window ? window_selection.beat_slice : result.beat_activation;
    const std::vector<float>& downbeat_activation =
        window_selection.use_window ? window_selection.downbeat_slice : result.downbeat_activation;

    if (config.dbn_mode == BeatitConfig::DBNMode::Calmdad) {
        outcome.decoded = calmdad_decoder.decode({
            beat_activation,
            downbeat_activation,
            fps,
            outcome.min_bpm,
            outcome.max_bpm,
            config.dbn_bpm_step,
        });
        return outcome;
    }

    outcome.decoded = decode_dbn_beats_beatit(beat_activation,
                                              downbeat_activation,
                                              fps,
                                              outcome.min_bpm,
                                              outcome.max_bpm,
                                              config);
    return outcome;
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

    const DBNWindowSelection window_selection =
        select_dbn_processing_window(result, phase_energy, config, fps, min_bpm, max_bpm);
    const std::size_t window_start = window_selection.start;
    const bool use_window = window_selection.use_window;
    const std::vector<float>& beat_slice = window_selection.beat_slice;
    const std::vector<float>& downbeat_slice = window_selection.downbeat_slice;

    const CalmdadDecoder calmdad_decoder(config);
    const DBNQualityMetrics quality_metrics =
        (fps > 0.0) ? evaluate_dbn_quality(result, window_selection, config, fps)
                    : DBNQualityMetrics{};

    const auto dbn_start = std::chrono::steady_clock::now();
    DBNDecodeOutcome decode_outcome = run_dbn_decode(result,
                                                     window_selection,
                                                     config,
                                                     calmdad_decoder,
                                                     fps,
                                                     min_bpm,
                                                     max_bpm);
    const auto dbn_end = std::chrono::steady_clock::now();
    dbn_ms += std::chrono::duration<double, std::milli>(dbn_end - dbn_start).count();
    DBNDecodeResult& decoded = decode_outcome.decoded;
    min_bpm = decode_outcome.min_bpm;
    max_bpm = decode_outcome.max_bpm;

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
            quality_metrics.valid,
            quality_metrics.qkur,
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
