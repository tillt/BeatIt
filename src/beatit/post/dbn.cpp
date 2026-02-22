//
//  dbn.cpp
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#include "beatit/pp_dbn.h"

#include "beatit/pp_dbn_decode.h"
#include "beatit/pp_helpers.h"
#include "beatit/pp_result_ops.h"
#include "beatit/pp_tempo_fit.h"
#include "beatit/pp_window.h"
#include "beatit/logging.hpp"
#include "beatit/dbn_beatit.h"
#include "beatit/dbn_calmdad.h"

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

bool run_dbn_postprocess(CoreMLResult& result,
                         const std::vector<float>* phase_energy,
                         const CoreMLConfig& config,
                         double sample_rate,
                         float reference_bpm,
                         std::size_t grid_total_frames,
                         float min_bpm,
                         float max_bpm,
                         double fps,
                         double hop_scale,
                         std::size_t analysis_latency_frames,
                         double analysis_latency_frames_f,
                         double& dbn_ms,
                         double peaks_ms) {
    const std::size_t used_frames = result.beat_activation.size();
    if (used_frames == 0) {
        return false;
    }

    constexpr std::size_t kRefineWindow = 2;
    const float hard_min_bpm = std::max(1.0f, config.min_bpm);
    const float hard_max_bpm = std::max(hard_min_bpm + 1.0f, config.max_bpm);
    auto clamp_bpm_range = [&](float* min_value, float* max_value) {
        *min_value = std::max(hard_min_bpm, *min_value);
        *max_value = std::min(hard_max_bpm, *max_value);
        if (*max_value <= *min_value) {
            *min_value = hard_min_bpm;
            *max_value = hard_max_bpm;
        }
    };

    auto select_dbn_window = [&](const std::vector<float>& activation,
                                 double window_seconds,
                                 bool intro_mid_outro,
                                 float local_min_bpm,
                                 float local_max_bpm,
                                 float peak_threshold) -> std::pair<std::size_t, std::size_t> {
        return detail::select_dbn_window(activation,
                                         window_seconds,
                                         intro_mid_outro,
                                         local_min_bpm,
                                         local_max_bpm,
                                         peak_threshold,
                                         fps);
    };

    auto select_dbn_window_energy = [&](const std::vector<float>& energy,
                                        double window_seconds,
                                        bool intro_mid_outro) -> std::pair<std::size_t, std::size_t> {
        return detail::select_dbn_window_energy(energy,
                                                window_seconds,
                                                intro_mid_outro,
                                                fps);
    };

    std::size_t window_start = 0;
    std::size_t window_end = used_frames;
    bool use_window = false;
    bool window_energy = false;
    std::vector<float> beat_slice;
    std::vector<float> downbeat_slice;

    auto process_window_selection = [&] {
        const float window_peak_threshold =
            std::max(config.activation_threshold, config.dbn_activation_floor);
        std::pair<std::size_t, std::size_t> window{0, used_frames};
        bool selected_by_energy = false;
        const bool phase_energy_ok =
            phase_energy && !phase_energy->empty() && phase_energy->size() >= used_frames;
        if (phase_energy_ok) {
            window = select_dbn_window_energy(*phase_energy,
                                              config.dbn_window_seconds,
                                              false);
            selected_by_energy = true;
        } else {
            window = select_dbn_window(result.beat_activation,
                                       config.dbn_window_seconds,
                                       true,
                                       min_bpm,
                                       max_bpm,
                                       window_peak_threshold);
        }
        window_energy = selected_by_energy;
        window_start = window.first;
        window_end = window.second;
        use_window = (window_start > 0 || window_end < used_frames);
        std::vector<float> local_beat_slice;
        std::vector<float> local_downbeat_slice;
        if (use_window) {
            local_beat_slice.assign(result.beat_activation.begin() + window_start,
                                    result.beat_activation.begin() + window_end);
            if (!result.downbeat_activation.empty()) {
                local_downbeat_slice.assign(result.downbeat_activation.begin() + window_start,
                                            result.downbeat_activation.begin() + window_end);
            }
            if (config.verbose) {
                BEATIT_LOG_DEBUG("DBN window: start=" << window_start
                                 << " end=" << window_end
                                 << " frames=" << (window_end - window_start)
                                 << " (" << ((window_end - window_start) / fps) << "s)"
                                 << " selector="
                                 << (window_energy ? "best-energy-phase" : "tempo")
                                 << " energy=" << (window_energy ? "phase" : "beat"));
            }
        }
        beat_slice = std::move(local_beat_slice);
        downbeat_slice = std::move(local_downbeat_slice);
    };

    double quality_qpar = 0.0;
    double quality_qmax = 0.0;
    double quality_qkur = 0.0;
    bool quality_valid = false;
    auto process_quality_gate = [&] {
        const std::vector<float>& quality_src =
            use_window ? beat_slice : result.beat_activation;
        if (quality_src.size() >= 16) {
                const double min_bpm_q = std::max(1.0, static_cast<double>(config.min_bpm));
                const double max_bpm_q = std::max(min_bpm_q + 1.0,
                                                  static_cast<double>(config.max_bpm));
                const std::size_t min_lag =
                    static_cast<std::size_t>(std::max(1.0, std::floor((60.0 * fps) / max_bpm_q)));
                const std::size_t max_lag =
                    static_cast<std::size_t>(std::max<double>(min_lag + 1,
                                                              std::ceil((60.0 * fps) / min_bpm_q)));
                const std::size_t max_lag_clamped =
                    std::min<std::size_t>(max_lag, quality_src.size() - 1);
                if (max_lag_clamped > min_lag) {
                    std::vector<double> salience;
                    salience.reserve(max_lag_clamped - min_lag + 1);
                    for (std::size_t lag = min_lag; lag <= max_lag_clamped; ++lag) {
                        double sum = 0.0;
                        std::size_t count = 0;
                        for (std::size_t i = lag; i < quality_src.size(); ++i) {
                            sum += static_cast<double>(quality_src[i]) *
                                   static_cast<double>(quality_src[i - lag]);
                            ++count;
                        }
                        const double value = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
                        salience.push_back(value);
                    }
                    double mean = 0.0;
                    for (double v : salience) {
                        mean += v;
                    }
                    mean /= static_cast<double>(salience.size());
                    double var = 0.0;
                    for (double v : salience) {
                        const double d = v - mean;
                        var += d * d;
                    }
                    var /= static_cast<double>(salience.size());
                    const double rms = std::sqrt(var + mean * mean);
                    double max_val = 0.0;
                    for (double v : salience) {
                        if (v > max_val) {
                            max_val = v;
                        }
                    }
                    double kurtosis = 0.0;
                    if (var > 1e-12) {
                        double m4 = 0.0;
                        for (double v : salience) {
                            const double d = v - mean;
                            m4 += d * d * d * d;
                        }
                        m4 /= static_cast<double>(salience.size());
                        kurtosis = m4 / (var * var);
                    }
                    quality_qpar = (rms > 1e-12) ? (max_val / rms) : 0.0;
                    quality_qmax = max_val;
                    quality_qkur = kurtosis;
                    quality_valid = true;
                    if (config.dbn_trace) {
                        BEATIT_LOG_DEBUG("DBN quality: qpar=" << quality_qpar
                                         << " qmax=" << quality_qmax
                                         << " qkur=" << quality_qkur
                                         << " lags=[" << min_lag << "," << max_lag_clamped << "]"
                                         << " frames=" << quality_src.size());
                    }
                }
        }
    };

    DBNDecodeResult decoded;
    const CalmdadDecoder calmdad_decoder(config);
    auto process_decode = [&] {
        if (config.dbn_mode == CoreMLConfig::DBNMode::Calmdad) {
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
                    clamp_bpm_range(&min_bpm, &max_bpm);
                    if (config.verbose) {
                        BEATIT_LOG_DEBUG("DBN calmdad prior: bpm=" << prior_bpm
                                         << " peaks=" << prior_peaks.size()
                                         << " window_pct=" << window_pct
                                         << " clamp=[" << min_bpm << "," << max_bpm << "]");
                    }
                } else if (config.verbose) {
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

    process_window_selection();

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
            quality_qpar,
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
