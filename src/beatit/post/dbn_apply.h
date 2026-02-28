//
//  dbn_apply.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#ifndef BEATIT_POST_DBN_APPLY_H
#define BEATIT_POST_DBN_APPLY_H

#include "beatit/config.h"
#include "beatit/dbn/beatit.h"
#include "beatit/dbn/calmdad.h"

#include <cstddef>
#include <vector>

namespace beatit::detail {

struct DBNProcessingContext {
    const BeatitConfig& config;
    const CalmdadDecoder& calmdad_decoder;
    double sample_rate = 0.0;
    double fps = 0.0;
    double hop_scale = 1.0;
    std::size_t analysis_latency_frames = 0;
    double analysis_latency_frames_f = 0.0;
    std::size_t refine_window = 0;
};

struct DBNWindowContext {
    std::size_t used_frames = 0;
    bool use_window = false;
    std::size_t window_start = 0;
    const std::vector<float>& beat_slice;
    const std::vector<float>& downbeat_slice;
};

struct DBNBpmContext {
    float reference_bpm = 0.0f;
    std::size_t grid_total_frames = 0;
    float min_bpm = 0.0f;
    float max_bpm = 0.0f;
};

struct DBNQualityContext {
    bool valid = false;
    double qkur = 0.0;
};

struct DBNDecodedPostprocessContext {
    DBNProcessingContext processing;
    DBNWindowContext window;
    DBNBpmContext bpm;
    DBNQualityContext quality;
};

bool run_dbn_decoded_postprocess(CoreMLResult& result,
                                 DBNDecodeResult& decoded,
                                 const DBNDecodedPostprocessContext& context);

} // namespace beatit::detail

#endif // BEATIT_POST_DBN_APPLY_H
