//
//  coreml_postprocess_dbn_decoded.h
//  BeatIt
//
//  Created by Till Toenshoff on 2026-02-22.
//  Copyright © 2026 Till Toenshoff. All rights reserved.
//

#ifndef BEATIT_COREML_POSTPROCESS_DBN_DECODED_H
#define BEATIT_COREML_POSTPROCESS_DBN_DECODED_H

#include "beatit/coreml.h"
#include "beatit/dbn_beatit.h"
#include "beatit/dbn_calmdad.h"

#include <cstddef>
#include <vector>

namespace beatit::detail {

bool run_dbn_decoded_postprocess(CoreMLResult& result,
                                 DBNDecodeResult& decoded,
                                 const CoreMLConfig& config,
                                 const CalmdadDecoder& calmdad_decoder,
                                 double sample_rate,
                                 float reference_bpm,
                                 std::size_t grid_total_frames,
                                 float min_bpm,
                                 float max_bpm,
                                 double fps,
                                 double hop_scale,
                                 std::size_t analysis_latency_frames,
                                 double analysis_latency_frames_f,
                                 std::size_t refine_window,
                                 std::size_t used_frames,
                                 bool use_window,
                                 std::size_t window_start,
                                 const std::vector<float>& beat_slice,
                                 const std::vector<float>& downbeat_slice,
                                 bool quality_valid,
                                 double quality_qpar,
                                 double quality_qkur);

} // namespace beatit::detail

#endif // BEATIT_COREML_POSTPROCESS_DBN_DECODED_H
