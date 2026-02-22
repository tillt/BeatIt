#ifndef BEATIT_DBN_BEATIT_H
#define BEATIT_DBN_BEATIT_H

#include "beatit/coreml.h"
#include "beatit/dbn_calmdad.h"

#include <vector>

namespace beatit {

DBNDecodeResult decode_dbn_beats_beatit(const std::vector<float>& beat_activation,
                                        const std::vector<float>& downbeat_activation,
                                        double fps,
                                        float min_bpm,
                                        float max_bpm,
                                        const CoreMLConfig& config,
                                        float reference_bpm);

}  // namespace beatit

#endif  // BEATIT_DBN_BEATIT_H
