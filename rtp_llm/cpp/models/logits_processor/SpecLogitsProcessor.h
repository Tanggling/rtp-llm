#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

struct SpecLogitsProcessorRequest {
    const int32_t* draft_tokens = nullptr;
    int            propose_step = 0;

    // Packed bitmask rows, shape [(propose_step + 1), bitmask_size_int32].
    // bit=1 means allowed; bit=0 means masked.
    int32_t* bitmask_cpu_out    = nullptr;
    size_t   bitmask_size_int32 = 0;
    size_t   vocab_size         = 0;
};

class SpecLogitsProcessor: public BaseLogitsProcessor {
public:
    ~SpecLogitsProcessor() override = default;

    // Keep score-batch classification tied to the same eligibility predicate used by
    // SpecLogitsVerifyRunner, so implementations cannot diverge from the verify path.
    ScoreBatchRole scoreBatchRole() const final override {
        return isSpecVerifyEligible() ? ScoreBatchRole::kSpecVerify : ScoreBatchRole::kNormalDecodeOnly;
    }

    virtual bool isSpecVerifyEligible() const = 0;

    // Returns accept cap in [0, propose_step]; on failure stashes ErrorInfo and returns 0.
    virtual int tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) = 0;

    static size_t bitmaskWordCount(size_t vocab_size) {
        return (vocab_size + 31) / 32;
    }

    static constexpr int32_t kBitmaskAllowAll = -1;
};

using SpecLogitsProcessorPtr = std::shared_ptr<SpecLogitsProcessor>;

}  // namespace rtp_llm
