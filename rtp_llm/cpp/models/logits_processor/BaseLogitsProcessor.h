#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

enum class ScoreBatchRole {
    kNormalDecodeOnly,   // process() is only defined for normal one-step decode.
    kScoreBatchProcess,  // process() can be expanded per score_batch row.
    kSpecVerify,         // mask via spec verify; skip gatherer process() path.
};

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor() = default;
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
    virtual void                     updateMultiSeqStatus(const std::vector<int>& src_batch_indices)           = 0;

    virtual bool isStateful() const {
        return false;
    }
    virtual ScoreBatchRole scoreBatchRole() const {
        return ScoreBatchRole::kScoreBatchProcess;
    }
    // Number of committed *output* tokens from this processor's point of view.
    // MUST stay aligned with GenerateStream::outputTokenLen() (seqLength - inputLength);
    // GenerateStream::validateStatefulLogitsProcessorState() compares the two and errors on
    // mismatch. NOTE: this is the stream output length, NOT a grammar matcher's accepted-token
    // count — e.g. ReasoningGrammar counts think-phase tokens here even though they never enter
    // the grammar matcher.
    virtual int64_t committedOutputLen() const {
        return 0;
    }

    // Called once per committed token under GenerateStream::mutex_; spec/disagg paths included.
    virtual std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) = 0;

    void          memFill(const torch::Tensor& new_tokens_logits, size_t vocab_size, size_t index);
    void          maskLogits(torch::Tensor& new_token_logits, const torch::Tensor& vocab_mask);
    torch::Tensor generateVocabMask(size_t                                  batch_size,
                                    size_t                                  vocab_size,
                                    const std::vector<std::vector<size_t>>& batch_candidate_token_ids);
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

}  // namespace rtp_llm
