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

class SpecLogitsProcessor;

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor() = default;
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
    virtual void                     updateMultiSeqStatus(const std::vector<int>& src_batch_indices)           = 0;

    // Normal decode lifecycle callback. Processors that also participate in speculative
    // verification or require exactly-once speculative commits expose the corresponding
    // optional facet interfaces registered in LogitsProcessors.
    virtual std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) = 0;

    void          memFill(const torch::Tensor& new_tokens_logits, size_t vocab_size, size_t index);
    void          maskLogits(torch::Tensor& new_token_logits, const torch::Tensor& vocab_mask);
    torch::Tensor generateVocabMask(size_t                                  batch_size,
                                    size_t                                  vocab_size,
                                    const std::vector<std::vector<size_t>>& batch_candidate_token_ids);
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

// Optional facet for processors that understand all score rows belonging to one
// speculative stream. Implementing BaseLogitsProcessor::process() alone does not
// imply that the processor can safely consume speculative score rows.
class ScoreBatchLogitsProcessor {
public:
    virtual ~ScoreBatchLogitsProcessor() = default;

    virtual std::optional<ErrorInfo>
    processScoreBatch(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
};

using ScoreBatchLogitsProcessorPtr = std::shared_ptr<ScoreBatchLogitsProcessor>;

// Optional facet for processors whose committed state must be advanced on every
// token emitted by speculative decoding. Normal decode continues to use the
// BaseLogitsProcessor::updateStatus() lifecycle callback.
class StatefulLogitsProcessor {
public:
    virtual ~StatefulLogitsProcessor() = default;

    virtual std::optional<ErrorInfo> commitTokens(const torch::Tensor& new_tokens, int32_t num_new_tokens) = 0;
    virtual int64_t                  committedOutputLen() const                                            = 0;
};

using StatefulLogitsProcessorPtr = std::shared_ptr<StatefulLogitsProcessor>;

// Processors installed on one stream, indexed by their orthogonal execution
// facets so callers do not need role switches or dynamic casts.
class LogitsProcessors {
public:
    void add(BaseLogitsProcessorPtr               normal,
             ScoreBatchLogitsProcessorPtr         score_batch = nullptr,
             std::shared_ptr<SpecLogitsProcessor> spec        = nullptr,
             StatefulLogitsProcessorPtr           stateful    = nullptr);

    const std::vector<BaseLogitsProcessorPtr>& normalProcessors() const {
        return normal_processors_;
    }

    const std::vector<ScoreBatchLogitsProcessorPtr>& scoreBatchProcessors() const {
        return score_batch_processors_;
    }

    const std::vector<std::shared_ptr<SpecLogitsProcessor>>& specProcessors() const {
        return spec_processors_;
    }

    const std::vector<StatefulLogitsProcessorPtr>& statefulProcessors() const {
        return stateful_processors_;
    }

    bool mtpCompatible() const {
        return mtp_incompatible_processors_.empty();
    }

private:
    std::vector<BaseLogitsProcessorPtr>                 normal_processors_;
    std::vector<ScoreBatchLogitsProcessorPtr>           score_batch_processors_;
    std::vector<std::shared_ptr<SpecLogitsProcessor>>   spec_processors_;
    std::vector<StatefulLogitsProcessorPtr>             stateful_processors_;
    std::vector<BaseLogitsProcessorPtr>                 mtp_incompatible_processors_;
};

}  // namespace rtp_llm
