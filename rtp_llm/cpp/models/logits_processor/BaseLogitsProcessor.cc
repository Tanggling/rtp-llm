#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "ATen/cuda/CUDAContext.h"
#endif

using namespace std;

namespace rtp_llm {

const float BaseLogitsProcessor::neg_inf = -std::numeric_limits<float>::max();

void LogitsProcessors::add(BaseLogitsProcessorPtr               normal,
                           ScoreBatchLogitsProcessorPtr         score_batch,
                           std::shared_ptr<SpecLogitsProcessor> spec,
                           StatefulLogitsProcessorPtr           stateful) {
    RTP_LLM_CHECK_WITH_INFO(normal != nullptr, "logits processor requires a normal processor facet");

    const bool has_mtp_facet = score_batch != nullptr || spec != nullptr;
    normal_processors_.push_back(normal);
    if (score_batch) {
        score_batch_processors_.push_back(std::move(score_batch));
    }
    if (spec) {
        spec_processors_.push_back(std::move(spec));
    }
    if (stateful) {
        stateful_processors_.push_back(std::move(stateful));
    }
    if (!has_mtp_facet) {
        mtp_incompatible_processors_.push_back(std::move(normal));
    }
}

void BaseLogitsProcessor::memFill(const torch::Tensor& new_tokens_logits, size_t vocab_size, size_t index) {
    RTP_LLM_CHECK(new_tokens_logits.dim() == 1);
    auto tensor = new_tokens_logits;
    tensor.fill_(neg_inf);
    tensor[index] = 1;
}

torch::Tensor BaseLogitsProcessor::generateVocabMask(
    size_t batch_size, size_t vocab_size, const std::vector<std::vector<size_t>>& batch_candidate_token_ids) {
    RTP_LLM_CHECK(batch_candidate_token_ids.size() == batch_size);
    auto vocab_mask_cpu = torch::ones({(int64_t)batch_size, (int64_t)vocab_size}, torch::kUInt8);
    auto vocab_mask_ptr = vocab_mask_cpu.data_ptr<uint8_t>();

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto& candidate_token_ids = batch_candidate_token_ids[batch_idx];
        for (const auto& token_id : candidate_token_ids) {
            if (token_id < vocab_size) {
                vocab_mask_ptr[batch_idx * vocab_size + token_id] = 0;
            }
        }
    }

    return vocab_mask_cpu.to(torch::kCUDA);
}
void BaseLogitsProcessor::maskLogits(torch::Tensor& new_tokens_logits, const torch::Tensor& vocab_mask) {
    RTP_LLM_CHECK(new_tokens_logits.dim() == 2);
    RTP_LLM_CHECK(vocab_mask.dim() == 2);
    RTP_LLM_CHECK(new_tokens_logits.size(0) == vocab_mask.size(0));
    RTP_LLM_CHECK(new_tokens_logits.size(1) == vocab_mask.size(1));
#if USING_CUDA
    cudaMaskLogits(new_tokens_logits, vocab_mask, at::cuda::getCurrentCUDAStream().stream());
#else
    new_tokens_logits.masked_fill_(vocab_mask.to(torch::kBool), neg_inf);
#endif
}

}  // namespace rtp_llm
