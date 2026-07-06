#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

class RtpGrammarMatcher;

class ProcessorErrorState {
public:
    bool hasError() const {
        return has_error_.load(std::memory_order_acquire);
    }

    ErrorInfo error() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_info_;
    }

    void set(const ErrorInfo& info) {
        if (!info.hasError()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_error_.load(std::memory_order_relaxed)) {
            error_info_ = info;
            has_error_.store(true, std::memory_order_release);
        }
    }

    void set(ErrorCode code, std::string msg) {
        set(ErrorInfo(code, std::move(msg)));
    }

private:
    mutable std::mutex mutex_;
    std::atomic<bool>  has_error_{false};
    ErrorInfo          error_info_;
};

class GrammarLogitsProcessor final: public SpecLogitsProcessor {
public:
    explicit GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher, int64_t eos_token_id = 0);

    ~GrammarLogitsProcessor() override;

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    void updateMultiSeqStatus(const std::vector<int>& /*src_batch_indices*/) override {}

    bool isStateful() const override {
        return true;
    }

    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;

    int64_t committedOutputLen() const override {
        return accepted_token_len_;
    }

    bool hasError() const override {
        return error_state_.hasError();
    }
    ErrorInfo error() const override {
        return error_state_.error();
    }

private:
    class DecodeMaskBuilder;

    void setError(ErrorCode code, std::string msg) {
        error_state_.set(code, std::move(msg));
    }
    void setError(const ErrorInfo& info) {
        error_state_.set(info);
    }

    ErrorInfo acceptCommittedLocked(const int32_t* tokens, size_t n);

    std::shared_ptr<RtpGrammarMatcher> matcher_;
    int64_t                            eos_token_id_       = 0;
    int64_t                            accepted_token_len_ = 0;
    std::unique_ptr<DecodeMaskBuilder> decode_mask_builder_;

    mutable std::mutex  state_mutex_;
    ProcessorErrorState error_state_;
};

}  // namespace rtp_llm
