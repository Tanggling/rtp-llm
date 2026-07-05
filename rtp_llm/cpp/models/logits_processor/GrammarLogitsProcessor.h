#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <c10/core/Event.h>

#include "rtp_llm/cpp/engine_base/grammar/GrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

class GrammarLogitsProcessor final: public SpecLogitsProcessor {
public:
    explicit GrammarLogitsProcessor(std::shared_ptr<GrammarMatcher> matcher, int64_t eos_token_id = 0);

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
        return has_error_.load(std::memory_order_acquire);
    }
    ErrorInfo error() const override {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return error_info_;
    }

private:
    enum class RowState {
        Active,
        Finished,
        Terminated,
        Failed,
    };

    enum class DeviceMaskMode {
        UNSET,
        NOOP,
        MASK,
        TERMINATED,
        FINISHED,
    };

    struct DeviceMaskState {
        DeviceMaskMode              mode      = DeviceMaskMode::UNSET;
        int64_t                     token_len = -1;
        c10::Device                 device    = c10::Device(c10::DeviceType::CPU);
        torch::Tensor               vocab_mask;
        int32_t                     grammar_vocab_size = 0;
        std::shared_ptr<c10::Event> mask_ready;
    };

    void setError(ErrorCode code, std::string msg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!has_error_.load(std::memory_order_relaxed)) {
            error_info_ = ErrorInfo(code, std::move(msg));
            has_error_.store(true, std::memory_order_release);
        }
    }
    void setError(const ErrorInfo& info) {
        if (!info.hasError()) {
            return;
        }
        setError(info.code(), info.ToString());
    }

    bool    finished() const;
    bool    isTerminated() const;
    int64_t numAcceptedTokens() const;
    int32_t vocabSize() const;
    void    markFinished();
    void    rollback(int n);
    bool    acceptToken(int32_t token_id);

    DeviceMaskState buildDeviceMaskStateLocked(const c10::Device& device, ErrorInfo& out_err);
    void            publishMaskToDevice(DeviceMaskState& state, torch::Tensor vocab_mask, const c10::Device& device);
    void            applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state);
    void            applyMaskLocked(const torch::Tensor& logits, ErrorInfo& out_err);
    void            acceptCommittedLocked(const int32_t* tokens, size_t n, ErrorInfo& out_err);

    ErrorInfo preflightSpecRequest(const SpecLogitsProcessorRequest& request) const;
    ErrorInfo validateMatcherInvariantsLocked(const SpecLogitsProcessorRequest& request);
    RowState  fillGrammarRowLocked(int32_t* row, size_t W, size_t model_vocab_size);
    int       runSpecVerifyGuarded(int32_t*    bitmask_cpu_out,
                                   size_t      W,
                                   const char* who,
                                   const std::function<int(int& grammar_accepted_prefix, ErrorInfo& walk_err)>& walk,
                                   ErrorInfo&                                                                   out_err);
    int       runSpecVerifyLocked(const SpecLogitsProcessorRequest& request, ErrorInfo& out_err);

    static void forceToken(const torch::Tensor& logits, int64_t token_id);

    std::shared_ptr<GrammarMatcher> matcher_;
    int64_t                         eos_token_id_       = 0;
    int64_t                         accepted_token_len_ = 0;
    std::optional<c10::Device>      last_mask_device_;
    DeviceMaskState                 device_mask_state_{};
    torch::Tensor                   reusable_bitmask_cpu_;
    torch::Tensor                   reusable_vocab_mask_cpu_;
    int32_t                         reusable_mask_words_ = 0;

    mutable std::mutex state_mutex_;
    std::atomic<bool>  has_error_{false};
    ErrorInfo          error_info_;
};

}  // namespace rtp_llm
