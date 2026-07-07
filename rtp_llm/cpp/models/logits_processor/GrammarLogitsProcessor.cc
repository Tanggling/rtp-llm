#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include <dlpack/dlpack.h>
#include <c10/core/Event.h>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

enum class SpecVerifyRowState {
    Active,
    Finished,
    Terminated,
};

ErrorInfo preflightSpecVerifyRequest(const SpecLogitsProcessorRequest& request) {
    if (request.bitmask_size_int32 < SpecLogitsProcessor::bitmaskWordCount(request.vocab_size)) {
        return ErrorInfo(ErrorCode::GRAMMAR_BITMASK_BUFFER_TOO_SMALL,
                         "grammar MTP verify: bitmask buffer smaller than model vocab (words="
                             + std::to_string(request.bitmask_size_int32)
                             + ", vocab=" + std::to_string(request.vocab_size) + ")");
    }
    return {};
}

ErrorInfo validateSpecVerifyMatcher(RtpGrammarMatcher& matcher, int64_t eos_token_id, size_t W) {
    auto grammar_vocab_size_or = matcher.vocabSize();
    if (!grammar_vocab_size_or.ok()) {
        matcher.markFinished();
        return grammar_vocab_size_or.status();
    }
    const int32_t grammar_vocab_size = grammar_vocab_size_or.value();
    if (grammar_vocab_size <= 0) {
        matcher.markFinished();
        return ErrorInfo(ErrorCode::INVALID_PARAMS,
                         "grammar MTP verify: invalid grammar vocab size " + std::to_string(grammar_vocab_size));
    }
    if (SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size) > W) {
        matcher.markFinished();
        return ErrorInfo(ErrorCode::GRAMMAR_VOCAB_EXCEEDS_MODEL_VOCAB,
                         "grammar vocab exceeds model vocab in MTP verify (grammar="
                             + std::to_string(grammar_vocab_size) + ", model_words=" + std::to_string(W) + ")");
    }

    auto token_in_range = [W](int64_t t) { return t >= 0 && static_cast<size_t>(t / 32) < W; };
    if (!token_in_range(eos_token_id)) {
        matcher.markFinished();
        return ErrorInfo(ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                         "grammar MTP verify: eos_token_id (" + std::to_string(eos_token_id)
                             + ") out of model vocab bitmask (words=" + std::to_string(W) + ")");
    }
    return {};
}

ErrorResult<SpecVerifyRowState>
failSpecVerifyRow(RtpGrammarMatcher& matcher, int32_t* row, size_t W, int64_t eos_token_id, const ErrorInfo& error) {
    matcher.markFinished();
    forceTokenInBitmask(row, W, eos_token_id);
    return error;
}

ErrorResult<SpecVerifyRowState>
fillSpecVerifyRow(RtpGrammarMatcher& matcher, int64_t eos_token_id, int32_t* row, size_t W, size_t model_vocab_size) {
    std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
    if (matcher.finished()) {
        forceTokenInBitmask(row, W, eos_token_id);
        return SpecVerifyRowState::Finished;
    }
    auto terminated = matcher.isTerminated();
    if (!terminated.ok()) {
        return failSpecVerifyRow(matcher, row, W, eos_token_id, terminated.status());
    }
    if (terminated.value()) {
        forceTokenInBitmask(row, W, eos_token_id);
        return SpecVerifyRowState::Terminated;
    }

    auto grammar_vocab_size_or = matcher.vocabSize();
    if (!grammar_vocab_size_or.ok()) {
        return failSpecVerifyRow(matcher, row, W, eos_token_id, grammar_vocab_size_or.status());
    }
    const int32_t grammar_vocab_size = grammar_vocab_size_or.value();
    const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);

    int64_t  dl_shape[2];
    DLTensor dl     = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
    auto     filled = matcher.fillBitmask(&dl, 0);
    if (!filled.ok()) {
        return failSpecVerifyRow(matcher, row, W, eos_token_id, filled.status());
    }
    if (!filled.value()) {
        return failSpecVerifyRow(matcher,
                                 row,
                                 W,
                                 eos_token_id,
                                 ErrorInfo(ErrorCode::GRAMMAR_FILL_BITMASK_FAILED,
                                           "grammar MTP verify: matcher fillBitmask failed; matcher state corrupted"));
    }
    clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(model_vocab_size));
    return SpecVerifyRowState::Active;
}

bool specVerifyRowCanConsumeDraft(SpecVerifyRowState row_state) {
    return row_state == SpecVerifyRowState::Active;
}

bool specVerifyDraftTokenAllowed(const int32_t* row, size_t W, size_t vocab_size, int32_t token) {
    return token >= 0 && static_cast<size_t>(token) < vocab_size && bitmaskAllowsToken(row, W, token);
}

class ProvisionalSpecAcceptGuard {
public:
    explicit ProvisionalSpecAcceptGuard(RtpGrammarMatcher& matcher): matcher_(matcher) {}

    void recordAccepted() {
        ++accepted_prefix_;
    }

    ErrorInfo rollbackAndReport(int32_t* fallback_row, size_t W, int64_t eos_token_id) {
        auto rollback_err = rollback();
        if (!rollback_err.hasError()) {
            return ErrorInfo::OkStatus();
        }
        matcher_.markFinished();
        forceTokenInBitmask(fallback_row, W, eos_token_id);
        return rollback_err;
    }

private:
    ErrorInfo rollback() {
        if (accepted_prefix_ > 0) {
            return matcher_.rollback(accepted_prefix_);
        }
        return ErrorInfo::OkStatus();
    }

    RtpGrammarMatcher& matcher_;
    int                accepted_prefix_ = 0;
};

[[nodiscard]] ErrorResult<int> verifyDraftPrefixAndFillBitmask(RtpGrammarMatcher&                matcher,
                                                               int64_t                           eos_token_id,
                                                               const SpecLogitsProcessorRequest& request,
                                                               ProvisionalSpecAcceptGuard&       provisional) {
    const int  P = request.propose_step;
    const auto W = request.bitmask_size_int32;

    for (int offset = 0; offset <= P; ++offset) {
        int32_t*   row       = request.bitmask_cpu_out + offset * W;
        const auto row_state = fillSpecVerifyRow(matcher, eos_token_id, row, W, request.vocab_size);
        if (!row_state.ok()) {
            return row_state.status();
        }
        if (offset == P) {
            return int(P);
        }
        if (!specVerifyRowCanConsumeDraft(row_state.value())) {
            return int(offset);
        }

        const int32_t draft_token = request.draft_tokens[offset];
        if (!specVerifyDraftTokenAllowed(row, W, request.vocab_size, draft_token)) {
            return int(offset);
        }
        auto accepted = matcher.acceptToken(draft_token);
        if (!accepted.ok()) {
            return accepted.status();
        }
        if (!accepted.value()) {
            return int(offset);
        }
        provisional.recordAccepted();
    }

    return int(P);
}

ErrorResult<int> verifySpecDraftAndFillBitmask(RtpGrammarMatcher&                matcher,
                                               int64_t                           eos_token_id,
                                               const SpecLogitsProcessorRequest& request) {
    if (auto err = preflightSpecVerifyRequest(request); err.hasError()) {
        return err;
    }

    const auto W = request.bitmask_size_int32;

    if (auto err = validateSpecVerifyMatcher(matcher, eos_token_id, W); err.hasError()) {
        return err;
    }

    ProvisionalSpecAcceptGuard provisional(matcher);

    auto cap = verifyDraftPrefixAndFillBitmask(matcher, eos_token_id, request, provisional);
    if (!cap.ok()) {
        matcher.markFinished();
        forceTokenInBitmask(request.bitmask_cpu_out, W, eos_token_id);
        return cap.status();
    }

    auto rollback_err = provisional.rollbackAndReport(request.bitmask_cpu_out, W, eos_token_id);
    if (rollback_err.hasError()) {
        return rollback_err;
    }
    return int(cap.value());
}

}  // namespace

class GrammarLogitsProcessor::DecodeMaskBuilder final {
public:
    ErrorInfo
    apply(const torch::Tensor& logits, RtpGrammarMatcher& matcher, int64_t accepted_token_len, int64_t eos_token_id) {
        last_mask_device_ = logits.device();

        if (device_mask_state_.mode != DeviceMaskMode::UNSET && device_mask_state_.token_len == accepted_token_len
            && device_mask_state_.device == logits.device()) {
            applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
            return {};
        }

        auto state_or = buildState(logits.device(), matcher, accepted_token_len);
        if (!state_or.ok()) {
            device_mask_state_ = finishedState(logits.device(), accepted_token_len);
            applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
            return state_or.status();
        }

        device_mask_state_ = std::move(state_or.value());
        applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
        return {};
    }

    ErrorInfo refreshAfterCommit(RtpGrammarMatcher& matcher, int64_t accepted_token_len) {
        auto device   = last_mask_device_.value_or(c10::Device(c10::DeviceType::CPU));
        auto state_or = buildState(device, matcher, accepted_token_len);
        if (!state_or.ok()) {
            device_mask_state_ = finishedState(device, accepted_token_len);
            return state_or.status();
        }

        device_mask_state_ = std::move(state_or.value());
        return {};
    }

private:
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

    static DeviceMaskState finishedState(const c10::Device& device, int64_t accepted_token_len) {
        DeviceMaskState state;
        state.token_len = accepted_token_len;
        state.device    = device;
        state.mode      = DeviceMaskMode::FINISHED;
        return state;
    }

    ErrorResult<DeviceMaskState>
    buildState(const c10::Device& device, RtpGrammarMatcher& matcher, int64_t accepted_token_len) {
        DeviceMaskState state;
        state.token_len = accepted_token_len;
        state.device    = device;

        if (matcher.finished()) {
            state.mode = DeviceMaskMode::FINISHED;
            return ErrorResult<DeviceMaskState>(std::move(state));
        }
        auto terminated = matcher.isTerminated();
        if (!terminated.ok()) {
            matcher.markFinished();
            return terminated.status();
        }
        if (terminated.value()) {
            state.mode = DeviceMaskMode::TERMINATED;
            return ErrorResult<DeviceMaskState>(std::move(state));
        }

        auto grammar_vocab_size_or = matcher.vocabSize();
        if (!grammar_vocab_size_or.ok()) {
            matcher.markFinished();
            return grammar_vocab_size_or.status();
        }
        const int32_t grammar_vocab_size = grammar_vocab_size_or.value();
        if (grammar_vocab_size <= 0) {
            state.mode = DeviceMaskMode::NOOP;
            return ErrorResult<DeviceMaskState>(std::move(state));
        }

        auto bitmask = prepareBitmask(grammar_vocab_size);
        auto filled  = fillMatcherBitmask(matcher, bitmask);
        if (!filled.ok()) {
            matcher.markFinished();
            return filled.status();
        }
        if (!filled.value()) {
            matcher.markFinished();
            const std::string msg = "grammar matcher fillBitmask failed; matcher state corrupted";
            return ErrorInfo(ErrorCode::GRAMMAR_FILL_BITMASK_FAILED, msg);
        }

        state.mode               = DeviceMaskMode::MASK;
        state.grammar_vocab_size = grammar_vocab_size;
        materializeVocabMask(state, bitmask);
        publishMaskToDevice(state, state.vocab_mask, device);
        return ErrorResult<DeviceMaskState>(std::move(state));
    }

    torch::Tensor prepareBitmask(int32_t grammar_vocab_size) {
        const int32_t words = (grammar_vocab_size + 31) / 32;
        if (!reusable_bitmask_cpu_.defined() || reusable_mask_words_ < words) {
            reusable_bitmask_cpu_ = at::full({1, words}, -1, at::dtype(at::kInt)).pin_memory();
            reusable_mask_words_  = words;
        } else {
            reusable_bitmask_cpu_.fill_(-1);
        }
        return reusable_bitmask_cpu_.narrow(1, 0, words);
    }

    static ErrorResult<bool> fillMatcherBitmask(RtpGrammarMatcher& matcher, const torch::Tensor& bitmask) {
        int64_t  dl_shape[2];
        DLTensor dl =
            makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), static_cast<int32_t>(bitmask.size(1)), dl_shape);
        return matcher.fillBitmask(&dl, 0);
    }

    void materializeVocabMask(DeviceMaskState& state, const torch::Tensor& bitmask) {
        const int32_t grammar_vocab_size = state.grammar_vocab_size;
        if (!reusable_vocab_mask_cpu_.defined() || reusable_vocab_mask_cpu_.size(0) < grammar_vocab_size) {
            auto mask_options        = torch::TensorOptions().dtype(torch::kBool).pinned_memory(state.device.is_cuda());
            reusable_vocab_mask_cpu_ = torch::empty({grammar_vocab_size}, mask_options);
        }

        auto           vocab_mask  = reusable_vocab_mask_cpu_.narrow(0, 0, grammar_vocab_size);
        bool*          mask_ptr    = vocab_mask.data_ptr<bool>();
        const int32_t* bitmask_ptr = bitmask.data_ptr<int32_t>();
        const size_t   words       = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);
        for (int32_t token_id = 0; token_id < grammar_vocab_size; ++token_id) {
            mask_ptr[token_id] = !bitmaskAllowsToken(bitmask_ptr, words, token_id);
        }
        state.vocab_mask = vocab_mask;
    }

    static void publishMaskToDevice(DeviceMaskState& state, torch::Tensor vocab_mask, const c10::Device& device) {
        if (!device.is_cuda()) {
            state.vocab_mask = std::move(vocab_mask);
            return;
        }

        state.vocab_mask = vocab_mask.to(device, /*non_blocking=*/true);
#if USING_CUDA
        if (device.is_cuda()) {
            auto event = std::make_shared<c10::Event>(c10::DeviceType::CUDA);
            event->record(at::cuda::getCurrentCUDAStream(device.index()).unwrap());
            state.mask_ready = std::move(event);
        }
#endif
    }

    static void applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state, int64_t eos_token_id) {
        switch (state.mode) {
            case DeviceMaskMode::UNSET:
            case DeviceMaskMode::NOOP:
            case DeviceMaskMode::FINISHED:
                return;
            case DeviceMaskMode::TERMINATED:
                forceToken(logits, eos_token_id);
                return;
            case DeviceMaskMode::MASK:
                break;
        }

        if (!state.vocab_mask.defined()) {
            return;
        }
        auto mask = state.vocab_mask;
        if (mask.device() != logits.device()) {
            mask = mask.to(logits.device(), /*non_blocking=*/true);
        } else if (state.mask_ready) {
#if USING_CUDA
            if (logits.device().is_cuda()) {
                state.mask_ready->block(at::cuda::getCurrentCUDAStream(logits.device().index()).unwrap());
            }
#endif
        }
        const int64_t mask_vocab_size = std::min<int64_t>(logits.size(0), mask.size(0));
        if (mask_vocab_size > 0) {
            logits.narrow(0, 0, mask_vocab_size)
                .masked_fill_(mask.narrow(0, 0, mask_vocab_size), BaseLogitsProcessor::neg_inf);
        }
        if (mask.size(0) < logits.size(0)) {
            logits.narrow(0, mask.size(0), logits.size(0) - mask.size(0)).fill_(BaseLogitsProcessor::neg_inf);
        }
    }

    static void forceToken(const torch::Tensor& logits, int64_t token_id) {
        if (token_id < 0 || token_id >= logits.size(0)) {
            return;
        }
        logits.fill_(BaseLogitsProcessor::neg_inf);
        logits[token_id] = 0.0f;
    }

    std::optional<c10::Device> last_mask_device_;
    DeviceMaskState            device_mask_state_{};
    torch::Tensor              reusable_bitmask_cpu_;
    torch::Tensor              reusable_vocab_mask_cpu_;
    int32_t                    reusable_mask_words_ = 0;
};

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher, int64_t eos_token_id):
    matcher_(std::move(matcher)),
    eos_token_id_(eos_token_id),
    decode_mask_builder_(std::make_unique<DecodeMaskBuilder>()) {}

GrammarLogitsProcessor::~GrammarLogitsProcessor() = default;

void GrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (hasError()) {
        return;
    }
    if (!matcher_) {
        return;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size == 0) {
        return;
    }
    if (batch_size != 1) {
        setError(ErrorCode::INVALID_PARAMS, "grammar logits processor only supports single sequence decoding");
        return;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = inputs.finished_mask.data_ptr<bool>();
        if (finished[start_idx]) {
            return;
        }
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        setError(decode_mask_builder_->apply(inputs.logits[start_idx], *matcher_, accepted_token_len_, eos_token_id_));
    }
}

void GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (hasError()) {
        return;
    }
    if (!matcher_) {
        return;
    }

    RTP_LLM_CHECK(new_tokens.dim() == 2);
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(new_tokens.size(1) >= num_new_tokens);
    RTP_LLM_CHECK(new_tokens.is_contiguous());

    const int batch_size = static_cast<int>(new_tokens.size(0));
    // Keep parity with process(): this processor owns one matcher state machine,
    // so multi-sequence updates would corrupt parser state.
    if (batch_size != 1) {
        setError(ErrorCode::INVALID_PARAMS, "grammar logits processor only supports single sequence decoding");
        return;
    }
    const auto* data = new_tokens.data_ptr<int32_t>();

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        setError(acceptCommittedLocked(data, static_cast<size_t>(num_new_tokens)));
    }
}

bool GrammarLogitsProcessor::isSpecVerifyEligible() const {
    return matcher_ != nullptr;
}

int GrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (hasError()) {
        return 0;
    }
    if (!matcher_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return static_cast<int>(request.propose_step);
    }

    int cap_out = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto                        cap_or = verifySpecDraftAndFillBitmask(*matcher_, eos_token_id_, request);
        if (!cap_or.ok()) {
            setError(cap_or.status());
            return 0;
        }
        cap_out = cap_or.value();
    }
    return cap_out;
}

ErrorInfo GrammarLogitsProcessor::acceptCommittedLocked(const int32_t* tokens, size_t n) {
    if (!matcher_ || matcher_->finished() || n == 0) {
        return ErrorInfo::OkStatus();
    }

    RTP_LLM_PROFILE_SCOPE("grammar.acceptToken");

    auto finish_with_error = [this](const ErrorInfo& error) {
        matcher_->markFinished();
        return error;
    };

    for (size_t i = 0; i < n; ++i) {
        const int32_t tok        = tokens[i];
        auto          terminated = matcher_->isTerminated();
        if (!terminated.ok()) {
            return finish_with_error(terminated.status());
        }
        if (terminated.value()) {
            matcher_->markFinished();
            if (tok == static_cast<int32_t>(eos_token_id_)) {
                accepted_token_len_ = matcher_->numAcceptedTokens() + 1;
            } else {
                return ErrorInfo(ErrorCode::GRAMMAR_NON_EOS_AFTER_TERMINAL,
                                 "grammar received non-EOS token after terminal state");
            }
            break;
        }
        auto accepted = matcher_->acceptToken(tok);
        if (!accepted.ok()) {
            return finish_with_error(accepted.status());
        }
        if (!accepted.value()) {
            return finish_with_error(ErrorInfo(ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN,
                                               "grammar commit error: parser rejected token " + std::to_string(tok)));
        }
        accepted_token_len_ = matcher_->numAcceptedTokens();
    }

    return decode_mask_builder_->refreshAfterCommit(*matcher_, accepted_token_len_);
}

}  // namespace rtp_llm
