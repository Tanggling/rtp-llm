#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstddef>
#include <exception>
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
    Failed,
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
    const int32_t grammar_vocab_size = matcher.vocabSize();
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

SpecVerifyRowState
fillSpecVerifyRow(RtpGrammarMatcher& matcher, int64_t eos_token_id, int32_t* row, size_t W, size_t model_vocab_size) {
    std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
    if (matcher.finished()) {
        forceTokenInBitmask(row, W, eos_token_id);
        return SpecVerifyRowState::Finished;
    }
    if (matcher.isTerminated()) {
        forceTokenInBitmask(row, W, eos_token_id);
        return SpecVerifyRowState::Terminated;
    }

    const int32_t grammar_vocab_size = matcher.vocabSize();
    const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);

    int64_t  dl_shape[2];
    DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
    if (!matcher.fillBitmask(&dl, 0)) {
        matcher.markFinished();
        forceTokenInBitmask(row, W, eos_token_id);
        return SpecVerifyRowState::Failed;
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

    void captureException(const std::exception& e) {
        if (verify_exception_what_.empty()) {
            verify_exception_what_ = e.what();
        }
    }

    void captureUnknownException() {
        if (verify_exception_what_.empty()) {
            verify_exception_what_ = "unknown";
        }
    }

    bool rollbackAndReport(int32_t* fallback_row, size_t W, int64_t eos_token_id, ErrorInfo& out_err) {
        rollback();
        if (verify_exception_what_.empty()) {
            return true;
        }
        matcher_.markFinished();
        forceTokenInBitmask(fallback_row, W, eos_token_id);
        out_err =
            ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION, "grammar MTP verify exception: " + verify_exception_what_);
        return false;
    }

private:
    void rollback() {
        try {
            if (accepted_prefix_ > 0) {
                matcher_.rollback(accepted_prefix_);
            }
        } catch (const std::exception& e) {
            matcher_.markFinished();
            if (verify_exception_what_.empty()) {
                verify_exception_what_ = std::string("rollback: ") + e.what();
            }
        } catch (...) {
            matcher_.markFinished();
            if (verify_exception_what_.empty()) {
                verify_exception_what_ = "rollback: unknown";
            }
        }
    }

    RtpGrammarMatcher& matcher_;
    int                accepted_prefix_ = 0;
    std::string        verify_exception_what_;
};

[[nodiscard]] int verifyDraftPrefixAndFillBitmask(RtpGrammarMatcher&                matcher,
                                                  int64_t                           eos_token_id,
                                                  const SpecLogitsProcessorRequest& request,
                                                  ProvisionalSpecAcceptGuard&       provisional) {
    const int  P = request.propose_step;
    const auto W = request.bitmask_size_int32;

    for (int offset = 0; offset <= P; ++offset) {
        int32_t*   row       = request.bitmask_cpu_out + offset * W;
        const auto row_state = fillSpecVerifyRow(matcher, eos_token_id, row, W, request.vocab_size);
        if (offset == P) {
            return P;
        }
        if (!specVerifyRowCanConsumeDraft(row_state)) {
            return offset;
        }

        const int32_t draft_token = request.draft_tokens[offset];
        if (!specVerifyDraftTokenAllowed(row, W, request.vocab_size, draft_token)) {
            return offset;
        }
        if (!matcher.acceptToken(draft_token)) {
            return offset;
        }
        provisional.recordAccepted();
    }

    return P;
}

int verifySpecDraftAndFillBitmask(RtpGrammarMatcher&                matcher,
                                  int64_t                           eos_token_id,
                                  const SpecLogitsProcessorRequest& request,
                                  ErrorInfo&                        out_err) {
    if (auto err = preflightSpecVerifyRequest(request); err.hasError()) {
        out_err = err;
        return 0;
    }

    const auto W = request.bitmask_size_int32;

    if (auto err = validateSpecVerifyMatcher(matcher, eos_token_id, W); err.hasError()) {
        out_err = err;
        return 0;
    }

    ProvisionalSpecAcceptGuard provisional(matcher);
    int                        cap = 0;

    try {
        cap = verifyDraftPrefixAndFillBitmask(matcher, eos_token_id, request, provisional);
    } catch (const std::exception& e) {
        provisional.captureException(e);
    } catch (...) {
        provisional.captureUnknownException();
    }

    if (!provisional.rollbackAndReport(request.bitmask_cpu_out, W, eos_token_id, out_err)) {
        return 0;
    }
    return cap;
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

        ErrorInfo build_err;
        device_mask_state_ = buildState(logits.device(), matcher, accepted_token_len, build_err);
        applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
        return build_err;
    }

    ErrorInfo refreshAfterCommit(RtpGrammarMatcher& matcher, int64_t accepted_token_len) {
        ErrorInfo build_err;
        device_mask_state_ = buildState(
            last_mask_device_.value_or(c10::Device(c10::DeviceType::CPU)), matcher, accepted_token_len, build_err);
        return build_err;
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

    DeviceMaskState
    buildState(const c10::Device& device, RtpGrammarMatcher& matcher, int64_t accepted_token_len, ErrorInfo& out_err) {
        DeviceMaskState state;
        state.token_len = accepted_token_len;
        state.device    = device;

        if (matcher.finished()) {
            state.mode = DeviceMaskMode::FINISHED;
            return state;
        }
        if (matcher.isTerminated()) {
            state.mode = DeviceMaskMode::TERMINATED;
            return state;
        }

        const int32_t grammar_vocab_size = matcher.vocabSize();
        if (grammar_vocab_size <= 0) {
            state.mode = DeviceMaskMode::NOOP;
            return state;
        }

        auto bitmask = prepareBitmask(grammar_vocab_size);
        if (!fillMatcherBitmask(matcher, bitmask)) {
            matcher.markFinished();
            state.mode = DeviceMaskMode::FINISHED;
            out_err    = ErrorInfo(ErrorCode::GRAMMAR_FILL_BITMASK_FAILED,
                                "grammar matcher fillBitmask failed; matcher state corrupted");
            return state;
        }

        state.mode               = DeviceMaskMode::MASK;
        state.grammar_vocab_size = grammar_vocab_size;
        materializeVocabMask(state, bitmask);
        publishMaskToDevice(state, state.vocab_mask, device);
        return state;
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

    static bool fillMatcherBitmask(RtpGrammarMatcher& matcher, const torch::Tensor& bitmask) {
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

    ErrorInfo local_err;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        local_err =
            decode_mask_builder_->apply(inputs.logits[start_idx], *matcher_, accepted_token_len_, eos_token_id_);
    }
    if (local_err.hasError()) {
        setError(local_err);
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

    ErrorInfo local_err;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        acceptCommittedLocked(data, static_cast<size_t>(num_new_tokens), local_err);
    }
    if (local_err.hasError()) {
        setError(local_err);
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

    ErrorInfo local_err;
    int       cap_out = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        cap_out = verifySpecDraftAndFillBitmask(*matcher_, eos_token_id_, request, local_err);
    }
    if (local_err.hasError()) {
        setError(local_err);
        return 0;
    }
    return cap_out;
}

void GrammarLogitsProcessor::acceptCommittedLocked(const int32_t* tokens, size_t n, ErrorInfo& out_err) {
    if (!matcher_ || matcher_->finished() || n == 0) {
        return;
    }

    RTP_LLM_PROFILE_SCOPE("grammar.acceptToken");

    for (size_t i = 0; i < n; ++i) {
        const int32_t tok = tokens[i];
        if (matcher_->isTerminated()) {
            matcher_->markFinished();
            if (tok == static_cast<int32_t>(eos_token_id_)) {
                accepted_token_len_ = matcher_->numAcceptedTokens() + 1;
            } else {
                out_err = ErrorInfo(ErrorCode::GRAMMAR_NON_EOS_AFTER_TERMINAL,
                                    "grammar received non-EOS token after terminal state");
            }
            break;
        }
        if (!matcher_->acceptToken(tok)) {
            matcher_->markFinished();
            out_err = ErrorInfo(ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN,
                                "grammar commit error: parser rejected token " + std::to_string(tok));
            break;
        }
        accepted_token_len_ = matcher_->numAcceptedTokens();
    }

    if (!out_err.hasError()) {
        out_err = decode_mask_builder_->refreshAfterCommit(*matcher_, accepted_token_len_);
    }
}

}  // namespace rtp_llm
