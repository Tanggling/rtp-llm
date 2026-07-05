#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include <dlpack/dlpack.h>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<GrammarMatcher> matcher, int64_t eos_token_id):
    matcher_(std::move(matcher)), eos_token_id_(eos_token_id) {}

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
        applyMaskLocked(inputs.logits[start_idx], local_err);
    }
    if (local_err.hasError()) {
        setError(local_err);
    }
}

void GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (hasError()) {
        return;
    }
    if (!matcher_ || finished()) {
        return;
    }

    RTP_LLM_CHECK(new_tokens.dim() == 2);
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(new_tokens.size(1) >= num_new_tokens);
    RTP_LLM_CHECK(new_tokens.is_contiguous());

    const int batch_size = static_cast<int>(new_tokens.size(0));
    // Mirror process()'s single-sequence guard: with batch_size > 1 the loop below
    // concatenates tokens from different sequences into one vector fed to a single
    // matcher state machine, corrupting the parse state. LogitsProcessorFactory
    // already rejects beam/num_return_sequences>1, but keep the defensive parity
    // with process(); feeding multiple sequences into one matcher corrupts parser state.
    if (batch_size != 1) {
        setError(ErrorCode::INVALID_PARAMS, "grammar logits processor only supports single sequence decoding");
        return;
    }
    const int            stride = static_cast<int>(new_tokens.size(1));
    const auto*          data   = new_tokens.data_ptr<int32_t>();
    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(batch_size * num_new_tokens));
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_new_tokens; ++j) {
            tokens.push_back(data[i * stride + j]);
        }
    }

    ErrorInfo local_err;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        acceptCommittedLocked(tokens.data(), tokens.size(), local_err);
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
    if (auto err = preflightSpecRequest(request); err.hasError()) {
        setError(err);
        return 0;
    }

    ErrorInfo local_err;
    int       cap_out = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        cap_out = runSpecVerifyLocked(request, local_err);
    }
    if (local_err.hasError()) {
        setError(local_err);
        return 0;
    }
    return cap_out;
}

bool GrammarLogitsProcessor::finished() const {
    return !matcher_ || matcher_->finished();
}

bool GrammarLogitsProcessor::isTerminated() const {
    return matcher_ && matcher_->isTerminated();
}

int64_t GrammarLogitsProcessor::numAcceptedTokens() const {
    return matcher_ ? matcher_->numAcceptedTokens() : 0;
}

int32_t GrammarLogitsProcessor::vocabSize() const {
    return matcher_ ? matcher_->vocabSize() : 0;
}

void GrammarLogitsProcessor::markFinished() {
    if (matcher_) {
        matcher_->markFinished();
    }
}

void GrammarLogitsProcessor::rollback(int n) {
    if (matcher_) {
        matcher_->rollback(n);
    }
}

bool GrammarLogitsProcessor::acceptToken(int32_t token_id) {
    return matcher_ && matcher_->acceptToken(token_id);
}

GrammarLogitsProcessor::DeviceMaskState
GrammarLogitsProcessor::buildDeviceMaskStateLocked(const c10::Device& device, ErrorInfo& out_err) {
    DeviceMaskState state;
    state.token_len = accepted_token_len_;
    state.device    = device;

    if (!matcher_) {
        state.mode = DeviceMaskMode::NOOP;
        return state;
    }
    if (matcher_->finished()) {
        state.mode = DeviceMaskMode::FINISHED;
        return state;
    }
    if (matcher_->isTerminated()) {
        state.mode = DeviceMaskMode::TERMINATED;
        return state;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size <= 0) {
        state.mode = DeviceMaskMode::NOOP;
        return state;
    }

    const int32_t words = (grammar_vocab_size + 31) / 32;
    if (!reusable_bitmask_cpu_.defined() || reusable_mask_words_ < words) {
        reusable_bitmask_cpu_ = at::full({1, words}, -1, at::dtype(at::kInt)).pin_memory();
        reusable_mask_words_  = words;
    } else {
        reusable_bitmask_cpu_.fill_(-1);
    }
    auto    bitmask = reusable_bitmask_cpu_.narrow(1, 0, words);
    int64_t dl_shape[2];
    DLTensor dl = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words, dl_shape);
    if (!matcher_->fillBitmask(&dl, 0)) {
        matcher_->markFinished();
        state.mode = DeviceMaskMode::FINISHED;
        out_err    = ErrorInfo(ErrorCode::GRAMMAR_FILL_BITMASK_FAILED,
                            "grammar matcher fillBitmask failed; matcher state corrupted");
        return state;
    }

    state.mode               = DeviceMaskMode::MASK;
    state.grammar_vocab_size = grammar_vocab_size;

    if (!reusable_vocab_mask_cpu_.defined() || reusable_vocab_mask_cpu_.size(0) < grammar_vocab_size) {
        auto mask_options        = torch::TensorOptions().dtype(torch::kBool).pinned_memory(device.is_cuda());
        reusable_vocab_mask_cpu_ = torch::empty({grammar_vocab_size}, mask_options);
    }
    auto           vocab_mask  = reusable_vocab_mask_cpu_.narrow(0, 0, grammar_vocab_size);
    bool*          mask_ptr    = vocab_mask.data_ptr<bool>();
    const int32_t* bitmask_ptr = bitmask.data_ptr<int32_t>();
    const size_t   words_sz    = static_cast<size_t>(words);
    for (int32_t token_id = 0; token_id < grammar_vocab_size; ++token_id) {
        mask_ptr[token_id] = !bitmaskAllowsToken(bitmask_ptr, words_sz, token_id);
    }

    publishMaskToDevice(state, vocab_mask, device);
    return state;
}

void GrammarLogitsProcessor::publishMaskToDevice(DeviceMaskState&   state,
                                                 torch::Tensor      vocab_mask,
                                                 const c10::Device& device) {
    if (!device.is_cuda()) {
        state.vocab_mask = vocab_mask;
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

void GrammarLogitsProcessor::applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state) {
    switch (state.mode) {
        case DeviceMaskMode::UNSET:
        case DeviceMaskMode::NOOP:
        case DeviceMaskMode::FINISHED:
            return;
        case DeviceMaskMode::TERMINATED:
            forceToken(logits, eos_token_id_);
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

void GrammarLogitsProcessor::forceToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        return;
    }
    logits.fill_(BaseLogitsProcessor::neg_inf);
    logits[token_id] = 0.0f;
}

void GrammarLogitsProcessor::applyMaskLocked(const torch::Tensor& logits, ErrorInfo& out_err) {
    if (!matcher_) {
        return;
    }
    last_mask_device_ = logits.device();

    if (device_mask_state_.mode != DeviceMaskMode::UNSET && device_mask_state_.token_len == accepted_token_len_
        && device_mask_state_.device == logits.device()) {
        applyDeviceMaskState(logits, device_mask_state_);
        return;
    }

    ErrorInfo build_err;
    device_mask_state_ = buildDeviceMaskStateLocked(logits.device(), build_err);
    if (build_err.hasError()) {
        out_err = build_err;
    }
    applyDeviceMaskState(logits, device_mask_state_);
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
        ErrorInfo build_err;
        device_mask_state_ =
            buildDeviceMaskStateLocked(last_mask_device_.value_or(c10::Device(c10::DeviceType::CPU)), build_err);
        if (build_err.hasError()) {
            out_err = build_err;
        }
    }
}

ErrorInfo GrammarLogitsProcessor::preflightSpecRequest(const SpecLogitsProcessorRequest& request) const {
    if (request.bitmask_size_int32 < static_cast<size_t>((request.vocab_size + 31) / 32)) {
        return ErrorInfo(ErrorCode::GRAMMAR_BITMASK_BUFFER_TOO_SMALL,
                         "grammar MTP verify: bitmask buffer smaller than model vocab (words="
                             + std::to_string(request.bitmask_size_int32)
                             + ", vocab=" + std::to_string(request.vocab_size) + ")");
    }
    return {};
}

ErrorInfo GrammarLogitsProcessor::validateMatcherInvariantsLocked(const SpecLogitsProcessorRequest& request) {
    const auto W = request.bitmask_size_int32;

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size > 0 && SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size) > W) {
        matcher_->markFinished();
        return ErrorInfo(ErrorCode::GRAMMAR_VOCAB_EXCEEDS_MODEL_VOCAB,
                         "grammar vocab exceeds model vocab in MTP verify (grammar="
                             + std::to_string(grammar_vocab_size) + ", model_words=" + std::to_string(W) + ")");
    }

    auto token_in_range = [W](int64_t t) { return t >= 0 && static_cast<size_t>(t / 32) < W; };
    if (!token_in_range(eos_token_id_)) {
        matcher_->markFinished();
        return ErrorInfo(ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                         "grammar MTP verify: eos_token_id (" + std::to_string(eos_token_id_)
                             + ") out of model vocab bitmask (words=" + std::to_string(W) + ")");
    }
    return {};
}

GrammarLogitsProcessor::RowState
GrammarLogitsProcessor::fillGrammarRowLocked(int32_t* row, size_t W, size_t model_vocab_size) {
    std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
    if (matcher_->finished()) {
        forceTokenInBitmask(row, W, eos_token_id_);
        return RowState::Finished;
    }
    if (matcher_->isTerminated()) {
        forceTokenInBitmask(row, W, eos_token_id_);
        return RowState::Terminated;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);

    int64_t  dl_shape[2];
    DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
    if (!matcher_->fillBitmask(&dl, 0)) {
        matcher_->markFinished();
        forceTokenInBitmask(row, W, eos_token_id_);
        return RowState::Failed;
    }
    clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(model_vocab_size));
    return RowState::Active;
}

int GrammarLogitsProcessor::runSpecVerifyGuarded(
    int32_t*                                                                     bitmask_cpu_out,
    size_t                                                                       W,
    const char*                                                                  who,
    const std::function<int(int& grammar_accepted_prefix, ErrorInfo& walk_err)>& walk,
    ErrorInfo&                                                                   out_err) {
    int         grammar_accepted_prefix = 0;
    int         cap                     = 0;
    ErrorInfo   walk_err;
    std::string verify_exception_what;

    auto rollback_provisional = [&]() {
        try {
            if (grammar_accepted_prefix > 0) {
                matcher_->rollback(grammar_accepted_prefix);
            }
        } catch (const std::exception& e) {
            matcher_->markFinished();
            if (verify_exception_what.empty()) {
                verify_exception_what = std::string("rollback: ") + e.what();
            }
        } catch (...) {
            matcher_->markFinished();
            if (verify_exception_what.empty()) {
                verify_exception_what = "rollback: unknown";
            }
        }
    };

    try {
        cap = walk(grammar_accepted_prefix, walk_err);
    } catch (const std::exception& e) {
        verify_exception_what = e.what();
    } catch (...) {
        verify_exception_what = "unknown";
    }

    rollback_provisional();

    if (!verify_exception_what.empty()) {
        matcher_->markFinished();
        forceTokenInBitmask(bitmask_cpu_out, W, eos_token_id_);
        out_err = ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION,
                            std::string(who) + " MTP verify exception: " + verify_exception_what);
        return 0;
    }
    if (walk_err.hasError()) {
        out_err = walk_err;
        return 0;
    }
    return cap;
}

int GrammarLogitsProcessor::runSpecVerifyLocked(const SpecLogitsProcessorRequest& request, ErrorInfo& out_err) {
    if (auto err = validateMatcherInvariantsLocked(request); err.hasError()) {
        out_err = err;
        return 0;
    }

    const int  P = request.propose_step;
    const auto W = request.bitmask_size_int32;

    return runSpecVerifyGuarded(
        request.bitmask_cpu_out,
        W,
        "grammar",
        [&](int& accepted_prefix, ErrorInfo& /*walk_err*/) -> int {
            int cap = P;
            for (int offset = 0; offset <= P; ++offset) {
                int32_t* row       = request.bitmask_cpu_out + offset * W;
                RowState row_state = fillGrammarRowLocked(row, W, request.vocab_size);
                if (offset == P) {
                    break;
                }
                if (row_state == RowState::Terminated || row_state == RowState::Finished
                    || row_state == RowState::Failed) {
                    cap = offset;
                    break;
                }

                const int32_t draft_token = request.draft_tokens[offset];
                if (draft_token < 0 || static_cast<size_t>(draft_token) >= request.vocab_size
                    || !bitmaskAllowsToken(row, W, draft_token)) {
                    cap = offset;
                    break;
                }
                if (!matcher_->acceptToken(draft_token)) {
                    cap = offset;
                    break;
                }
                ++accepted_prefix;
            }
            return cap;
        },
        out_err);
}

}  // namespace rtp_llm
