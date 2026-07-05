#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <optional>
#include <utility>

#include <dlpack/dlpack.h>
#include <c10/core/Event.h>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarSpecVerify.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

class GrammarLogitsProcessor::DecodeMaskBuilder final {
public:
    void apply(const torch::Tensor& logits,
               GrammarMatcher&      matcher,
               int64_t              accepted_token_len,
               int64_t              eos_token_id,
               ErrorInfo&           out_err) {
        last_mask_device_ = logits.device();

        if (device_mask_state_.mode != DeviceMaskMode::UNSET && device_mask_state_.token_len == accepted_token_len
            && device_mask_state_.device == logits.device()) {
            applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
            return;
        }

        ErrorInfo build_err;
        device_mask_state_ = buildState(logits.device(), matcher, accepted_token_len, build_err);
        if (build_err.hasError()) {
            out_err = build_err;
        }
        applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
    }

    void refreshAfterCommit(GrammarMatcher& matcher, int64_t accepted_token_len, ErrorInfo& out_err) {
        ErrorInfo build_err;
        device_mask_state_ = buildState(
            last_mask_device_.value_or(c10::Device(c10::DeviceType::CPU)), matcher, accepted_token_len, build_err);
        if (build_err.hasError()) {
            out_err = build_err;
        }
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
    buildState(const c10::Device& device, GrammarMatcher& matcher, int64_t accepted_token_len, ErrorInfo& out_err) {
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

        const int32_t words = (grammar_vocab_size + 31) / 32;
        if (!reusable_bitmask_cpu_.defined() || reusable_mask_words_ < words) {
            reusable_bitmask_cpu_ = at::full({1, words}, -1, at::dtype(at::kInt)).pin_memory();
            reusable_mask_words_  = words;
        } else {
            reusable_bitmask_cpu_.fill_(-1);
        }
        auto     bitmask = reusable_bitmask_cpu_.narrow(1, 0, words);
        int64_t  dl_shape[2];
        DLTensor dl = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words, dl_shape);
        if (!matcher.fillBitmask(&dl, 0)) {
            matcher.markFinished();
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

    static void publishMaskToDevice(DeviceMaskState& state, torch::Tensor vocab_mask, const c10::Device& device) {
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

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<GrammarMatcher> matcher, int64_t eos_token_id):
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
        decode_mask_builder_->apply(inputs.logits[start_idx], *matcher_, accepted_token_len_, eos_token_id_, local_err);
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

    ErrorInfo local_err;
    int       cap_out = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        cap_out = grammar_spec_verify::verify(*matcher_, eos_token_id_, request, local_err);
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
        ErrorInfo build_err;
        decode_mask_builder_->refreshAfterCommit(*matcher_, accepted_token_len_, build_err);
        if (build_err.hasError()) {
            out_err = build_err;
        }
    }
}

}  // namespace rtp_llm
