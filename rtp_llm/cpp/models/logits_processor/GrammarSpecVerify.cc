#include "rtp_llm/cpp/models/logits_processor/GrammarSpecVerify.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <string>

#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"

namespace rtp_llm {
namespace grammar_spec_verify {
namespace {

enum class RowState {
    Active,
    Finished,
    Terminated,
    Failed,
};

ErrorInfo preflightRequest(const SpecLogitsProcessorRequest& request) {
    if (request.bitmask_size_int32 < SpecLogitsProcessor::bitmaskWordCount(request.vocab_size)) {
        return ErrorInfo(ErrorCode::GRAMMAR_BITMASK_BUFFER_TOO_SMALL,
                         "grammar MTP verify: bitmask buffer smaller than model vocab (words="
                             + std::to_string(request.bitmask_size_int32)
                             + ", vocab=" + std::to_string(request.vocab_size) + ")");
    }
    return {};
}

ErrorInfo validateMatcherInvariants(GrammarMatcher& matcher, int64_t eos_token_id, size_t W) {
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

RowState
fillGrammarRow(GrammarMatcher& matcher, int64_t eos_token_id, int32_t* row, size_t W, size_t model_vocab_size) {
    std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
    if (matcher.finished()) {
        forceTokenInBitmask(row, W, eos_token_id);
        return RowState::Finished;
    }
    if (matcher.isTerminated()) {
        forceTokenInBitmask(row, W, eos_token_id);
        return RowState::Terminated;
    }

    const int32_t grammar_vocab_size = matcher.vocabSize();
    const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);

    int64_t  dl_shape[2];
    DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
    if (!matcher.fillBitmask(&dl, 0)) {
        matcher.markFinished();
        forceTokenInBitmask(row, W, eos_token_id);
        return RowState::Failed;
    }
    clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(model_vocab_size));
    return RowState::Active;
}

void rollbackProvisional(GrammarMatcher& matcher, int accepted_prefix, std::string& verify_exception_what) {
    try {
        if (accepted_prefix > 0) {
            matcher.rollback(accepted_prefix);
        }
    } catch (const std::exception& e) {
        matcher.markFinished();
        if (verify_exception_what.empty()) {
            verify_exception_what = std::string("rollback: ") + e.what();
        }
    } catch (...) {
        matcher.markFinished();
        if (verify_exception_what.empty()) {
            verify_exception_what = "rollback: unknown";
        }
    }
}

}  // namespace

int verify(GrammarMatcher& matcher,
           int64_t eos_token_id,
           const SpecLogitsProcessorRequest& request,
           ErrorInfo& out_err) {
    if (auto err = preflightRequest(request); err.hasError()) {
        out_err = err;
        return 0;
    }

    const int  P = request.propose_step;
    const auto W = request.bitmask_size_int32;

    if (auto err = validateMatcherInvariants(matcher, eos_token_id, W); err.hasError()) {
        out_err = err;
        return 0;
    }

    int         accepted_prefix = 0;
    int         cap             = P;
    std::string verify_exception_what;

    try {
        for (int offset = 0; offset <= P; ++offset) {
            int32_t* row       = request.bitmask_cpu_out + offset * W;
            RowState row_state = fillGrammarRow(matcher, eos_token_id, row, W, request.vocab_size);
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
            if (!matcher.acceptToken(draft_token)) {
                cap = offset;
                break;
            }
            ++accepted_prefix;
        }
    } catch (const std::exception& e) {
        verify_exception_what = e.what();
    } catch (...) {
        verify_exception_what = "unknown";
    }

    rollbackProvisional(matcher, accepted_prefix, verify_exception_what);

    if (!verify_exception_what.empty()) {
        matcher.markFinished();
        forceTokenInBitmask(request.bitmask_cpu_out, W, eos_token_id);
        out_err = ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION,
                            "grammar MTP verify exception: " + verify_exception_what);
        return 0;
    }
    return cap;
}

}  // namespace grammar_spec_verify
}  // namespace rtp_llm
