#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

ErrorInfo matcherExceptionError(const char* op, const std::exception& e) {
    return ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION,
                     std::string("grammar matcher ") + op + " exception: " + e.what());
}

ErrorInfo matcherUnknownError(const char* op) {
    return ErrorInfo(ErrorCode::GRAMMAR_VERIFY_EXCEPTION, std::string("grammar matcher ") + op + " exception: unknown");
}

template<typename Fn>
auto matcherResult(const char* op, Fn&& fn) -> ErrorResult<std::decay_t<decltype(fn())>> {
    try {
        return std::forward<Fn>(fn)();
    } catch (const std::exception& e) {
        auto error = matcherExceptionError(op, e);
        return error;
    } catch (...) {
        auto error = matcherUnknownError(op);
        return error;
    }
}

template<typename Fn>
ErrorInfo matcherStatus(const char* op, Fn&& fn) {
    try {
        std::forward<Fn>(fn)();
        return ErrorInfo::OkStatus();
    } catch (const std::exception& e) {
        auto error = matcherExceptionError(op, e);
        return error;
    } catch (...) {
        auto error = matcherUnknownError(op);
        return error;
    }
}

}  // namespace

RtpGrammarMatcher::RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                     std::optional<std::vector<int32_t>>        override_stop_tokens,
                                     bool                                       terminate_without_stop_token,
                                     int                                        max_rollback_tokens):
    compiled_(std::move(compiled)) {
    if (!compiled_) {
        throw std::invalid_argument("RtpGrammarMatcher requires a non-null CompiledGrammar");
    }

    matcher_ = std::make_unique<xgrammar::GrammarMatcher>(
        *compiled_, std::move(override_stop_tokens), terminate_without_stop_token, max_rollback_tokens);
}

ErrorResult<bool> RtpGrammarMatcher::acceptToken(int32_t token_id) {
    return matcherResult("acceptToken", [&] {
        const bool ok = matcher_->AcceptToken(token_id);
        if (!ok) {
            // Spec-verify DFS reacts on the bool; keep at DEBUG to avoid log floods.
            RTP_LLM_LOG_DEBUG("RtpGrammarMatcher::acceptToken REJECTED token=%d, num_accepted=%ld, terminated=%d",
                              token_id,
                              num_accepted_,
                              static_cast<int>(matcher_->IsTerminated()));
            return false;
        }
        ++num_accepted_;
        return true;
    });
}

ErrorResult<bool> RtpGrammarMatcher::acceptTokens(const std::vector<int32_t>& tokens) {
    for (int32_t token_id : tokens) {
        auto accepted = acceptToken(token_id);
        if (!accepted.ok()) {
            return accepted.status();
        }
        if (!accepted.value()) {
            return false;
        }
    }
    return true;
}

ErrorResult<bool> RtpGrammarMatcher::fillBitmask(DLTensor* bitmask, int32_t idx) {
    return matcherResult("fillBitmask", [&] { return matcher_->FillNextTokenBitmask(bitmask, idx); });
}

ErrorResult<bool> RtpGrammarMatcher::isTerminated() const {
    return matcherResult("isTerminated", [&] { return matcher_->IsTerminated(); });
}

ErrorInfo RtpGrammarMatcher::rollback(int n) {
    if (n <= 0) {
        return ErrorInfo::OkStatus();
    }
    return matcherStatus("rollback", [&] {
        matcher_->Rollback(n);
        num_accepted_ = std::max<int64_t>(0, num_accepted_ - n);
    });
}

ErrorResult<int32_t> RtpGrammarMatcher::vocabSize() const {
    return matcherResult("vocabSize", [&] { return compiled_->GetTokenizerInfo().GetVocabSize(); });
}

}  // namespace rtp_llm
