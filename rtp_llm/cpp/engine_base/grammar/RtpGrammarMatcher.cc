#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

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

bool RtpGrammarMatcher::acceptToken(int32_t token_id) {
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
}

bool RtpGrammarMatcher::acceptTokens(const std::vector<int32_t>& tokens) {
    for (int32_t token_id : tokens) {
        if (!acceptToken(token_id)) {
            return false;
        }
    }
    return true;
}

bool RtpGrammarMatcher::fillBitmask(DLTensor* bitmask, int32_t idx) {
    return matcher_->FillNextTokenBitmask(bitmask, idx);
}

bool RtpGrammarMatcher::isTerminated() const {
    return matcher_->IsTerminated();
}

void RtpGrammarMatcher::rollback(int n) {
    if (n <= 0) {
        return;
    }
    matcher_->Rollback(n);
    num_accepted_ = std::max<int64_t>(0, num_accepted_ - n);
}

}  // namespace rtp_llm
