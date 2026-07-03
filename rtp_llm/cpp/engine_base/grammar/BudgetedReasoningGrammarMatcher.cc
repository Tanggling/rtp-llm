#include "rtp_llm/cpp/engine_base/grammar/BudgetedReasoningGrammarMatcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

const char* phaseName(BudgetedReasoningGrammarMatcher::Phase phase) {
    switch (phase) {
        case BudgetedReasoningGrammarMatcher::Phase::BeforeThink:
            return "BeforeThink";
        case BudgetedReasoningGrammarMatcher::Phase::Thinking:
            return "Thinking";
        case BudgetedReasoningGrammarMatcher::Phase::ClosingThink:
            return "ClosingThink";
        case BudgetedReasoningGrammarMatcher::Phase::AfterThink:
            return "AfterThink";
    }
    return "Unknown";
}

int32_t* bitmaskRow(DLTensor* bitmask, int32_t idx) {
    if (!bitmask || !bitmask->data || !bitmask->shape || bitmask->ndim != 2 || idx < 0) {
        return nullptr;
    }
    const auto words = bitmask->shape[1];
    if (words <= 0 || idx >= bitmask->shape[0]) {
        return nullptr;
    }
    auto* data = static_cast<int32_t*>(bitmask->data);
    return data + static_cast<int64_t>(idx) * words;
}

bool rowAllowsToken(const int32_t* row, size_t words, int32_t token_id) {
    if (!row || token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return false;
    }
    const uint32_t word = static_cast<uint32_t>(row[token_id / 32]);
    return (word & (1u << (token_id % 32))) != 0u;
}

}  // namespace

BudgetedReasoningGrammarMatcher::BudgetedReasoningGrammarMatcher(std::shared_ptr<GrammarMatcher> inner,
                                                                 std::optional<int32_t> think_begin_id,
                                                                 std::vector<int32_t>   end_think_token_ids,
                                                                 int64_t                max_thinking_tokens,
                                                                 int                    max_rollback_tokens):
    inner_(std::move(inner)),
    phase_(think_begin_id ? Phase::BeforeThink : Phase::Thinking),
    think_begin_id_(think_begin_id),
    end_think_token_ids_(std::move(end_think_token_ids)),
    max_thinking_tokens_(max_thinking_tokens),
    history_ring_(std::max(1, max_rollback_tokens)) {
    if (!inner_) {
        throw std::invalid_argument("BudgetedReasoningGrammarMatcher requires a non-null inner matcher");
    }
    if (end_think_token_ids_.empty()) {
        throw std::invalid_argument("BudgetedReasoningGrammarMatcher requires non-empty end_think_token_ids");
    }
    RTP_LLM_LOG_DEBUG("BudgetedReasoningGrammarMatcher init: phase=%s, begin_think_id=%d, "
                      "end_think_token_count=%zu, max_thinking_tokens=%ld, max_rollback_tokens=%d",
                      phaseName(phase_),
                      think_begin_id_ ? *think_begin_id_ : -1,
                      end_think_token_ids_.size(),
                      max_thinking_tokens_,
                      max_rollback_tokens);
}

void BudgetedReasoningGrammarMatcher::pushHistory(const Snapshot& snapshot, bool accepted_by_inner) {
    if (history_ring_.empty()) {
        return;
    }
    history_ring_[head_] = HistoryEntry{snapshot, accepted_by_inner};
    head_                = (head_ + 1) % history_ring_.size();
    size_                = std::min<size_t>(size_ + 1, history_ring_.size());
}

BudgetedReasoningGrammarMatcher::HistoryEntry BudgetedReasoningGrammarMatcher::popHistory() {
    if (size_ == 0 || history_ring_.empty()) {
        throw std::runtime_error("BudgetedReasoningGrammarMatcher history underflow");
    }
    head_ = (head_ + history_ring_.size() - 1) % history_ring_.size();
    --size_;
    return history_ring_[head_];
}

bool BudgetedReasoningGrammarMatcher::intersectBitmaskRowWithToken(DLTensor* bitmask,
                                                                   int32_t   idx,
                                                                   int32_t   token_id) const {
    if (!bitmask || !bitmask->shape) {
        return false;
    }
    const auto words = static_cast<size_t>(bitmask->shape[1]);
    int32_t*   row   = bitmaskRow(bitmask, idx);
    if (!row || !rowAllowsToken(row, words, token_id)) {
        return false;
    }
    std::fill(row, row + words, 0);
    row[token_id / 32] |= static_cast<int32_t>(1u << (token_id % 32));
    return true;
}

void BudgetedReasoningGrammarMatcher::clearBitmaskRowToken(DLTensor* bitmask, int32_t idx, int32_t token_id) const {
    if (!bitmask || !bitmask->shape || token_id < 0) {
        return;
    }
    const auto words = static_cast<size_t>(bitmask->shape[1]);
    int32_t*   row   = bitmaskRow(bitmask, idx);
    if (!row || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    const int32_t word_id = token_id / 32;
    uint32_t      word    = static_cast<uint32_t>(row[word_id]);
    word &= ~(1u << (token_id % 32));
    row[word_id] = static_cast<int32_t>(word);
}

bool BudgetedReasoningGrammarMatcher::fillBitmask(DLTensor* bitmask, int32_t idx) {
    if (phase_ == Phase::BeforeThink) {
        if (!think_begin_id_) {
            return true;
        }
        if (!intersectBitmaskRowWithToken(bitmask, idx, *think_begin_id_)) {
            RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher failed to force begin-think token: token=%d, "
                                "idx=%d, accepted=%ld",
                                *think_begin_id_,
                                idx,
                                total_accepted_);
            return false;
        }
        return true;
    }

    if (phase_ == Phase::Thinking) {
        if (tokens_in_think_ < max_thinking_tokens_) {
            return true;
        }
        if (close_pos_ >= end_think_token_ids_.size()) {
            RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher invalid close_pos while forcing think end: "
                                "close_pos=%zu, end_token_count=%zu, tokens_in_think=%ld, budget=%ld",
                                close_pos_,
                                end_think_token_ids_.size(),
                                tokens_in_think_,
                                max_thinking_tokens_);
            return false;
        }
        if (!intersectBitmaskRowWithToken(bitmask, idx, end_think_token_ids_[close_pos_])) {
            RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher failed to force think-end token: "
                                "phase=%s, token=%d, idx=%d, tokens_in_think=%ld, budget=%ld, close_pos=%zu",
                                phaseName(phase_),
                                end_think_token_ids_[close_pos_],
                                idx,
                                tokens_in_think_,
                                max_thinking_tokens_,
                                close_pos_);
            return false;
        }
        return true;
    }

    if (phase_ == Phase::ClosingThink) {
        if (close_pos_ >= end_think_token_ids_.size()) {
            RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher invalid close_pos while closing think: "
                                "close_pos=%zu, end_token_count=%zu, accepted=%ld",
                                close_pos_,
                                end_think_token_ids_.size(),
                                total_accepted_);
            return false;
        }
        if (!intersectBitmaskRowWithToken(bitmask, idx, end_think_token_ids_[close_pos_])) {
            RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher failed to force closing think-end token: "
                                "token=%d, idx=%d, close_pos=%zu/%zu, accepted=%ld",
                                end_think_token_ids_[close_pos_],
                                idx,
                                close_pos_,
                                end_think_token_ids_.size(),
                                total_accepted_);
            return false;
        }
        return true;
    }

    if (!inner_->fillBitmask(bitmask, idx)) {
        RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher inner grammar failed to fill mask after think: "
                            "accepted=%ld, tokens_in_think=%ld",
                            total_accepted_,
                            tokens_in_think_);
        return false;
    }
    clearBitmaskRowToken(bitmask, idx, end_think_token_ids_[0]);
    if (think_begin_id_) {
        clearBitmaskRowToken(bitmask, idx, *think_begin_id_);
    }
    return true;
}

bool BudgetedReasoningGrammarMatcher::acceptToken(int32_t token_id) {
    const Snapshot before{phase_, tokens_in_think_, close_pos_, total_accepted_};

    switch (phase_) {
        case Phase::BeforeThink:
            if (!think_begin_id_ || token_id != *think_begin_id_) {
                RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher rejected token before think: token=%d, "
                                    "expected_begin=%d, accepted=%ld",
                                    token_id,
                                    think_begin_id_ ? *think_begin_id_ : -1,
                                    total_accepted_);
                return false;
            }
            pushHistory(before, /*accepted_by_inner=*/false);
            ++total_accepted_;
            phase_ = Phase::Thinking;
            break;
        case Phase::Thinking:
            pushHistory(before, /*accepted_by_inner=*/false);
            ++total_accepted_;
            if (token_id == end_think_token_ids_[0]) {
                close_pos_ = 1;
                if (close_pos_ == end_think_token_ids_.size()) {
                    phase_     = Phase::AfterThink;
                    close_pos_ = 0;
                } else {
                    phase_ = Phase::ClosingThink;
                }
            } else {
                ++tokens_in_think_;
            }
            break;
        case Phase::ClosingThink:
            if (close_pos_ >= end_think_token_ids_.size() || token_id != end_think_token_ids_[close_pos_]) {
                RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher rejected token while closing think: token=%d, "
                                    "expected=%d, close_pos=%zu/%zu, accepted=%ld",
                                    token_id,
                                    close_pos_ < end_think_token_ids_.size() ? end_think_token_ids_[close_pos_] : -1,
                                    close_pos_,
                                    end_think_token_ids_.size(),
                                    total_accepted_);
                return false;
            }
            pushHistory(before, /*accepted_by_inner=*/false);
            ++total_accepted_;
            ++close_pos_;
            if (close_pos_ == end_think_token_ids_.size()) {
                phase_     = Phase::AfterThink;
                close_pos_ = 0;
            }
            break;
        case Phase::AfterThink:
            if (!inner_->acceptToken(token_id)) {
                RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher inner grammar rejected token after think: "
                                    "token=%d, accepted=%ld, tokens_in_think=%ld",
                                    token_id,
                                    total_accepted_,
                                    tokens_in_think_);
                return false;
            }
            pushHistory(before, /*accepted_by_inner=*/true);
            ++total_accepted_;
            break;
    }
    return true;
}

bool BudgetedReasoningGrammarMatcher::isTerminated() const {
    return phase_ == Phase::AfterThink && inner_->isTerminated();
}

bool BudgetedReasoningGrammarMatcher::isPassthroughForMask() const {
    return phase_ == Phase::Thinking && tokens_in_think_ < max_thinking_tokens_;
}

void BudgetedReasoningGrammarMatcher::rollback(int n) {
    if (n <= 0) {
        return;
    }
    if (static_cast<size_t>(n) > size_) {
        RTP_LLM_LOG_WARNING("BudgetedReasoningGrammarMatcher rollback exceeds history window: n=%d, history_size=%zu, "
                            "accepted=%ld, phase=%s",
                            n,
                            size_,
                            total_accepted_,
                            phaseName(phase_));
        throw std::runtime_error("BudgetedReasoningGrammarMatcher::rollback exceeds history window");
    }
    Snapshot restored{phase_, tokens_in_think_, close_pos_, total_accepted_};
    int      inner_rollback = 0;
    for (int i = 0; i < n; ++i) {
        auto entry = popHistory();
        restored   = entry.before;
        if (entry.accepted_by_inner) {
            ++inner_rollback;
        }
    }
    if (inner_rollback > 0) {
        inner_->rollback(inner_rollback);
    }
    phase_           = restored.phase;
    tokens_in_think_ = restored.tokens_in_think;
    close_pos_       = restored.close_pos;
    total_accepted_  = restored.total_accepted;
}

int64_t BudgetedReasoningGrammarMatcher::numAcceptedTokens() const {
    return total_accepted_;
}

int32_t BudgetedReasoningGrammarMatcher::vocabSize() const {
    return inner_->vocabSize();
}

void BudgetedReasoningGrammarMatcher::markFinished() {
    inner_->markFinished();
}

bool BudgetedReasoningGrammarMatcher::finished() const {
    return inner_->finished();
}

}  // namespace rtp_llm
