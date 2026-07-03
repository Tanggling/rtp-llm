#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/engine_base/grammar/GrammarMatcher.h"

namespace rtp_llm {

class BudgetedReasoningGrammarMatcher final: public GrammarMatcher {
public:
    enum class Phase {
        BeforeThink,
        Thinking,
        ClosingThink,
        AfterThink,
    };

    BudgetedReasoningGrammarMatcher(std::shared_ptr<GrammarMatcher> inner,
                                    std::optional<int32_t>          think_begin_id,
                                    std::vector<int32_t>            end_think_token_ids,
                                    int64_t                         max_thinking_tokens,
                                    int                             max_rollback_tokens = 200);

    bool acceptToken(int32_t token_id) override;
    bool fillBitmask(DLTensor* bitmask, int32_t idx) override;

    bool isPassthroughForMask() const override;
    bool isTerminated() const override;
    void rollback(int n) override;

    int64_t numAcceptedTokens() const override;
    int32_t vocabSize() const override;

    void markFinished() override;
    bool finished() const override;

    Phase   phase() const noexcept {
        return phase_;
    }
    int64_t tokensInThink() const noexcept {
        return tokens_in_think_;
    }
    size_t closePos() const noexcept {
        return close_pos_;
    }

private:
    struct Snapshot {
        Phase   phase;
        int64_t tokens_in_think;
        size_t  close_pos;
        int64_t total_accepted;
    };

    struct HistoryEntry {
        Snapshot before;
        bool     accepted_by_inner;
    };

    void pushHistory(const Snapshot& snapshot, bool accepted_by_inner);
    HistoryEntry popHistory();
    bool intersectBitmaskRowWithToken(DLTensor* bitmask, int32_t idx, int32_t token_id) const;
    void clearBitmaskRowToken(DLTensor* bitmask, int32_t idx, int32_t token_id) const;

    std::shared_ptr<GrammarMatcher> inner_;
    Phase                           phase_;
    std::optional<int32_t>          think_begin_id_;
    std::vector<int32_t>            end_think_token_ids_;
    int64_t                         max_thinking_tokens_;
    int64_t                         tokens_in_think_ = 0;
    size_t                          close_pos_       = 0;
    int64_t                         total_accepted_  = 0;

    std::vector<HistoryEntry> history_ring_;
    size_t                    head_ = 0;
    size_t                    size_ = 0;
};

}  // namespace rtp_llm
