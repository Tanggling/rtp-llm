#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

#include "rtp_llm/cpp/engine_base/grammar/GrammarMatcher.h"

namespace rtp_llm {

// Per-stream xgrammar::GrammarMatcher adapter.
class RtpGrammarMatcher final: public GrammarMatcher {
public:
    RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                      std::optional<std::vector<int32_t>>        override_stop_tokens         = std::nullopt,
                      bool                                       terminate_without_stop_token = false,
                      int                                        max_rollback_tokens          = 200);

    RtpGrammarMatcher(const RtpGrammarMatcher&)            = delete;
    RtpGrammarMatcher& operator=(const RtpGrammarMatcher&) = delete;
    RtpGrammarMatcher(RtpGrammarMatcher&&)                 = default;
    RtpGrammarMatcher& operator=(RtpGrammarMatcher&&)      = default;

    [[nodiscard]] bool acceptToken(int32_t token_id) override;
    [[nodiscard]] bool acceptTokens(const std::vector<int32_t>& tokens) override;

    bool fillBitmask(DLTensor* bitmask, int32_t idx) override;

    bool isTerminated() const override;
    void rollback(int n) override;

    int64_t numAcceptedTokens() const override {
        return num_accepted_;
    }
    int32_t vocabSize() const override {
        return compiled_->GetTokenizerInfo().GetVocabSize();
    }

    void markFinished() override {
        finished_ = true;
    }
    bool finished() const override {
        return finished_;
    }

private:
    std::shared_ptr<xgrammar::CompiledGrammar> compiled_;
    std::unique_ptr<xgrammar::GrammarMatcher>  matcher_;

    int64_t num_accepted_ = 0;
    bool    finished_     = false;
};

}  // namespace rtp_llm
