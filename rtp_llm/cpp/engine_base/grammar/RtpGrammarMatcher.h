#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <dlpack/dlpack.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

namespace rtp_llm {

// Per-stream xgrammar::GrammarMatcher adapter.
class RtpGrammarMatcher final {
public:
    RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                      std::optional<std::vector<int32_t>>        override_stop_tokens         = std::nullopt,
                      bool                                       terminate_without_stop_token = false,
                      int                                        max_rollback_tokens          = 200);

    RtpGrammarMatcher(const RtpGrammarMatcher&)            = delete;
    RtpGrammarMatcher& operator=(const RtpGrammarMatcher&) = delete;
    RtpGrammarMatcher(RtpGrammarMatcher&&)                 = default;
    RtpGrammarMatcher& operator=(RtpGrammarMatcher&&)      = default;

    [[nodiscard]] bool acceptToken(int32_t token_id);
    [[nodiscard]] bool acceptTokens(const std::vector<int32_t>& tokens);

    bool fillBitmask(DLTensor* bitmask, int32_t idx);

    bool isTerminated() const;
    void rollback(int n);

    int64_t numAcceptedTokens() const {
        return num_accepted_;
    }
    int32_t vocabSize() const {
        return compiled_->GetTokenizerInfo().GetVocabSize();
    }

    void markFinished() {
        finished_ = true;
    }
    bool finished() const {
        return finished_;
    }

private:
    std::shared_ptr<xgrammar::CompiledGrammar> compiled_;
    std::unique_ptr<xgrammar::GrammarMatcher>  matcher_;

    int64_t num_accepted_ = 0;
    bool    finished_     = false;
};

}  // namespace rtp_llm
