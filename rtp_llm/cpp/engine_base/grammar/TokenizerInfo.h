#pragma once

#include <memory>
#include <utility>

namespace rtp_llm {

class ModelConfig;
class XGrammarBackend;

// Runtime tokenizer-info handle. Keeps xgrammar's constructor-built derived
// indexes alive without exposing xgrammar symbols from this public API.
class TokenizerInfo {
public:
    TokenizerInfo() = default;

    // Returns empty() on failure (grammar silently disabled). Does not throw.
    // model_vocab_size (model_config.vocab_size) widens the grammar vocab over a padded model vocab.
    static TokenizerInfo fromHuggingFaceTokenizer(const ModelConfig& model_config) noexcept;

    bool empty() const noexcept {
        return !direct_info_;
    }

private:
    explicit TokenizerInfo(std::shared_ptr<const void> direct_info): direct_info_(std::move(direct_info)) {}

    std::shared_ptr<const void> direct_info_;
    friend class XGrammarBackend;
};

}  // namespace rtp_llm
