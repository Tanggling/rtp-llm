#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class ModelConfig;
struct GrammarConfig;

class LogitsProcessorFactory {
public:
    // Engine-scoped init: owns tokenizer-bound grammar backend and loads tree-decode prefix dictionary.
    LogitsProcessorFactory(const ModelConfig& model_config,
                           const GrammarConfig& grammar_config,
                           const std::string& tree_decode_config);

    // Build-time errors come back as non-ok ErrorResult; caller surfaces on the stream.
    ErrorResult<std::vector<BaseLogitsProcessorPtr>>
    createLogitsProcessors(std::shared_ptr<GenerateInput> generate_input,
                           int32_t init_batch_size,
                           int32_t max_batch_size,
                           int64_t eos_token_id) const;

private:
    std::shared_ptr<XGrammarBackend> grammar_backend_;
};

}  // namespace rtp_llm
