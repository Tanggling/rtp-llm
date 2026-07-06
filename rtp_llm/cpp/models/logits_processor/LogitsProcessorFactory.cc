#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/models/logits_processor/RecommendationLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

GrammarKeyCpp keyFromGenerateConfig(const GenerateConfig& config) {
    // Fixed priority json_schema > regex > ebnf > structural_tag silently drops the
    // others when a caller sets multiple. Warn so we can audit the call site.
    const bool multi_grammar = (config.json_schema.has_value() + config.regex.has_value() + config.ebnf.has_value()
                                + config.structural_tag.has_value())
                               > 1;
    if (multi_grammar) {
        RTP_LLM_LOG_WARNING("GenerateConfig sets multiple grammar fields simultaneously; "
                            "applying priority json_schema>regex>ebnf>structural_tag");
    }
    if (config.json_schema.has_value()) {
        return {"json", config.json_schema.value()};
    }
    if (config.regex.has_value()) {
        return {"regex", config.regex.value()};
    }
    if (config.ebnf.has_value()) {
        return {"ebnf", config.ebnf.value()};
    }
    if (config.structural_tag.has_value()) {
        return {"structural_tag", config.structural_tag.value()};
    }
    // response_format envelope is projected to typed fields above in Python ResponseFormatBuilder.
    // The C++ engine only consumes typed fields.
    return {};
}

ErrorResult<BaseLogitsProcessorPtr>
createGrammarProcessor(const LogitsProcessorFactoryParams& params, const GrammarKeyCpp& key, int64_t eos_token_id) {
    const auto& input = params.generate_input;
    if (!input || !input->generate_config || key.empty()) {
        return BaseLogitsProcessorPtr{};
    }
    auto& config = *input->generate_config;
    if (!params.grammar_backend) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS,
                         "structured output requested but constraint backend is disabled "
                         "(check engine startup logs: tokenizer info empty or backend init failed).");
    }

    const bool terminate_without_stop_token = config.grammar_terminate_without_stop_token || key.key_type == "json";
    RTP_LLM_LOG_DEBUG("grammar matcher install: type=%s, len=%zu, in_think_mode=%d, "
                      "terminate_without_stop_token=%d",
                      key.key_type.c_str(),
                      key.key_string.size(),
                      static_cast<int>(config.in_think_mode),
                      static_cast<int>(terminate_without_stop_token));

    auto matcher_or = params.grammar_backend->createMatcherFromKey(key, terminate_without_stop_token);
    if (!matcher_or.ok()) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, std::string(matcher_or.status().message()));
    }
    return BaseLogitsProcessorPtr(
        std::make_shared<GrammarLogitsProcessor>(std::move(matcher_or.value()), eos_token_id));
}

}  // namespace

void LogitsProcessorFactory::init(const std::string& ckpt_path, const std::string& tree_decode_config) {
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(ckpt_path, tree_decode_config);
}

ErrorResult<std::vector<BaseLogitsProcessorPtr>>
LogitsProcessorFactory::createLogitsProcessors(const LogitsProcessorFactoryParams& params) {
    std::vector<BaseLogitsProcessorPtr> result;

    const auto& generate_input = params.generate_input;
    auto&       config         = *generate_input->generate_config;

    GrammarKeyCpp grammar_key = keyFromGenerateConfig(config);

    auto think_processor = ThinkModeLogitsProcessor::fromGenerateInput(generate_input, params.max_batch_size);
    if (think_processor != nullptr && grammar_key.empty()) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(think_processor));
    }

    if (!grammar_key.empty()) {
        if (config.hasNumBeams() || config.num_return_sequences > 1) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS,
                             "grammar-constrained decoding does not support beam search or "
                             "num_return_sequences > 1");
        }
        auto grammar_processor_result = createGrammarProcessor(params, grammar_key, params.eos_token_id);
        if (!grammar_processor_result.ok()) {
            return grammar_processor_result.status();
        }
        result.push_back(std::move(grammar_processor_result.value()));
    }

    auto tree_processor = TreeLogitsProcessor::fromGenerateInput(generate_input, params.init_batch_size);
    if (tree_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(tree_processor));
    }

    auto rec_processor = RecommendationLogitsProcessor::fromGenerateInput(generate_input, params.init_batch_size);
    if (rec_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(rec_processor));
    }

    auto multi_seq_processor = MultiSeqLogitsProcessor::fromGenerateInput(generate_input, params.eos_token_id);
    if (multi_seq_processor != nullptr) {
        result.push_back(std::static_pointer_cast<BaseLogitsProcessor>(multi_seq_processor));
    }

    return std::move(result);
}

}  // namespace rtp_llm
