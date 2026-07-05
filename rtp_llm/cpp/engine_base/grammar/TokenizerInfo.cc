#include "rtp_llm/cpp/engine_base/grammar/TokenizerInfo.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/config/SpecialTokens.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarTokenizerInfoCooker.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

// Single-token EOS / stop ids from a model's SpecialTokens, deduped + sorted.
// Multi-token stop sequences are dropped — xgrammar's stop_token_ids is per-token.
std::vector<int32_t> collectStopTokenIds(const SpecialTokens& st) {
    std::vector<int32_t> ids;
    auto                 add_id = [&](int64_t id) {
        if (id < 0)
            return;
        const auto v = static_cast<int32_t>(id);
        if (std::find(ids.begin(), ids.end(), v) == ids.end()) {
            ids.push_back(v);
        }
    };
    add_id(st.eos_token_id);
    for (const auto& seq : st.stop_words_id_list) {
        if (seq.size() == 1) {
            add_id(seq[0]);
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

}  // namespace

TokenizerInfo TokenizerInfo::fromHuggingFaceTokenizer(const ModelConfig& model_config) noexcept {
    try {
        const auto stops       = collectStopTokenIds(model_config.special_tokens);
        auto       info        = std::make_shared<xgrammar::TokenizerInfo>(xgrammar_impl::cookTokenizerInfo(
            model_config.tokenizer_vocab, model_config.tokenizer_backend_str, stops, model_config.vocab_size));
        auto       direct_info = std::static_pointer_cast<const void>(info);
        RTP_LLM_LOG_INFO(
            "TokenizerInfo: cooked direct vocab_size=%d, stop_tokens=%zu", info->GetVocabSize(), stops.size());
        return TokenizerInfo(std::move(direct_info));
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("TokenizerInfo cook failed (%s); grammar disabled", e.what());
        return TokenizerInfo();
    } catch (...) {
        RTP_LLM_LOG_ERROR("TokenizerInfo cook failed (unknown exception); grammar disabled");
        return TokenizerInfo();
    }
}

}  // namespace rtp_llm
