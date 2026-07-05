#pragma once

#include <cstdint>

#include "rtp_llm/cpp/engine_base/grammar/GrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {
namespace grammar_spec_verify {

int verify(GrammarMatcher& matcher,
           int64_t eos_token_id,
           const SpecLogitsProcessorRequest& request,
           ErrorInfo& out_err);

}  // namespace grammar_spec_verify
}  // namespace rtp_llm
