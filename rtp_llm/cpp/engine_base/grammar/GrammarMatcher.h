#pragma once

#include <cstdint>
#include <vector>

#include <dlpack/dlpack.h>

namespace rtp_llm {

class GrammarMatcher {
public:
    virtual ~GrammarMatcher() = default;

    virtual bool acceptToken(int32_t token_id) = 0;

    virtual bool acceptTokens(const std::vector<int32_t>& tokens) {
        for (int32_t token_id : tokens) {
            if (!acceptToken(token_id)) {
                return false;
            }
        }
        return true;
    }

    virtual bool fillBitmask(DLTensor* bitmask, int32_t idx) = 0;

    virtual bool isTerminated() const = 0;
    virtual void rollback(int n)      = 0;

    virtual int64_t numAcceptedTokens() const = 0;
    virtual int32_t vocabSize() const         = 0;

    virtual void markFinished() = 0;
    virtual bool finished() const = 0;
};

}  // namespace rtp_llm
