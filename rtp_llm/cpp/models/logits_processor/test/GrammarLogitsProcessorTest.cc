// CPU unit tests for GrammarLogitsProcessor::tryAcceptAndFillBitmask over a 128-char ASCII vocab.

#include "rtp_llm/cpp/engine_base/grammar/BudgetedReasoningGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/GrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <xgrammar/tokenizer_info.h>

namespace rtp_llm {
namespace {

std::string makeAsciiTokenizerInfoJson() {
    std::vector<std::string> vocab;
    vocab.reserve(128);
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    xgrammar::TokenizerInfo info(vocab,
                                 xgrammar::VocabType::RAW,
                                 /*vocab_size=*/128,
                                 /*stop_token_ids=*/std::vector<int32_t>{0});
    return info.SerializeJSON();
}

XGrammarBackendOptions defaultOptions() {
    XGrammarBackendOptions opts;
    opts.any_whitespace        = true;
    opts.strict_mode           = true;
    opts.max_compiler_threads  = 2;
    opts.compiler_cache_bytes  = -1;
    return opts;
}

struct ProcessorBundle {
    std::shared_ptr<GrammarLogitsProcessor> proc;
    std::shared_ptr<RtpGrammarMatcher>      matcher;

    GrammarLogitsProcessor* operator->() const noexcept {
        return proc.get();
    }
};

// terminate_without_stop_token=true so the matcher flips IsTerminated() the moment the regex completes.
ProcessorBundle makeProcessor(XGrammarBackend& backend, const std::string& regex) {
    auto compiled = backend.compileNow({"regex", regex}).compiled;
    EXPECT_TRUE(compiled);
    std::shared_ptr<RtpGrammarMatcher> matcher = backend.createMatcher(compiled,
                                                                       /*terminate_without_stop_token=*/true);
    auto                               proc    = std::make_shared<GrammarLogitsProcessor>(matcher);
    return {std::move(proc), std::move(matcher)};
}

bool rowAllows(const std::vector<int32_t>& bm, size_t words, int row, int token) {
    const int32_t word = bm[static_cast<size_t>(row) * words + token / 32];
    return (static_cast<uint32_t>(word) & (1u << (token % 32))) != 0u;
}

class AllowAllMatcher final: public GrammarMatcher {
public:
    bool acceptToken(int32_t token_id) override {
        accepted_tokens_.push_back(token_id);
        return true;
    }
    bool fillBitmask(DLTensor* bitmask, int32_t idx) override {
        ++fill_calls_;
        auto* data = static_cast<int32_t*>(bitmask->data);
        std::fill(data + static_cast<size_t>(idx) * bitmask->shape[1],
                  data + static_cast<size_t>(idx + 1) * bitmask->shape[1],
                  SpecLogitsProcessor::kBitmaskAllowAll);
        return true;
    }
    bool isTerminated() const override {
        return false;
    }
    void rollback(int n) override {
        if (n > static_cast<int>(accepted_tokens_.size())) {
            throw std::runtime_error("rollback overflow");
        }
        accepted_tokens_.resize(accepted_tokens_.size() - static_cast<size_t>(n));
    }
    int64_t numAcceptedTokens() const override {
        return static_cast<int64_t>(accepted_tokens_.size());
    }
    int32_t vocabSize() const override {
        return 128;
    }
    void markFinished() override {
        finished_ = true;
    }
    bool finished() const override {
        return finished_;
    }

    int fillCalls() const {
        return fill_calls_;
    }

private:
    std::vector<int32_t> accepted_tokens_;
    int                  fill_calls_ = 0;
    bool                 finished_   = false;
};

struct BudgetedProcessorBundle {
    std::shared_ptr<GrammarLogitsProcessor>           proc;
    std::shared_ptr<BudgetedReasoningGrammarMatcher>  matcher;
    std::shared_ptr<AllowAllMatcher>                  inner;

    GrammarLogitsProcessor* operator->() const noexcept {
        return proc.get();
    }
};

BudgetedProcessorBundle makeBudgetedProcessor(int64_t budget, std::vector<int32_t> end_tokens) {
    auto inner   = std::make_shared<AllowAllMatcher>();
    auto matcher = std::make_shared<BudgetedReasoningGrammarMatcher>(inner,
                                                                     /*think_begin_id=*/std::nullopt,
                                                                     std::move(end_tokens),
                                                                     budget);
    auto proc = std::make_shared<GrammarLogitsProcessor>(matcher, /*eos_token_id=*/0);
    return {std::move(proc), std::move(matcher), std::move(inner)};
}

constexpr int kA   = 'a';  // token id 97
constexpr int kB   = 'b';  // token id 98
constexpr int kX   = 'x';  // token id 120
constexpr int kEos = 0;    // stop token in makeAsciiTokenizerInfoJson
constexpr int kZ   = 'z';

}  // namespace

// regex "ab": legal sequence is 'a' then 'b'. A fully-legal draft chain should
// return cap == propose_step with each row constraining to the expected token.
TEST(GrammarLogitsProcessorTest, AcceptsLegalDraftChain) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, propose_step) << "every draft token is grammar-legal";
    EXPECT_TRUE(rowAllows(bm, words, 0, kA)) << "row 0 must allow 'a'";
    EXPECT_FALSE(rowAllows(bm, words, 0, kB)) << "row 0 must NOT allow 'b' at the start";
    EXPECT_TRUE(rowAllows(bm, words, 1, kB)) << "row 1 (after 'a') must allow 'b'";
}

// A draft token that violates the grammar caps at that offset.
TEST(GrammarLogitsProcessorTest, CapsAtFirstIllegalDraftToken) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kX};  // 'a' ok, then 'x' illegal (expected 'b')

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, 1) << "draft[1]='x' is illegal after 'a', so cap == 1";
}

// Regression for the terminated-matcher guard: once the grammar terminates
// (after "ab"), verify must stop with cap == 2 and must NOT call acceptToken on
// the already-terminated matcher for trailing draft tokens (including EOS).
TEST(GrammarLogitsProcessorTest, TerminatedMatcherLeavesAllowAllWithoutCrash) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 3;  // one past the grammar's natural end
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, SpecLogitsProcessor::kBitmaskAllowAll);
    std::vector<int32_t> draft{kA, kB, kA, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    int cap = 0;
    ASSERT_NO_THROW({ cap = proc->tryAcceptAndFillBitmask(req); });
    EXPECT_EQ(cap, 2) << "grammar completes after 'ab'; trailing draft must not advance matcher";
    EXPECT_FALSE(proc.matcher.get()->isTerminated()) << "provisional accepts must roll back committed matcher state";
    // Unfilled rows stay at allow-all (init + fill_row never reached offset 2+).
    EXPECT_TRUE(rowAllows(bm, words, 3, kX));
}

TEST(GrammarLogitsProcessorTest, VerifyCapStopsWhenDraftContinuesWithEos) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 3;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kB, kEos, kX};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, 2);
    EXPECT_FALSE(proc.matcher.get()->isTerminated());
}

TEST(GrammarLogitsProcessorTest, VerifyCapZeroWhenGrammarAlreadyComplete) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    ASSERT_TRUE(proc.matcher.get()->acceptToken(kA));
    ASSERT_TRUE(proc.matcher.get()->acceptToken(kB));
    EXPECT_TRUE(proc.matcher->isTerminated());

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kX, kX, kX};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(proc->tryAcceptAndFillBitmask(req), 0);
    EXPECT_TRUE(proc.matcher->isTerminated());
}

// tryAcceptAndFillBitmask must leave the matcher's committed state unchanged
// (it rolls back any provisional accepts): a second identical call yields the
// same cap.
TEST(GrammarLogitsProcessorTest, RollsBackProvisionalAccepts) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int r1 = proc->tryAcceptAndFillBitmask(req);
    const int r2 = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(r1, r2) << "state must be unchanged across calls (rollback)";
}

TEST(GrammarLogitsProcessorTest, VerifyCapIsDraftRejectIndex) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> bad_draft{kX, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = bad_draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(proc->tryAcceptAndFillBitmask(req), 0);
}

// Undersized bitmask buffer must error out as GRAMMAR_BITMASK_BUFFER_TOO_SMALL, not corrupt the caller's heap.
TEST(GrammarLogitsProcessorTest, RejectsUndersizedBitmaskBuffer) {
    XGrammarBackend backend(makeAsciiTokenizerInfoJson(), defaultOptions());
    auto            proc = makeProcessor(backend, "ab");

    const int propose_step = 1;
    // Allocate a deliberately too-small buffer for vocab=128 (needs 4 words).
    const size_t         words = 1;
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kA};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    EXPECT_EQ(proc->tryAcceptAndFillBitmask(req), 0);
    ASSERT_TRUE(proc->hasError());
    EXPECT_EQ(proc->error().code(), ErrorCode::GRAMMAR_BITMASK_BUFFER_TOO_SMALL);
}

TEST(GrammarLogitsProcessorTest, ClearBitmaskTokenRangeClearsFullWordsAndEdges) {
    const size_t         words = SpecLogitsProcessor::bitmaskWordCount(96);
    std::vector<int32_t> bm(words, SpecLogitsProcessor::kBitmaskAllowAll);

    clearBitmaskTokenRange(bm.data(), words, 35, 70);

    EXPECT_TRUE(rowAllows(bm, words, 0, 34));
    EXPECT_FALSE(rowAllows(bm, words, 0, 35));
    EXPECT_FALSE(rowAllows(bm, words, 0, 63));
    EXPECT_FALSE(rowAllows(bm, words, 0, 64));
    EXPECT_FALSE(rowAllows(bm, words, 0, 69));
    EXPECT_TRUE(rowAllows(bm, words, 0, 70));
}

TEST(GrammarLogitsProcessorTest, BudgetedReasoningPassthroughRowsMaskEos) {
    auto proc = makeBudgetedProcessor(/*budget=*/100, /*end_tokens=*/{kZ});

    const int            propose_step = 3;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kX, kA, kEos, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, 2) << "draft[2]=EOS is masked in think passthrough";

    EXPECT_TRUE(rowAllows(bm, words, 0, kX));
    EXPECT_TRUE(rowAllows(bm, words, 1, kA));
    EXPECT_TRUE(rowAllows(bm, words, 2, kB));
    EXPECT_FALSE(rowAllows(bm, words, 0, kEos));
    EXPECT_FALSE(rowAllows(bm, words, 1, kEos));
    EXPECT_FALSE(rowAllows(bm, words, 2, kEos));
    EXPECT_EQ(proc.matcher->phase(), BudgetedReasoningGrammarMatcher::Phase::Thinking);
    EXPECT_EQ(proc.matcher->tokensInThink(), 0);
    EXPECT_EQ(proc.inner->fillCalls(), 0);
}

TEST(GrammarLogitsProcessorTest, BudgetedReasoningForcesEndWhenBudgetExhausted) {
    auto proc = makeBudgetedProcessor(/*budget=*/1, /*end_tokens=*/{kZ});

    const int            propose_step = 3;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kX, kZ, kA};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, propose_step);

    EXPECT_TRUE(rowAllows(bm, words, 0, kX));

    EXPECT_TRUE(rowAllows(bm, words, 1, kZ));
    EXPECT_FALSE(rowAllows(bm, words, 1, kX));
    EXPECT_FALSE(rowAllows(bm, words, 1, kA));

    EXPECT_TRUE(rowAllows(bm, words, 2, kA));
    EXPECT_TRUE(rowAllows(bm, words, 3, kB));

    EXPECT_EQ(proc->committedOutputLen(), 0);
    EXPECT_EQ(proc.matcher->numAcceptedTokens(), 0);
    EXPECT_EQ(proc.matcher->phase(), BudgetedReasoningGrammarMatcher::Phase::Thinking);
    EXPECT_EQ(proc.inner->fillCalls(), 2);
}

TEST(GrammarLogitsProcessorTest, BudgetedReasoningForcesMultiTokenEndSequence) {
    auto proc = makeBudgetedProcessor(/*budget=*/0, /*end_tokens=*/{kZ, kB});

    const int            propose_step = 2;
    const size_t         words        = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(static_cast<size_t>(propose_step + 1) * words, 0);
    std::vector<int32_t> draft{kZ, kB};

    SpecLogitsProcessorRequest req;
    req.draft_tokens       = draft.data();
    req.propose_step       = propose_step;
    req.bitmask_cpu_out    = bm.data();
    req.bitmask_size_int32 = words;
    req.vocab_size         = 128;

    const int cap = proc->tryAcceptAndFillBitmask(req);
    EXPECT_EQ(cap, propose_step);
    EXPECT_TRUE(rowAllows(bm, words, 0, kZ));
    EXPECT_FALSE(rowAllows(bm, words, 0, kB));
    EXPECT_TRUE(rowAllows(bm, words, 1, kB));
    EXPECT_FALSE(rowAllows(bm, words, 1, kZ));
    EXPECT_EQ(proc.matcher->numAcceptedTokens(), 0);
}

TEST(GrammarLogitsProcessorTest, BudgetedReasoningMasksThinkBoundaryTokensAfterThink) {
    auto inner   = std::make_shared<AllowAllMatcher>();
    auto matcher = std::make_shared<BudgetedReasoningGrammarMatcher>(inner,
                                                                     /*think_begin_id=*/std::optional<int32_t>(kX),
                                                                     /*end_think_token_ids=*/std::vector<int32_t>{kZ, kB},
                                                                     /*max_thinking_tokens=*/0);

    ASSERT_TRUE(matcher->acceptToken(kX));
    ASSERT_TRUE(matcher->acceptToken(kZ));
    ASSERT_TRUE(matcher->acceptToken(kB));
    ASSERT_EQ(matcher->phase(), BudgetedReasoningGrammarMatcher::Phase::AfterThink);

    const size_t         words = SpecLogitsProcessor::bitmaskWordCount(128);
    std::vector<int32_t> bm(words, 0);
    int64_t              dl_shape[2];
    DLTensor             dl = makeSingleRowBitmaskView(bm.data(), static_cast<int32_t>(words), dl_shape);

    ASSERT_TRUE(matcher->fillBitmask(&dl, 0));
    EXPECT_FALSE(rowAllows(bm, words, 0, kX)) << "begin-think token must not re-open think after AfterThink";
    EXPECT_FALSE(rowAllows(bm, words, 0, kZ)) << "END[0] must not start another think close tag after AfterThink";
    EXPECT_TRUE(rowAllows(bm, words, 0, kB)) << "END[1] may be an ordinary token such as newline";
    EXPECT_TRUE(rowAllows(bm, words, 0, kA));
    EXPECT_EQ(inner->fillCalls(), 1);
}

}  // namespace rtp_llm
