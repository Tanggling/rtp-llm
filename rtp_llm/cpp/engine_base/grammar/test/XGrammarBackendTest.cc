// XGrammarBackend + RtpGrammarMatcher unit tests (native-C++ path, no Python).

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

namespace rtp_llm {
namespace {

// 128-char ASCII fixture vocab — enough to construct TokenizerInfo + trie.
xgrammar::TokenizerInfo makeTokenizerInfo() {
    std::vector<std::string> vocab;
    vocab.reserve(128);
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    return xgrammar::TokenizerInfo(vocab,
                                   xgrammar::VocabType::RAW,
                                   /*vocab_size=*/128,
                                   /*stop_token_ids=*/std::vector<int32_t>{0});
}

XGrammarBackendOptions defaultOptions() {
    XGrammarBackendOptions opts;
    opts.any_whitespace       = true;
    opts.strict_mode          = true;
    opts.max_compiler_threads = 2;
    opts.compiler_cache_bytes = -1;
    return opts;
}

TEST(XGrammarBackendTest, ConstructFromDirectTokenizerInfo) {
    XGrammarBackend backend(makeTokenizerInfo(), defaultOptions());
    EXPECT_TRUE(backend.isEnabled());
}

TEST(XGrammarBackendTest, CompileBuiltinJSONViaSentinel) {
    XGrammarBackend backend(makeTokenizerInfo(), defaultOptions());

    GrammarKeyCpp key{"json", "$$ANY$$"};
    auto          result = backend.compile(key);
    ASSERT_TRUE(result.compiled) << "$$ANY$$ should map to builtin JSON grammar";
    EXPECT_FALSE(result.is_invalid);
    EXPECT_GT(result.compiled->MemorySizeBytes(), 0u);
}

TEST(XGrammarBackendTest, CompileSimpleJsonSchema) {
    XGrammarBackend backend(makeTokenizerInfo(), defaultOptions());
    GrammarKeyCpp   key{"json", R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})"};

    auto result = backend.compile(key);
    ASSERT_TRUE(result.compiled);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_TRUE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CompileMalformedJsonSchemaIsInvalid) {
    XGrammarBackend backend(makeTokenizerInfo(), defaultOptions());
    // Malformed JSON must surface as cacheable is_invalid, not throw.
    GrammarKeyCpp key{"json", "{this is not json at all"};

    auto result = backend.compile(key);
    EXPECT_FALSE(result.compiled);
    EXPECT_TRUE(result.is_invalid);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CompileStructuralTagWithBoundedAnyText) {
    XGrammarBackend backend(makeTokenizerInfo(), defaultOptions());
    GrammarKeyCpp   key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                        R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":1},"end":"z"},)"
                        R"({"type":"regex","pattern":"a"}]}})"};

    auto result = backend.compile(key);
    ASSERT_TRUE(result.compiled);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_TRUE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CompileStructuralTagWithBoundedAnyTextTokenEnd) {
    XGrammarBackend backend(makeTokenizerInfo(), defaultOptions());
    GrammarKeyCpp   key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                        R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":1},)"
                        R"("end":{"type":"token","token":122}},)"
                        R"({"type":"regex","pattern":"a"}]}})"};

    auto result = backend.compile(key);
    ASSERT_TRUE(result.compiled);
    EXPECT_FALSE(result.is_invalid);
    EXPECT_TRUE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CompileStructuralTagRejectsMultipleBoundedRegions) {
    XGrammarBackend backend(makeTokenizerInfo(), defaultOptions());
    GrammarKeyCpp   key{"structural_tag",
                      R"({"type":"structural_tag","format":{"type":"sequence","elements":[)"
                        R"({"type":"tag","begin":"","content":{"type":"any_text","max_tokens":1},"end":"z"},)"
                        R"({"type":"any_text","max_tokens":1}]}})"};

    auto result = backend.compile(key);
    EXPECT_FALSE(result.compiled);
    EXPECT_TRUE(result.is_invalid);
    EXPECT_FALSE(result.error_message.empty());
}

TEST(XGrammarBackendTest, CreateMatcherProducesUsableObject) {
    XGrammarBackend backend(makeTokenizerInfo(), defaultOptions());
    auto            result = backend.compile({"json", "$$ANY$$"});
    ASSERT_TRUE(result.compiled);

    auto matcher = backend.createMatcher(result.compiled);
    ASSERT_TRUE(matcher);
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
    EXPECT_FALSE(matcher->isTerminated());
}

// ---- RtpGrammarMatcher rollback ----------------------------------------

TEST(RtpGrammarMatcherTest, RollbackRestoresAcceptedCount) {
    XGrammarBackend backend(makeTokenizerInfo(), defaultOptions());
    auto            result = backend.compile({"regex", "a"});
    ASSERT_TRUE(result.compiled);

    auto          matcher = backend.createMatcher(result.compiled);
    constexpr int kA      = 'a';
    EXPECT_TRUE(matcher->acceptToken(kA));
    EXPECT_EQ(matcher->numAcceptedTokens(), 1);
    matcher->rollback(1);
    EXPECT_EQ(matcher->numAcceptedTokens(), 0);
}

}  // namespace
}  // namespace rtp_llm
