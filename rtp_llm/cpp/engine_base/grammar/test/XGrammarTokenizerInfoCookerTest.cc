#include <gtest/gtest.h>

#include "autil/legacy/json.h"

namespace rtp_llm::xgrammar_impl {
namespace {

// Regression: DetectMetadataFromHF returns vocab_type as JsonNumber (not int64_t Any).
// cookTokenizerInfoOpaque must parse it via IsJsonNumber/JsonNumberCast.
TEST(XGrammarTokenizerInfoCookerTest, ParsesAutilJsonNumberVocabType) {
    const auto any = autil::legacy::json::ParseJson(R"({"vocab_type":2,"add_prefix_space":false})");
    const auto*  map = autil::legacy::AnyCast<autil::legacy::json::JsonMap>(&any);
    ASSERT_NE(map, nullptr);
    EXPECT_TRUE(autil::legacy::json::IsJsonNumber(map->at("vocab_type")));
    EXPECT_EQ(autil::legacy::json::JsonNumberCast<int64_t>(map->at("vocab_type")), 2);
}

}  // namespace
}  // namespace rtp_llm::xgrammar_impl
