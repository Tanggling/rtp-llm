#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

// Glue between MtpExecutor and per-stream SpecLogitsProcessors; emits a GPU bool disallow mask.
class SpecLogitsVerifyRunner {
public:
    struct ActiveProcessor {
        SpecLogitsProcessorPtr processor;
        size_t                 stream_idx = 0;
    };

    struct LaunchTask {
        std::vector<ActiveProcessor> active;
        size_t                       total_streams = 0;
        int                          propose_step  = 0;
        size_t                       vocab_size    = 0;
        torch::Tensor                draft_tokens;  // [B,P] or [B,P+1]
    };

    struct LaunchResult {
        torch::Tensor                          spec_vocab_mask_gpu;  // [rows, V] bool, true=disallow
        bool                                   has_active_processor = false;
        std::vector<std::optional<ErrorInfo>> processor_errors;
        // Keeps the pinned CPU source for the async H2D mask copy alive until the
        // caller has consumed spec_vocab_mask_gpu.
        torch::Tensor spec_vocab_mask_cpu_lifetime;
        torch::Tensor spec_cap_cpu;
    };

    SpecLogitsVerifyRunner() = default;

    SpecLogitsVerifyRunner(const SpecLogitsVerifyRunner&)            = delete;
    SpecLogitsVerifyRunner& operator=(const SpecLogitsVerifyRunner&) = delete;

    LaunchResult run(const LaunchTask& task);
    static void
    applyMaskToLogits(const torch::Tensor& logits, const torch::Tensor& spec_vocab_mask_gpu, size_t vocab_size);

private:
    struct VerifyShape {
        size_t batch_size    = 0;
        int    propose_step  = 0;
        size_t vocab_size    = 0;
        size_t bitmask_words = 0;
        size_t rows          = 0;
        size_t row_words     = 0;
        size_t buffer_rows   = 0;
    };

    struct MergeProcessorMasksResult {
        std::vector<size_t> active_rows;
        std::vector<std::optional<ErrorInfo>> processor_errors;
    };

    void ensureBuffersFit(size_t total_streams, int propose_step, size_t vocab_size, size_t bitmask_words);
    void materializeDraftTokensToCpu(const LaunchTask& task);
    void unpackRowToBoolDisallow(size_t row, size_t vocab_size, size_t bitmask_words);
    std::vector<size_t> resetPreviousActiveRows(const VerifyShape& shape);
    MergeProcessorMasksResult mergeProcessorMasks(const LaunchTask& task, const VerifyShape& shape);
    void                uploadChangedRows(const std::vector<size_t>& rows_to_reset,
                                          const std::vector<size_t>& active_rows,
                                          const VerifyShape&         shape);
    LaunchResult        makeResult(const VerifyShape& shape);

    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;  // [rows, W] int32 packed; bit=1 allow
    torch::Tensor disallow_mask_cpu_;   // [rows, V] pinned bool; true=disallow
    torch::Tensor disallow_mask_gpu_;   // [rows, V] cuda bool; mirrors disallow_mask_cpu_
    torch::Tensor spec_cap_cpu_;
    // Rows last filled by an active grammar processor; lets the next call only
    // reset + re-unpack + re-upload those rows instead of the full B*(P+1) buffer.
    std::vector<size_t> last_active_stream_rows_;
};

}  // namespace rtp_llm
