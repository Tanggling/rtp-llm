#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <thread>
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

using namespace std;

namespace rtp_llm {

void EplbPlanBuffers::init(size_t    log_exp_num,
                           size_t    phy_exp_num,
                           size_t    hidden_size,
                           size_t    moe_size,
                           size_t    ep_size,
                           DataType  dtype,
                           QuantAlgo quant_algo) {
    auto gpu_i32 = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    auto cpu_i32 = torch::kInt32;

    layer_id_buf          = torch::zeros({1}, gpu_i32);
    logic_expert_cnt      = torch::zeros({(int64_t)log_exp_num}, gpu_i32);
    logic_expert_cnt_host = torch::zeros({(int64_t)log_exp_num}, cpu_i32).pin_memory();
    log2phy               = torch::zeros({(int64_t)log_exp_num, (int64_t)(phy_exp_num - log_exp_num + 1)}, gpu_i32);
    log2phy_host = torch::zeros({(int64_t)log_exp_num, (int64_t)(phy_exp_num - log_exp_num + 1)}, cpu_i32).pin_memory();
    phy2log      = torch::zeros({(int64_t)phy_exp_num}, gpu_i32);

    size_t expert_per_ep = phy_exp_num / ep_size;

    // note: only support fp8 per token quant
    if (quant_algo.isFp8()) {
        is_quantized    = true;
        int  group_size = quant_algo.getGroupSize();
        auto gpu_fp8    = torch::TensorOptions(torch::kFloat8_e4m3fn).device(torch::kCUDA);
        auto gpu_fp32   = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);

        moe_weight_1 = torch::zeros({(int64_t)expert_per_ep, (int64_t)(moe_size * 2), (int64_t)hidden_size}, gpu_fp8);
        moe_scale_1  = torch::zeros(
            {(int64_t)expert_per_ep, (int64_t)(moe_size * 2 / group_size), (int64_t)(hidden_size / group_size)},
            gpu_fp32);
        moe_weight_2 = torch::zeros({(int64_t)expert_per_ep, (int64_t)hidden_size, (int64_t)moe_size}, gpu_fp8);
        moe_scale_2  = torch::zeros(
            {(int64_t)expert_per_ep, (int64_t)(hidden_size / group_size), (int64_t)(moe_size / group_size)}, gpu_fp32);
    } else {
        is_quantized   = false;
        auto gpu_dtype = torch::TensorOptions(dataTypeToTorchType(dtype)).device(torch::kCUDA);
        moe_weight_1 = torch::zeros({(int64_t)expert_per_ep, (int64_t)(moe_size * 2), (int64_t)hidden_size}, gpu_dtype);
        moe_weight_2 = torch::zeros({(int64_t)expert_per_ep, (int64_t)hidden_size, (int64_t)moe_size}, gpu_dtype);
    }
}

void BalanceStatsBuffers::init(int layer_num, int log_exp_num, int ep_size) {
    // host tensors
    log_stats = torch::zeros({layer_num, log_exp_num}, torch::kInt32);
    gpu_loads = torch::zeros({layer_num, ep_size}, torch::kInt32);

    // gpu tensors
    log_stats_gpu = torch::zeros({layer_num, log_exp_num}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    gpu_loads_gpu = torch::zeros({layer_num, ep_size}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
}

void BalanceStatsBuffers::reset() {
    log_stats.zero_();
    gpu_loads.zero_();
    log_stats_gpu.zero_();
    gpu_loads_gpu.zero_();
}

void LoadFlags::init() {
    flag_gpu  = torch::zeros({1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    flag_sync = torch::zeros({1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    flag_host = torch::zeros({1}, torch::kInt32).pin_memory();
}

void LoadFlags::setReady(bool ready) {
    int value = ready ? 0 : -1;
    flag_gpu.fill_(value);
}

bool LoadFlags::isReady() {
    if (!isCommOpsRegistered()) {
        // single-GPU: no cross-rank sync available, consume the local flag directly
        flag_host.copy_(flag_gpu);
        flag_gpu.fill_(-1);
        return flag_host.item<int>() == 0;
    }
    // sync all ranks load_flag_tensor_
    flag_sync = execAllReduce({flag_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP, flag_sync}).buffer;

    flag_host.copy_(flag_sync);
    // Repeatedly executing “x = ReduceSum(x)” with tp8 will cause an overflow and set the value of x to 0
    flag_gpu.fill_(-1);
    // if all load_flag_tensor_ is 0, return true
    return flag_host.item<int>() == 0;
}

void EplbController::init(const EPLBConfig& eplb_control_data, const EPLBConfig& eplb_config) {
    this->eplb_control_data = eplb_control_data;

    control_step = eplb_config.eplb_control_step;
    RTP_LLM_LOG_INFO("EPLB control step: %d", control_step);

    auto eplb_control_data_list  = eplb_control_data.toList();
    eplb_control_data_buf_host   = torch::zeros({(int64_t)eplb_control_data_list.size()}, torch::kInt32).pin_memory();
    eplb_control_data_buf_device = torch::zeros({(int64_t)eplb_control_data_list.size()},
                                                torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
}

void EplbController::setData(const EPLBConfig& updated_control_data) {
    // lock mutex
    lock_guard<mutex> lock(eplb_control_mutex);
    eplb_control_data = updated_control_data;
}

bool EplbController::stepAndCheckSyncStep() {
    // check if current step is sync step
    cur_step++;
    if (cur_step >= control_step) {
        cur_step = 0;
        return true;
    }
    return false;
}

EPLBConfig EplbController::getAndSyncData() {
    // copy control data to host buffer
    EPLBConfig cur_data;
    {
        lock_guard<mutex> lock(eplb_control_mutex);
        cur_data = eplb_control_data;
    }
    auto eplb_control_data_list     = eplb_control_data.toList();
    int* eplb_control_data_host_ptr = eplb_control_data_buf_host.data_ptr<int>();
    for (size_t i = 0; i < eplb_control_data_list.size(); ++i) {
        eplb_control_data_host_ptr[i] = eplb_control_data_list[i];
    }

    // copy to device
    eplb_control_data_buf_device.copy_(eplb_control_data_buf_host, /*non_blocking=*/true);

    // broadcast to all ranks (skip when comm ops unavailable, e.g. single GPU)
    if (isCommOpsRegistered()) {
        execBroadcast({{eplb_control_data_buf_device}, 0, ParallelMode::DP_AND_TP});
    }

    // copy to host
    eplb_control_data_buf_host.copy_(eplb_control_data_buf_device);

    // convert to EPLBConfig
    auto eplb_control_data = EPLBConfig::fromList(eplb_control_data_host_ptr);

    return eplb_control_data;
}

ExpertBalancer::ExpertBalancer(size_t                       log_exp_num,
                               size_t                       phy_exp_num,
                               size_t                       num_layers,
                               size_t                       moe_size,
                               size_t                       hidden_size,
                               size_t                       ep_rank,
                               size_t                       ep_size,
                               size_t                       stats_replication_size,
                               bool                         is_report_rank,
                               py::object                   py_eplb,
                               DataType                     dtype,
                               QuantAlgo                    quant_algo,
                               kmonitor::MetricsReporterPtr metrics_reporter,
                               const EPLBConfig&            eplb_config):
    num_logic_experts_(log_exp_num),
    num_physic_experts_(phy_exp_num),
    ep_rank_(ep_rank),
    ep_size_(ep_size),
    stats_replication_size_(stats_replication_size),
    is_report_rank_(is_report_rank),
    metrics_reporter_(metrics_reporter),
    eplb_python_wrapper_(py_eplb) {
    cout << "ExpertBalancer constructed with " << log_exp_num << " logical experts" << endl;
    printf("DEBUG: ExpertBalancer constructor called for linker debug\n");
    eplb_control_data_ = eplb_config;

    // init memory
    stats_.init(num_layers, log_exp_num, ep_size_);
    eplb_plan_buffers_.init(log_exp_num, phy_exp_num, hidden_size, moe_size, ep_size, dtype, quant_algo);
    eplb_plan_tensors_.init(log_exp_num, phy_exp_num);
    load_flags_.init();
    load_flags_.setReady(false);
    eplb_controller_.init(eplb_control_data_, eplb_config);

    test_mode_ = eplb_config.eplb_test_mode;

    balance_layer_per_step_ = eplb_config.eplb_balance_layer_per_step;

    stats_export_step_ = autil::EnvUtil::getEnv("EPLB_STATS_EXPORT_STEP", 100);
    stats_export_dir_  = autil::EnvUtil::getEnv("EPLB_STATS_EXPORT_DIR", std::string("eplb_stats"));
    if (stats_export_step_ > 0) {
        export_log_stats_ = torch::zeros({(int64_t)num_layers, (int64_t)log_exp_num}, torch::kInt64);
        export_gpu_loads_ = torch::zeros({(int64_t)num_layers, (int64_t)ep_size_}, torch::kInt64);
        printf("[EPLB] stats export enabled: every %ld steps -> %s (rank %zu)\n",
               (long)stats_export_step_,
               stats_export_dir_.c_str(),
               ep_rank_);
    }

    resetPlan(true);
}

ExpertBalancer::~ExpertBalancer() {}

void ExpertBalancer::stepForward(ModelBase& model, RtpLLMExecutorMetricsCollector& executor_collector) {
    syncController();

    OverallExpertStats& stats = model.overall_expert_stats_;

    // stats buffers are only allocated for MoE models (PyWrappedModel)
    if (!stats.stats_buf.log_stats_buf.defined()) {
        return;
    }

    if (eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::NONE)) {
        // drop this step's increments so stale counts never leak into a later mode switch
        stats.stats_buf.log_stats_buf.zero_();
        stats.stats_buf.gpu_loads_buf.zero_();
        return;
    }

    // report stats
    reportStats(stats);

    // periodic activation dump (STATS / ALL mode)
    exportStats(stats);

    // eplb plan
    excuteEplbPlan(stats, model);

    // L0 per-step zeroing: stats_buf holds single-step increments written by the
    // triton kernel; clear on the consumer side (a step may run multiple forwards)
    stats.stats_buf.log_stats_buf.zero_();
    stats.stats_buf.gpu_loads_buf.zero_();
}

bool ExpertBalancer::updateEplbConfig(const EPLBConfig& config) {
    eplb_controller_.setData(config);
    return true;
}

void ExpertBalancer::syncController() {
    // sync control data
    if (eplb_controller_.stepAndCheckSyncStep()) {
        auto eplb_control_data = eplb_controller_.getAndSyncData();
        if (eplb_control_data.eplb_mode != eplb_control_data_.eplb_mode
            || eplb_control_data.eplb_update_time != eplb_control_data_.eplb_update_time) {
            eplb_control_data_ = eplb_control_data;
            RTP_LLM_LOG_INFO("EPLB config changed to %s", eplb_control_data_.to_string().c_str());
        }
    }
}

void ExpertBalancer::reportStats(OverallExpertStats& stats) {
    if (!eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::STATS, EplbMode::ALL)) {
        return;
    }

    // Every rank must participate in the collective. Only the designated report
    // rank publishes the resulting EP columns, avoiding duplicate metric writers.
    auto gpu_loads_aggregated = stats.stats_buf.gpu_loads_buf.clone();
    if (isCommOpsRegistered()) {
        execAllReduce({gpu_loads_aggregated, ReduceOp::Sum, false, ParallelMode::DP_AND_TP});
    }
    if (stats_replication_size_ > 1) {
        gpu_loads_aggregated.div_(static_cast<int64_t>(stats_replication_size_), "trunc");
    }
    if (!is_report_rank_ || !metrics_reporter_) {
        return;
    }

    const int layer_num = stats.layer_num;
    executor_collector_.gpu_loads.resize(layer_num);
    auto gpu_loads_tensor = gpu_loads_aggregated.cpu();
    int* gpu_loads        = gpu_loads_tensor.data_ptr<int>();

    for (size_t report_ep_rank = 0; report_ep_rank < ep_size_; ++report_ep_rank) {
        executor_collector_.ep_rank = report_ep_rank;
        for (int i = 0; i < layer_num; ++i) {
            executor_collector_.gpu_loads[i] = gpu_loads[i * ep_size_ + report_ep_rank];
        }
        metrics_reporter_->report<RtpLLmEplbMetrics, RtpLLmEplbMetricsCollector>(nullptr, &executor_collector_);
    }
}

void ExpertBalancer::exportStats(OverallExpertStats& stats) {
    if (stats_export_step_ <= 0
        || !eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::STATS, EplbMode::ALL)) {
        return;
    }

    // accumulate this step's increments (stats_buf is zeroed per step by stepForward)
    export_log_stats_.add_(stats.stats_buf.log_stats_buf.cpu());
    export_gpu_loads_.add_(stats.stats_buf.gpu_loads_buf.cpu());
    total_step_cnt_++;
    stats_export_cnt_++;

    if (stats_export_cnt_ < stats_export_step_) {
        return;
    }
    stats_export_cnt_ = 0;

    int64_t        timestamp_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    const int64_t  layer_num    = export_log_stats_.size(0);
    const int64_t  log_exp_num  = export_log_stats_.size(1);
    const int64_t  ep_size      = export_gpu_loads_.size(1);
    const int64_t* log_stats    = export_log_stats_.data_ptr<int64_t>();
    const int64_t* gpu_loads    = export_gpu_loads_.data_ptr<int64_t>();

    std::error_code ec;
    std::filesystem::create_directories(stats_export_dir_, ec);
    if (ec) {
        RTP_LLM_LOG_WARNING(
            "EPLB stats export: failed to create dir %s: %s", stats_export_dir_.c_str(), ec.message().c_str());
        export_log_stats_.zero_();
        export_gpu_loads_.zero_();
        return;
    }

    std::string file_path = stats_export_dir_ + "/eplb_stats_rank" + std::to_string(ep_rank_) + "_step"
                            + std::to_string(total_step_cnt_) + "_" + std::to_string(timestamp_ms) + ".json";
    std::ofstream out(file_path);
    if (!out.is_open()) {
        RTP_LLM_LOG_WARNING("EPLB stats export: failed to open %s", file_path.c_str());
        export_log_stats_.zero_();
        export_gpu_loads_.zero_();
        return;
    }

    auto write_matrix = [&out](const int64_t* data, int64_t rows, int64_t cols) {
        out << "[";
        for (int64_t i = 0; i < rows; ++i) {
            out << (i ? ",[" : "[");
            for (int64_t j = 0; j < cols; ++j) {
                out << (j ? "," : "") << data[i * cols + j];
            }
            out << "]";
        }
        out << "]";
    };

    out << "{";
    out << "\"timestamp_ms\":" << timestamp_ms << ",";
    out << "\"total_step\":" << total_step_cnt_ << ",";
    out << "\"interval_steps\":" << stats_export_step_ << ",";
    out << "\"ep_rank\":" << ep_rank_ << ",";
    out << "\"ep_size\":" << ep_size << ",";
    out << "\"layer_num\":" << layer_num << ",";
    out << "\"log_exp_num\":" << log_exp_num << ",";
    out << "\"log_stats\":";
    write_matrix(log_stats, layer_num, log_exp_num);
    out << ",\"gpu_loads\":";
    write_matrix(gpu_loads, layer_num, ep_size);
    out << "}\n";
    out.close();

    // brief per-dump summary on stdout (C++ file logs do not reach stdout)
    int64_t total_activations = export_log_stats_.sum().item<int64_t>();
    printf("[EPLB] rank %zu step %ld: dumped expert stats (%ld activations in last %ld steps) -> %s\n",
           ep_rank_,
           (long)total_step_cnt_,
           (long)total_activations,
           (long)stats_export_step_,
           file_path.c_str());

    // top-K hottest logical experts, aggregated over all layers
    constexpr int64_t kTopK      = 8;
    const int64_t     k          = std::min(kTopK, log_exp_num);
    auto              per_expert = export_log_stats_.sum(0);  // [log_exp_num], INT64
    auto [topk_vals, topk_idxs]  = per_expert.topk(k);
    const int64_t* vals          = topk_vals.data_ptr<int64_t>();
    const int64_t* idxs          = topk_idxs.data_ptr<int64_t>();
    std::string    topk_str;
    for (int64_t i = 0; i < k; ++i) {
        if (i) {
            topk_str += ", ";
        }
        topk_str += std::to_string(idxs[i]) + ":" + std::to_string(vals[i]);
    }
    printf("[EPLB] rank %zu step %ld: top%ld experts (id:count, summed over %ld layers): %s\n",
           ep_rank_,
           (long)total_step_cnt_,
           (long)k,
           (long)layer_num,
           topk_str.c_str());
    fflush(stdout);

    export_log_stats_.zero_();
    export_gpu_loads_.zero_();
}

void ExpertBalancer::setPlanStatus(EplbPlanStatus status) {
    lock_guard<mutex> lock(eplb_plan_status_mutex_);
    eplb_plan_status_ = status;
}

EplbPlanStatus ExpertBalancer::getPlanStatus() const {
    lock_guard<mutex> lock(eplb_plan_status_mutex_);
    auto              status = eplb_plan_status_;
    return status;
}

void ExpertBalancer::excuteEplbPlan(OverallExpertStats& stats, ModelBase& model) {
    if (eplb_control_data_.checkEplbMode(eplb_control_data_.eplb_mode, EplbMode::EPLB, EplbMode::ALL)) {
        // accumulate in every state so increments produced while a plan is being
        // prepared/loaded are not lost
        updateStats(stats);

        EplbPlanStatus status = getPlanStatus();
        switch (status) {
            case EplbPlanStatus::INIT:
                update_cnt_++;
                if (update_cnt_ >= eplb_control_data_.eplb_update_time) {
                    setPlanStatus(EplbPlanStatus::PREPARING);
                }
                break;
            case EplbPlanStatus::PREPARING: {
                createPlan();
                setPlanStatus(EplbPlanStatus::LOADING);
                thread load_thread([this]() {
                    loadPlanWeights();
                    setPlanStatus(EplbPlanStatus::LOADED);
                });
                if (test_mode_) {
                    load_thread.join();
                } else {
                    load_thread.detach();
                }
                break;
            }
            case EplbPlanStatus::LOADING:
                syncPlanWeightsLoadStatus();
                break;
            case EplbPlanStatus::LOADED:
                load_flags_.setReady(true);
                if (syncPlanWeightsLoadStatus()) {
                    processPlanWeights();
                    applyPlanWeights(model);
                    load_flags_.setReady(false);
                    balance_layer_cnt_++;

                    if (balance_layer_cnt_ >= balance_layer_per_step_) {
                        update_cnt_        = 0;
                        balance_layer_cnt_ = 0;
                    } else {
                        update_cnt_ = eplb_control_data_.eplb_update_time;  // quick update
                    }

                    setPlanStatus(EplbPlanStatus::INIT);

                    resetPlan();
                }
                break;
        }
    }
}

void ExpertBalancer::resetPlan(bool force_clean) {
    stats_.reset();
}

void ExpertBalancer::copyFromTensor(const torch::Tensor& src, torch::Tensor& dst) {
    dst.copy_(src, /*non_blocking=*/true);
}

void ExpertBalancer::copyToTensor(const torch::Tensor& src, torch::Tensor& dst) {
    dst.copy_(src, /*non_blocking=*/true);
}

void ExpertBalancer::createPlan() {
    // pre run: aggregate per-rank stats (skip when comm ops unavailable, e.g. single GPU)
    if (isCommOpsRegistered()) {
        execAllReduce({stats_.log_stats_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP});
        execAllReduce({stats_.gpu_loads_gpu, ReduceOp::Sum, false, ParallelMode::DP_AND_TP});
    }
    if (stats_replication_size_ > 1) {
        stats_.log_stats_gpu.div_(static_cast<int64_t>(stats_replication_size_), "trunc");
        stats_.gpu_loads_gpu.div_(static_cast<int64_t>(stats_replication_size_), "trunc");
    }

    // copy stats gpu tensor to host tensor [implicit sync]
    stats_.log_stats.copy_(stats_.log_stats_gpu);
    stats_.gpu_loads.copy_(stats_.gpu_loads_gpu);

    // L1 per-plan-period zeroing: restart accumulation so the next plan sees
    // interval increments instead of the full history
    stats_.log_stats_gpu.zero_();
    stats_.gpu_loads_gpu.zero_();

    if (ep_rank_ == 0) {
        eplb_python_wrapper_.createBalancePlan(stats_.log_stats, stats_.gpu_loads, eplb_plan_tensors_);

        // copy tensor(host) to gpu tensor
        // note: it's ok to use async copy, since the tensor host ptr will not be released
        copyFromTensor(eplb_plan_tensors_.layer_id_buf, eplb_plan_buffers_.layer_id_buf);
        copyFromTensor(eplb_plan_tensors_.logic_expert_cnt, eplb_plan_buffers_.logic_expert_cnt);
        copyFromTensor(eplb_plan_tensors_.log2phy, eplb_plan_buffers_.log2phy);
        copyFromTensor(eplb_plan_tensors_.phy2log, eplb_plan_buffers_.phy2log);
    }

    if (isCommOpsRegistered()) {
        execBroadcast({{eplb_plan_buffers_.layer_id_buf,
                        eplb_plan_buffers_.logic_expert_cnt,
                        eplb_plan_buffers_.log2phy,
                        eplb_plan_buffers_.phy2log},
                       0,
                       ParallelMode::DP_AND_TP});
    }

    // copy plan gpu tensor to host tensor [implicit sync]
    copyToTensor(eplb_plan_buffers_.layer_id_buf, eplb_plan_tensors_.layer_id_buf);
    copyToTensor(eplb_plan_buffers_.logic_expert_cnt, eplb_plan_tensors_.logic_expert_cnt);
    copyToTensor(eplb_plan_buffers_.log2phy, eplb_plan_tensors_.log2phy);
    copyToTensor(eplb_plan_buffers_.phy2log, eplb_plan_tensors_.phy2log);
}

void ExpertBalancer::processPlanWeights() {
    eplb_plan_buffers_.layer_id = eplb_plan_tensors_.layer_id;
    if (eplb_plan_buffers_.is_quantized) {
        copyFromTensor(eplb_plan_tensors_.moe_weight_1, eplb_plan_buffers_.moe_weight_1);
        copyFromTensor(eplb_plan_tensors_.moe_scale_1, eplb_plan_buffers_.moe_scale_1);
        copyFromTensor(eplb_plan_tensors_.moe_weight_2, eplb_plan_buffers_.moe_weight_2);
        copyFromTensor(eplb_plan_tensors_.moe_scale_2, eplb_plan_buffers_.moe_scale_2);
    } else {
        copyFromTensor(eplb_plan_tensors_.moe_weight_1, eplb_plan_buffers_.moe_weight_1);
        copyFromTensor(eplb_plan_tensors_.moe_weight_2, eplb_plan_buffers_.moe_weight_2);
    }

    executor_collector_.update_weights_qps = true;
    executor_collector_.update_layer_id    = eplb_plan_buffers_.layer_id;
}

void ExpertBalancer::loadPlanWeights() {
    int64_t start_time_ms = autil::TimeUtility::currentTimeInMilliSeconds();

    eplb_python_wrapper_.loadBalanceWeight(ep_rank_, ep_size_, eplb_plan_tensors_);

    executor_collector_.update_weights_latency_ms = autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms;
}

void ExpertBalancer::applyPlanWeights(ModelBase& model) {
    auto& result         = eplb_plan_buffers_;
    auto& balanced_layer = model.weights_.layers[result.layer_id].ffn_weights;

    // Exchange DenseWeights kernel/scales/zeros with eplb plan tensors.
    auto exchange_tensor_data = [](torch::Tensor& model_tensor, torch::Tensor& plan_tensor) {
        auto old_data = model_tensor.clone();
        model_tensor.copy_(plan_tensor);
        plan_tensor.copy_(old_data);
    };

    auto exchange_dense_weight =
        [&result, &exchange_tensor_data](DenseWeights& dense, torch::Tensor& plan_weight, torch::Tensor& plan_scale) {
            exchange_tensor_data(dense.kernel, plan_weight);
            if (result.is_quantized) {
                exchange_tensor_data(dense.scales, plan_scale);
                dense.zeros = torch::Tensor();
            } else {
                dense.scales = torch::Tensor();
                dense.zeros  = torch::Tensor();
            }
        };

    exchange_dense_weight(*balanced_layer.moe_gate_weight, result.moe_weight_1, result.moe_scale_1);
    exchange_dense_weight(*balanced_layer.moe_down_weight, result.moe_weight_2, result.moe_scale_2);
    exchange_tensor_data(balanced_layer.log2phy, result.log2phy);
    exchange_tensor_data(balanced_layer.logic_expert_cnt, result.logic_expert_cnt);
}

bool ExpertBalancer::syncPlanWeightsLoadStatus() {
    return load_flags_.isReady();
}

void ExpertBalancer::updateStats(OverallExpertStats& stats) {
    stats_.log_stats_gpu.add_(stats.stats_buf.log_stats_buf);
    stats_.gpu_loads_gpu.add_(stats.stats_buf.gpu_loads_buf);
}

}  // namespace rtp_llm
