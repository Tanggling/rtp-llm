# MoE Log2Phy 分发与 Expert 统计设计

## 1. 背景与目标

MoE gate 选择的是逻辑 expert。开启 EPLB 冗余 expert 后，一个热点逻辑
expert 可能对应多个物理副本，并被放置到不同 EP rank 上。因此，top-k
产生的逻辑 ID 不能再直接交给 MoE dispatcher。

本次改造完成以下链路：

1. gate 在逻辑 expert 空间执行 top-k；
2. 根据 `log2phy` 将逻辑 ID 转换为物理 ID；
3. FusedMoE 和 DeepEP 按物理 expert 拓扑进行分发；
4. EPLB 更新 mapping 后，已构造的 Python layer 能立即使用新 mapping；
5. logical expert 热度和 physical EP load 分别按正确的 ID 空间统计；
6. `ExpertBalancer::reportStats` 聚合 DP 流量并去除 TP 重复数据。

验证代码不会调用现有 EPLB planner，也不会重新加载权重，而是直接 mock
一组 mapping 数据并原地更新。

## 2. 核心概念和数据布局

| 名称 | 含义 |
| --- | --- |
| logical expert | gate 选择的专家，ID 范围为 `[0, log_exp_num)` |
| physical expert | 实际加载的专家实例，ID 范围为 `[0, phy_exp_num)` |
| replica | logical expert 的一个物理副本 |
| `logic_expert_cnt` | 每个 logical expert 的物理副本数，shape 为 `[log_exp_num]` |
| `log2phy` | 每个 logical expert 对应的物理 ID，shape 为 `[log_exp_num, max_replica_num]` |
| `topk_ids` | FusedMoE 最终消费的 expert ID，转换后必须是 physical ID |

`log2phy` 中未使用的位置填充为 `-1`，每行只有前
`logic_expert_cnt[logical_id]` 项有效。

例如，4 个逻辑 expert、6 个物理 expert 可以使用如下布局：

```text
logic_expert_cnt = [2, 1, 2, 1]

log2phy = [
  [0, 4],  # logical 0 的物理副本为 0、4
  [1, -1], # logical 1 的物理副本为 1
  [2, 5],  # logical 2 的物理副本为 2、5
  [3, -1], # logical 3 的物理副本为 3
]
```

mapping 必须满足：

```text
0 < logic_expert_cnt[logical_id] <= log2phy.shape[1]
log2phy[logical_id, :logic_expert_cnt[logical_id]] 位于 [0, phy_exp_num)
```

## 3. 端到端分发链路

改造后的 forward 数据流如下：

```text
router logits
    |
    v
logical top-k IDs 和 weights
    |
    +----> logical expert 热度统计
    |
    v
logical_to_physical_experts(log2phy, logic_expert_cnt)
    |
    +----> physical EP load 统计
    |
    v
physical top-k IDs
    |
    v
FusedMoe router -> EP dispatch -> 本地物理 expert -> combine
```

`GenericMoeLayer` 保留原始 logical top-k tensor，并新建 physical top-k
tensor。前者用于统计 logical expert 热度，后者才进入 `FusedMoe`。

同时，`MoEConfigAdapter` 区分两种 expert 数量：

```python
self.logical_expert_num = model_config.expert_num
self.expert_num = model_config.eplb_config.phy_exp_num(model_config.expert_num)
```

因此各模块的语义为：

- gate 和 top-k 使用 logical expert 数；
- FusedMoE router/executor 使用 physical expert 数；
- DeepEP buffer 大小和 dispatch layout 使用 physical expert 数。

## 4. Log2Phy 转换算法

实现将二维 `topk_ids` 视为一段连续 ID。对 flattened position `i`、
逻辑 expert `logical_id` 和当前 `dp_rank`：

```text
replica_count = logic_expert_cnt[logical_id]
replica_slot  = (i + dp_rank) % replica_count
physical_id   = log2phy[logical_id, replica_slot]
```

这里使用 `dp_rank` 而不是 `ep_rank`，原因是普通 TP group 内的 ranks
处理同一份请求，必须得到一致的 logical-to-physical 映射。否则 TP ranks
会先生成不同的完整路由，再各自切 token，后续无法严格去重统计数据。

flattened position 仍会轮询不同副本，所以 `dp_rank=0` 不代表始终选择
第一个物理副本。

### 4.1 完整转换实例

继续使用第 2 节的 mapping。假设 gate 得到：

```text
logical_topk_ids = [
  [0, 2],
  [0, 1],
  [2, 3],
]
```

当 `dp_rank=0` 时：

| flat position | logical ID | 副本数 | replica slot | physical ID |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 2 | 0 | 0 |
| 1 | 2 | 2 | 1 | 5 |
| 2 | 0 | 2 | 0 | 0 |
| 3 | 1 | 1 | 0 | 1 |
| 4 | 2 | 2 | 0 | 2 |
| 5 | 3 | 1 | 0 | 3 |

最终交给 dispatcher 的数据为：

```text
physical_topk_ids = [
  [0, 5],
  [0, 1],
  [2, 3],
]
```

当 `dp_rank=1` 时，副本 slot 整体偏移一位：

```text
physical_topk_ids = [
  [4, 2],
  [4, 1],
  [5, 3],
]
```

这样不同 DP 请求会落到不同副本，而同一 DP group 内所有 TP ranks 的
映射保持一致。

## 5. Mapping 热更新如何对 Python Layer 生效

`GenericMoeLayer` 在构造时保存 `log2phy` 和 `logic_expert_cnt` 的 tensor
引用。如果 C++ 仅通过 `std::swap` 替换 `torch::Tensor` handle，已经持有
旧 tensor 引用的 Python layer 不一定能观察到新 handle。

`ExpertBalancer::applyPlanWeights` 因此改为原地交换 tensor 内容：

```cpp
auto old_data = model_tensor.clone();
model_tensor.copy_(plan_tensor);
plan_tensor.copy_(old_data);
```

该方案保证：

1. model tensor 的对象身份不变，Python layer 立即看到新数据；
2. 旧数据被放回 plan buffer，供下一轮交换使用。

MoE kernel、量化 scale、`log2phy` 和 `logic_expert_cnt` 都使用相同方式。

### 5.1 Mock 更新实例

第一次 forward 使用：

```text
log2phy = [
  [0, 4],
  [1, -1],
  [2, 5],
  [3, -1],
]
```

`dp_rank=0` 时得到：

```text
[[0, 5], [0, 1], [2, 3]]
```

测试直接 mock 新 plan，并原地更新同一个 tensor：

```text
new_log2phy = [
  [4, 0],
  [1, -1],
  [5, 2],
  [3, -1],
]

log2phy.copy_(new_log2phy)
```

不重建 `GenericMoeLayer`，下一次 forward 得到：

```text
[[4, 2], [4, 1], [5, 3]]
```

该过程验证了“mapping 被更新”以及“新 mapping 被实际用于 dispatch”。
测试没有调用 `create_balance_plan` 或 `load_moe_weight`。

## 6. Expert 统计语义

统计必须区分两个 ID 空间。

### 6.1 Logical Expert 热度

`log_stats[layer, logical_id]` 表示 logical expert 被 top-k 选中的次数，
必须使用转换前的 logical ID。EPLB 算法通过它判断哪些逻辑 expert 需要
增加副本。

对第 4 节的输入：

```text
log_stats = [2, 1, 2, 1]
```

### 6.2 Physical EP Load

`gpu_loads[layer, ep_rank]` 表示发往某个 EP rank 所拥有物理 expert 的
token/top-k pair 数量，必须使用转换后的 physical ID。

假设 `phy_exp_num=6`、`ep_size=2`，物理 expert 连续划分：

```text
EP rank 0: physical IDs [0, 1, 2]
EP rank 1: physical IDs [3, 4, 5]
```

对 physical IDs `[[0, 5], [0, 1], [2, 3]]`：

```text
gpu_loads = [4, 2]
```

旧实现根据 logical ID 静态计算 EP rank。在本例中，它无法反映 logical
expert 0 和 2 的副本已经被放到 EP rank 1，因此不符合实际 dispatch。

## 7. TP、EP、DP 聚合和去重

### 7.1 TP 去重

普通 TP 模式下，每个 TP rank 会看到相同 hidden states，并生成相同的
logical top-k，因此每个 rank 都会记录一份完全相同的数据。

假设 `TP=2`、`DP=2`，某一层两个 DP 请求分别产生：

```text
DP rank 0: A = [2, 1, 2, 1]
DP rank 1: B = [1, 3, 0, 2]
```

每份数据都有两个 TP 副本。world sum 为：

```text
2*A + 2*B = [6, 8, 4, 6]
```

除以统计复制因子 `TP=2` 后得到真实全局数据：

```text
A + B = [3, 4, 2, 3]
```

physical `gpu_loads` 使用同样规则。由于 TP group 内 mapping 使用相同
`dp_rank`，所以物理路由也能被严格去重。

### 7.2 DP 聚合

DP ranks 处理不同请求，其数据代表独立流量，必须相加而不能相除。
`reportStats` 先执行 `DP_AND_TP` all-reduce，再只除 TP 复制因子。

### 7.3 EP 聚合

每个 rank 记录一个包含所有 EP destination 的向量。world all-reduce 后，
只有 `world_rank=0` 负责逐列上报全部 EP ranks。

EP 各列表示不同物理 owner，不是同一统计量的副本，因此不做 EP 除法。

### 7.4 Context Parallelism

CP 场景下，token route 已在物理 TP ranks 之间切分。如果仍除以物理 TP
size，会造成少计。

传给 `ExpertBalancer` 的统计复制因子是 `get_attn_tp_size()`；CP 开启时
该值为 1，因此只聚合各分片，不执行 TP 去重。

### 7.5 `reportStats` 最终行为

```text
每个 rank 的 physical gpu_loads
    -> DP_AND_TP all-reduce sum
    -> 除以普通 TP 复制因子
    -> 非 report rank 返回
    -> world_rank=0 遍历全部 ep_rank 列
    -> 携带 ep_rank 和 layer tag 上报
```

所有 ranks 仍必须参加 collective；只限制 metrics publication 的 owner。
因此 `EP=1` 时不会由多个 TP ranks 重复写入同一 tag，`EP=WORLD_SIZE`
时也不会丢失任何 EP rank 的数据。

`createPlan` 对 `log_stats` 和 `gpu_loads` 使用相同的聚合及 TP 去重逻辑，
从而保证 planner 输入和监控指标语义一致。

## 8. Metrics Tag 修正

EPLB reporter 的调用方传入空 base tags。旧代码会解引用空指针，而且
`gpu_loads` 实际上只携带 layer tag，没有携带 `ep_rank`。

新实现为每个样本独立构造：

```text
ep_rank=<rank>, layer=<layer>
```

weight-update metrics 路径也对可选 base tags 增加了空指针保护。

## 9. 验证覆盖

验证代码覆盖：

- logical-to-physical 转换结果；
- mock plan 原地更新后，下一次 dispatch 立即使用新 mapping；
- `GenericMoeLayer.forward` 将 physical IDs 传给 capturing fake FusedMoe；
- logical heat 始终使用 logical ID；
- EP load 使用 physical expert owner；
- 原有 Triton interpreter 和 Torch fallback 统计场景。

已执行结果：

```text
17 项 mapping/statistics/end-to-end mock 测试：通过
21 项 FusedMoE config resolver 测试：通过
Python syntax compilation：通过
```

PPU 上的 Bazel 定向测试没有进入编译阶段。新增 model test target 的 Torch
依赖在 PPU 配置下引用了未定义的 `@pip_torch` repository，因此本次 Bazel
执行没有覆盖 C++ 编译验证。

## 10. 当前限制

1. `exportStats` 导出的是每 rank 原始 buffer，可用于调试，但不是全局聚合
   后的数据。全局事实应以 `reportStats` 或 planner 输入为准。
2. mapping helper 假设 logical ID 合法且副本数大于零；loader/planner 生成
   plan 时应保证这些 invariant。
3. 原地交换权重时会临时 clone tensor。EPLB 更新频率较低，当前优先保证
   引用正确性，尚未优化更新瞬间的额外显存。
4. 后续仍应在实际多 rank TP/DP/CP 拓扑中，将上报的 EP load 与 DeepEP
   receive count 进行对照验证。

## 11. 相关代码

| 文件 | 职责 |
| --- | --- |
| `rtp_llm/models_py/model_desc/generic_moe.py` | 保留 logical top-k、执行转换、分发 physical ID |
| `rtp_llm/models_py/triton_kernels/moe/ep_kernels.py` | mapping helper 和 logical/physical 统计 kernel |
| `rtp_llm/models_py/modules/factory/fused_moe/defs/config_adapter.py` | 区分 logical 和 physical expert 数 |
| `rtp_llm/models_py/distributed/deepep_wrapper.py` | 按 physical expert 数构造 DeepEP buffer |
| `rtp_llm/cpp/models/eplb/ExpertBalancer.cc` | 聚合统计并原地应用 plan |
| `rtp_llm/cpp/metrics/RtpLLMMetrics.cc` | 安全上报 EP/layer tagged metrics |
| `rtp_llm/models_py/model_desc/test/log2phy_dispatch_test.py` | mock 端到端更新和 dispatch 验证 |
| `rtp_llm/models_py/triton_kernels/moe/test/record_expert_stats_test.py` | mapping 和统计单测 |
