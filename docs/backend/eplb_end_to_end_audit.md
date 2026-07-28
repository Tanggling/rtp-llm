# MoE Top-K、Dispatch、统计上报与 EPLB Reload 全链路复盘

## 1. 文档范围与结论

本文复盘当前 `github-opensource` 中 MoE 从 gate 选出 top-K，到 logical-to-physical
映射、router dispatch、本地 expert 计算、统计、指标上报、EPLB 计划生成、异步加载和
新布局生效的完整链路。

审计基于当前工作区代码，不只覆盖本次 `log2phy` 改动。结论分为三类：

- **已确认正确**：能由代码契约或现有测试直接证明；
- **已确认问题**：存在明确触发条件和错误后果；
- **待运行时验证**：静态代码不足以证明真实通信量、CUDA Graph 或故障场景。

核心结论如下：

1. gate 始终在 logical expert 空间选 top-K；当前改动已在进入 FusedMoE 前将 logical ID
   转为 physical ID，所有 router 因而按物理布局 dispatch。
2. 当前只允许 `EP=1`，或 `EP=TP*DP=WORLD_SIZE`。后一种配置下，所有 physical experts
   由全局 EP ranks 共同分片，不是每个 DP group 各自包含一套完整 experts。
3. `log_stats` 统计 logical 热度，`gpu_loads` 按 physical ID 推算目标 EP rank。后者是
   **dispatch 前预测值**，不是通信层实际接收量。
4. world all-reduce 后除以普通 TP 复制数，可以去除同一 DP 请求在 TP ranks 上的重复
   gate 统计；DP 和 CP 数据应求和而不应去重。
5. 重复上报问题已改成所有 ranks 参与聚合、仅 `world_rank=0` 发布全部 EP columns。
6. reload 链路仍有高风险问题：detached thread 生命周期与异常处理、`EP=1` 时重复创建
   plan、失败无事务回滚，以及换权重时额外复制整层 expert tensors。

## 2. 并行拓扑和 expert 分布

### 2.1 当前支持的拓扑

配置校验位于 `rtp_llm/config/server_config_setup.py`：

```text
EP_SIZE = 1
或
EP_SIZE = TP_SIZE * DP_SIZE = WORLD_SIZE
```

因此不能把 TP 和 EP 一概视为相等：

- `EP=1, TP>1`：pure TP，所有 ranks 都持有同一组 logical/physical experts，expert 内部
  权重按 TP 计算方式处理；
- `EP=TP, DP=1`：EP rank 与 world rank 一一对应，此时数值上 `EP==TP`；
- `EP=TP*DP, DP>1`：`EP>TP`，TP 只是每个 DP group 内的计算/输入复制维度，EP 是全局
  expert 放置与通信维度。

### 2.2 `TP=4, DP=2, EP=8` 的具体分布

假设 `phy_exp_num=16`，每个 EP rank 放 2 个 physical experts：

| DP group | world ranks / EP ranks | 本组持有的 physical experts |
| --- | --- | --- |
| DP 0 | 0, 1, 2, 3 | 0-7 |
| DP 1 | 4, 5, 6, 7 | 8-15 |

这里每个 DP group **只持有全局 expert 集合的一部分**。完整的 0-15 只存在于全局 EP
group 中。`LoadConfig.get_selected_experts()` 按 `ep_rank` 对 `phy2log` 连续切片，DeepEP
也跨全局 EP group dispatch，所以一个 DP group 的 token 可以被发到另一个 DP group
所在 ranks 的 experts。

如果某个 logical expert 有多个副本，它们可以落在不同 EP ranks。logical expert
集合在语义上是全局完整的；physical expert instances 则由全局 EP group 分布式持有。

## 3. 端到端数据流

```text
hidden_states
  -> gate(hidden_states)
  -> router_logits(float32)
  -> SelectTopk / GroupTopK
  -> logical_topk_ids + topk_weights
  -> optional FakeBalanceExpert
  -> logical_to_physical_experts(log2phy, logic_expert_cnt, dp_rank)
  -> physical_topk_ids
  +-> record_expert_stats(logical IDs, physical IDs)
  -> FusedMoe.router.prepare()
  -> 跨 rank dispatch / 本地 ID 重写
  -> local physical expert execution
  -> router.finalize() / combine
  -> ExpertBalancer::stepForward()
      +-> reportStats()
      +-> exportStats()
      +-> updateStats() / createPlan()
  -> Python create_balance_plan()
  -> broadcast plan
  -> 每个 rank 异步 load_moe_weight()
  -> 全局 load flag 同步
  -> processPlanWeights()
  -> applyPlanWeights()
  -> 下一次 forward 使用新 weights + 新 log2phy
```

关键代码入口：

| 阶段 | 入口 |
| --- | --- |
| top-K 与映射 | `rtp_llm/models_py/model_desc/generic_moe.py:GenericMoeLayer.forward` |
| logical-to-physical | `rtp_llm/models_py/triton_kernels/moe/ep_kernels.py:logical_to_physical_experts` |
| 统计 kernel | `ep_kernels.py:record_expert_stats` |
| router 与 executor | `rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/routers/` |
| C++ 消费与上报 | `rtp_llm/cpp/models/eplb/ExpertBalancer.cc:stepForward` |
| Python planner/loader | `rtp_llm/eplb/ep_balancer.py` |
| expert 权重选择 | `rtp_llm/model_loader/load_config.py:get_selected_experts` |

## 4. Top-K 与 Log2Phy 契约

### 4.1 Top-K 输出属于 logical 空间

`router_logits` 最后一维等于 logical expert 数。`SelectTopk` 或 `GroupTopK` 输出：

```text
logical_topk_ids: [num_tokens, moe_k]
topk_weights:     [num_tokens, moe_k]
```

`FakeBalanceExpert` 若开启，也在 logical ID 上修改 top-K。统计 logical 热度必须保存这份
转换前的 tensor。

### 4.2 Logical-to-physical 算法

对 flatten 后位置 `i`、logical expert `l` 和 `dp_rank`：

```text
replica_count = logic_expert_cnt[l]
replica_slot  = (i + dp_rank) % replica_count
physical_id   = log2phy[l, replica_slot]
```

使用 `dp_rank` 作为 offset 的目的，是让同一 DP group 内复制同一输入的 TP ranks 得到
完全相同的 physical routes，而不同 DP 请求能偏移到不同副本。

例：

```text
logic_expert_cnt = [2, 1, 2, 1]
log2phy = [[0,4], [1,-1], [2,5], [3,-1]]
logical_topk_ids = [[0,2], [0,1], [2,3]]
```

则：

```text
dp_rank=0 -> [[0,5], [0,1], [2,3]]
dp_rank=1 -> [[4,2], [4,1], [5,3]]
```

转换后的 `physical_topk_ids` 才传入 `self.fused_moe(...)`。`MoEConfigAdapter.expert_num`
也已使用 `phy_exp_num`，所以 router 的 expert count、每 rank expert 范围和 DeepEP layout
均处于 physical 空间。

### 4.3 当前映射实现的边界风险

`logical_to_physical_experts()` 当前通过 `long()`、`index_select()`、`arange()` 和高级索引
组合实现。它有三个未闭合的契约：

1. `logical_ids` 中若有 `-1` 或越界值，会在 `index_select` 阶段失败，尚未像统计 kernel
   一样屏蔽 invalid ID；
2. `logic_expert_cnt` 为 0 会在 remainder 时失败，`log2phy` 中有效槽为 `-1` 会把无效
   physical ID 继续交给 router；
3. 每层每次 forward 会创建临时 tensors，CUDA Graph capture 和 hot-path 开销尚无真实
   GPU 测试或 benchmark。

建议恢复/实现 fused device kernel，在同一个 kernel 中完成合法性检查、replica slot 和
physical ID 写出，并增加 invalid ID、count=0、CUDA Graph capture/replay 测试。

## 5. Dispatch Router 全路径

所有路径接收的 `topk_ids` 都应是 physical ID。

| Router | prepare | 本地执行 ID | finalize | 统计复制语义 |
| --- | --- | --- | --- | --- |
| PureTP | 本地量化；必要时按本 rank expert 范围重写 | local physical ID 或全量 physical ID | TP all-reduce | 普通 TP 输入复制，需除 TP |
| DeepEP Normal | 每个 TP rank 切一段 token；全局 EP dispatch | 接收端 local experts | DeepEP combine，再 TP all-gather | 映射前每 TP rank 已统计全输入，需除 TP |
| DeepEP Low Latency | TP token slice；low-latency dispatch | per-expert packed input | low-latency combine，再 TP all-gather | 同上 |
| PureDP | world all-gather token/top-k，按本地 expert 过滤 | local physical ID | world reduce-scatter | 每个 DP rank 输入独立，不去重 |
| PureCP | TP all-gather CP token shards，按本地 expert 过滤 | local physical ID | TP reduce-scatter | CP token 已分片，不去重 |

### 5.1 PureTP

`PureTpRouterBase.prepare()` 用 `expert_start_id=ep_rank*expert_num_per_rank` 将 global
physical ID 重写成 local ID，不属于本 rank 的位置写成 `-1`。执行结果在 TP group
all-reduce。无量化单卡/全 experts 情况可以不重写。

### 5.2 DeepEP Normal

`DeepepNormalRouterBase.prepare()` 先按 TP rank 对 token 行切片，然后
`get_dispatch_layout(physical_topk_ids, phy_exp_num)`，再由 DeepEP dispatch 到全局 EP
ranks。本地执行使用接收端 expert 范围；combine 后在 TP group all-gather 恢复原 token
顺序。

`num_recv_tokens_per_expert_list` 是该路径可用于验证真实接收量的权威观测值。目前它只
进入 executor metadata，没有回流到 EPLB `gpu_loads`。

### 5.3 DeepEP Low Latency

流程与 Normal 相同，但由 `low_latency_dispatch/combine` 完成；真实本地接收量可从
`expert_num_tokens` 获得。当前 EPLB 统计同样没有消费该值。

### 5.4 PureDP 与 PureCP

PureDP 适用于 `TP=1, DP=EP>1`，先对不等长 DP batch 做 padding，再 world all-gather，
最后 world reduce-scatter。padding 的 top-k ID 为 `-1`，但发生在 log2phy 转换和统计
之后，所以不会污染当前统计。

PureCP 适用于 `DP=1, physical TP=EP>1` 且 prefill CP 开启。各 rank 输入是不同 context
分片，TP all-gather 后执行，最后 reduce-scatter。由于映射和统计发生在 all-gather 前，
每个真实 token 只统计一次。

## 6. 统计值的定义和聚合

### 6.1 L0：单 step device buffer

每个 rank 持有：

```text
log_stats_buf[layer, logical_expert] : INT32
gpu_loads_buf[layer, ep_rank]        : INT32
```

`record_expert_stats()` 对每个有效 token/top-k pair 各加 1：

- `log_stats` 使用 logical ID；
- `gpu_loads` 使用 physical ID，并按连续 physical expert 范围计算 owner EP rank。

单 rank、单 layer 应满足计数守恒：

```text
sum(log_stats_buf[layer]) == sum(gpu_loads_buf[layer])
                            == valid_token_count * moe_k
```

`stepForward()` 消费后清零 L0 buffer。它可能同时被上报、导出和加入 planner 周期累计，
三者读取的是同一批单 step increments。

### 6.2 `gpu_loads` 是预测值，不是实际值

当前计算发生在 `FusedMoe.router.prepare()` 之前：

```text
predicted_ep_rank = physical_id // experts_per_rank
```

因此它能反映 log2phy 放置，却不能证明 dispatch 已成功完成，也不能发现通信丢弃、
padding/masking 差异或 router bug。更准确的命名应是 `expected_dispatch_loads`。

准确性验证应同时采集并比较：

- DeepEP Normal：`num_recv_tokens_per_expert_list`；
- DeepEP Low Latency：`expert_num_tokens`；
- PureTP/PureDP/PureCP：`recompute_topk_ids_sum_expert_count()` 返回的本地 counts。

对每一层、每一 rank，应比较 predicted owner count 与 observed receive count；全局再检查
发送、接收和 logical top-K 三方守恒。

### 6.3 TP/DP/CP 聚合公式

设 `S[r]` 是 rank `r` 的本地统计，`R` 是普通 TP 输入复制数：

```text
GlobalStats = sum_over_world(S[r]) / R
R = get_attn_tp_size()
```

具体情况：

| 拓扑 | 本地输入关系 | `R` | 聚合 |
| --- | --- | ---: | --- |
| `EP=1, TP=4, DP=1` | 4 ranks 看到同一 token | 4 | world sum / 4 |
| `EP=8, TP=4, DP=2` | 每个 DP 请求复制 4 份，两个 DP 请求独立 | 4 | world sum / 4 |
| `EP=8, TP=1, DP=8` | 8 个 DP 请求独立 | 1 | world sum |
| CP：`EP=TP=8, DP=1` | context token 在 8 ranks 分片 | 1 | world sum |

例：`TP=2, DP=2`，DP0 的真实热度为 `A`，DP1 为 `B`：

```text
rank 数据 = A, A, B, B
world sum = 2A + 2B
除以 TP=2 => A + B
```

当前 NormalExecutor 和 MtpExecutor 都把 `get_attn_tp_size()` 传给
`stats_replication_size`。CP 下该值为 1，符合分片而非复制的语义。

### 6.4 L1：EPLB 周期累计

`updateStats()` 每 step 把 L0 加到 `stats_.log_stats_gpu/gpu_loads_gpu`。到更新周期后：

1. world all-reduce；
2. 除以普通 TP 复制数；
3. copy 到 CPU；
4. 清零 GPU 周期累计；
5. 交给 Python planner。

这些 buffer 是 INT32。长更新周期、高流量或很大的 `moe_k` 可能在创建 plan 前溢出，
应改为 INT64，或明确计算安全上限并在接近上限时提前 flush。

### 6.5 本地 JSON 导出

`exportStats()` 是每 rank 的原始 L0 周期累计，未做 world 聚合和 TP 去重；它不能直接
等同于上报值或 planner 输入。当前目录中的 step 100 样本为 `EP=1`，48 个 layers 均
满足：

```text
sum(log_stats[layer]) == sum(gpu_loads[layer]) == 960
```

这能证明 EP1 下计数守恒，但不能验证多 EP 的 physical 放置。

文件名使用 `ep_rank`。在 `EP=1, TP>1` 时所有进程都是 `rank0`，会造成来源歧义，时间戳
接近时也存在覆盖风险。应加入 `world_rank/tp_rank/dp_rank`，并在 JSON 中声明
`scope=local_raw`、topology 和 replication factor。

## 7. 指标上报与重复上报修复

### 7.1 旧行为

旧 `reportStats()` 在每个进程中只取本地 `gpu_loads[:, local_ep_rank]` 并调用 reporter。
因此：

- 每个 rank 都上报，不是只有一个 rank；
- 普通 TP 下同一份 token 被重复统计；
- DP 流量没有形成全局视图；
- 多进程可能写同一指标 tag，产生重复 writer 或覆盖。

### 7.2 当前修复

当前实现：

1. 所有 ranks 对完整 `[layer, ep_size]` 做 `DP_AND_TP` all-reduce；
2. 除以 `stats_replication_size` 去除普通 TP 重复；
3. 所有 ranks 都参与 collective，但只有 `world_rank=0` 继续发布；
4. root 遍历所有 `report_ep_rank`，发布完整 EP columns；
5. metric tag 每次从新的 `{ep_rank, layer}` 构造，避免 tag 在循环间累积。

这修复了重复 writer bug。需要和监控平台确认的一点是：所有 EP 指标现在都由 world0
实例身份发出；如果 dashboard 依赖 exporter 所在 host，而不只依赖显式 `ep_rank` tag，
需要增加 `source_world_rank` 或改用独立的全局指标 namespace。

### 7.3 上报准确性如何验证

建议提供 debug-only 的聚合快照，而不是依赖在线 dashboard：

1. 构造确定性 logical top-K 和 mock `log2phy`；
2. 每 rank 导出 local logical/physical IDs、local L0 stats；
3. root 导出 all-reduce 前后和除 TP 后的矩阵；
4. 同时导出 router observed receive counts；
5. 离线检查四个不变量：

```text
local logical sum == local predicted physical sum
global logical sum == global predicted physical sum
global predicted physical column == global observed receive column
reported metric value == deduplicated global predicted value
```

当前 Python UT 验证了 logical/physical 统计和 mock mapping 热更新，但尚没有多进程测试
直接断言 root reporter 的调用次数、tags 和数值。应增加 C++/distributed test：

```text
TP=2, DP=2, EP=4
预置 A/A/B/B 四份 rank stats
期望 root 仅 report EP_SIZE 次
期望矩阵为 (A+B) 对应的 EP columns
非 root reporter 调用次数为 0
```

## 8. EPLB Plan 与 Reload 全流程

### 8.1 状态机

```text
INIT
  -- update_cnt >= eplb_update_time --> PREPARING
PREPARING
  -- aggregate stats / create plan / broadcast --> LOADING
  -- detached thread: load_moe_weight() --> LOADED
LOADING
  -- 每 step 全局同步 load flags，等待所有 ranks
LOADED
  -- 本 rank flag=ready；所有 ranks ready
  -> processPlanWeights()
  -> applyPlanWeights()
  -> reset counters/stats
  -> INIT
```

### 8.2 计划生成

`createPlan()` 将去重后的 logical heat 和 predicted GPU load 传给 Python：

1. `HistoryStats` 维护窗口累计；
2. round/random/most-unbalanced/mix 选择一个 layer；
3. `rebalance_experts()` 产生 `phy2log/log2phy/logcnt`；
4. root 把 plan tensors 拷到 GPU；
5. 通过 `DP_AND_TP` broadcast 给所有 ranks；
6. 各 rank 拷回 CPU，供 loader 使用。

planner 用 `log_stats` 决定副本分配，用 `gpu_loads` 选择最不均衡 layer。由于后者是预测
值，planner 实际优化的是“按当前 mapping 预计的负载”，不是通信层实测负载。

### 8.3 每 rank 加载新 experts

`load_moe_weight()` 先把 layer 的 `phy2log` 更新到本 rank `LoadConfig`，再由
`get_selected_experts()` 按 `ep_rank` 切出本地 physical slots 对应的 logical expert IDs，
从数据库加载该组权重。所有 ranks 必须使用同一个 plan，否则权重和 dispatch mapping
会不一致。

### 8.4 新布局生效

所有 ranks load ready 后，主线程把 CPU 新权重 copy 到预分配 GPU plan buffers，再由
`applyPlanWeights()` 更新 model：

```text
moe_gate_weight.kernel/scales
moe_down_weight.kernel/scales
log2phy
logic_expert_cnt
```

`GenericMoeLayer` 在构造时保存 mapping tensor 引用，所以 mapping 必须原地 `copy_`，不能
只替换 C++ tensor handle。当前实现满足这一点；mock 测试也证明不重建 layer 时下一次
forward 能看到新 mapping 并把新 physical IDs 交给 FusedMoE。

## 9. 问题清单

### P0/P1：需要优先修复

#### P1-1 detached reload thread 可能导致进程终止或 use-after-free

- **证据**：`ExpertBalancer.cc:429-436` 创建捕获 `this` 的 thread 并 `detach()`；析构函数
  `ExpertBalancer.cc:212` 为空。`ep_balancer.py:246-265` 捕获加载异常后没有返回失败，仍会
  访问未赋值的 `res`，继而抛出 `UnboundLocalError`。
- **触发**：loader 抛 Python/C++ 异常，或 engine 在加载完成前析构。
- **后果**：未捕获异常会触发 `std::terminate`；悬空 `this` 会导致 use-after-free；状态也
  可能永久停在 LOADING。
- **建议**：持有 joinable worker/future，析构时停止并 join；线程入口捕获所有异常，把
  exception/status 传回主线程；增加 FAILED/CANCELLED 状态和超时。

#### P1-2 reload 没有跨 rank 事务和回滚

- **证据**：plan、各 rank load、flag all-reduce、逐 tensor apply 分阶段完成，没有 plan
  version、abort broadcast、失败回滚或超时。
- **触发**：任一 rank 数据库读取失败、shape/dtype 不符、进程变慢或退出。
- **后果**：其他 ranks 可无限等待 collective，或部分 rank 已准备/应用而全局状态分叉。
- **建议**：每轮分配 `plan_id`；先全 rank validate 到 staging buffers，再 all-reduce success；
  只有 unanimous commit 才 apply；失败则 broadcast abort 并保留旧 model tensors。

#### P1-3 `EP=1, TP>1` 时所有 ranks 都创建 plan

- **证据**：`createPlan()` 用 `if (ep_rank_ == 0)`；EP1 下每个进程的 `ep_rank` 都为 0。
- **触发**：pure TP 开 EPLB/ALL。
- **后果**：每个 TP rank 都调用 Python planner、更新各自 history、写 planner JSON，再由
  world root broadcast 覆盖；存在重复 side effect 和无谓 CPU/IO。
- **建议**：用 `is_report_rank_` 或明确 `world_rank==0` 作为唯一 plan creator，并断言它与
  broadcast root 一致。

#### P1-4 CUDA Graph padding 可能污染统计

- **证据**：CUDA Graph capture buffers 按较大 batch 分配，prepare 只复制真实 input rows，
  `GenericMoeLayer` 对 `hidden_states.shape[0]` 全部做 gate/top-K 和统计；统计 API 没有
  valid-token mask/count。
- **触发**：decode replay 的实际 batch 小于 capture batch，尾部 rows 为 graph padding。
- **后果**：padding rows 也产生有效 top-K，logical heat、predicted loads 和上报值偏大。
- **状态**：静态代码显示风险，但需要 CUDA Graph MoE 用例确认尾部 hidden state 的实际
  屏蔽方式。
- **建议**：把 device-side valid token count/mask 传到 mapping 和统计 kernel；增加
  capture batch 8、replay batch 3 的计数断言。

### P1/P2：性能、可观测性与健壮性

#### P1-5 每次 forward 上报引入全局同步

`reportStats()` 在 STATS/ALL 模式每 step 做 world all-reduce，root 随后 `.cpu()`。这会在
decode hot path 引入全局同步和 D2H 等待。建议在 device 上累计，仅按配置周期聚合上报；
指标表达为 interval sum/rate，并记录 interval steps/tokens。

#### P1-6 换权重时 clone 整个本地 expert tensor

`applyPlanWeights()` 对每个 kernel/scale 做 `model_tensor.clone()`，然后新旧数据双向 copy。
旧数据下一轮会被新 staging 权重覆盖，没有观察到后续消费者。这会在 reload 瞬间显著
增加 MoE layer 显存并可能 OOM。应确认无回滚依赖后改为单向 `model_tensor.copy_(plan)`；
若要事务回滚，应使用明确的双 staging buffer，而不是临时 clone。

#### P1-7 Torch fallback 与 CUDA Graph 声明不一致

`record_expert_stats()` 捕获任意 Triton 异常后永久切到 Torch fallback。fallback 使用动态
boolean indexing、`ones_like` 和临时分配，不能据此声称 CUDA Graph safe；捕获所有异常
还会隐藏 kernel correctness bug。建议只对已识别的 backend-unavailable 错误 fallback，
CUDA Graph 开启时 fail fast 或提供预分配、无动态 shape 的 device kernel。

#### P2-1 Controller 快照存在数据竞争

`EplbController::getAndSyncData()` 在锁内复制了 `cur_data`，随后却调用成员
`eplb_control_data.toList()`；`cur_data` 未使用。并发 `setData()` 时读取没有受锁保护。
应改为 `cur_data.toList()`。

#### P2-2 JSON 导出未聚合且文件名冲突

见 6.5。它适合单 rank debug，不适合验证全局上报。应增加 rank identity、scope 和可选
root aggregated dump。

#### P2-3 INT32 周期累计可能溢出

L0 和 planner L1 都是 INT32。应改为 INT64 或增加上限保护；不能因为 JSON accumulator
是 INT64 就认为 planner 数据不会溢出。

#### P2-4 `gpu_loads` 名称过度承诺

它是 dispatch 前由 mapping 推算的 owner count。建议重命名为
`expected_dispatch_loads`，另增 `observed_recv_loads`，两者差值作为正确性指标。

#### P2-5 缺少 mapping plan 的结构校验

plan apply 前应验证：

```text
0 < logic_expert_cnt[l] <= log2phy.width
有效 log2phy ID 位于 [0, phy_exp_num)
phy2log 与 log2phy 双向一致
phy_exp_num % ep_size == 0
本 rank 权重 tensor shape/dtype 与 model tensor 完全一致
所有 ranks 的 plan hash 相同
```

## 10. 验证现状与建议矩阵

| 验证项 | 当前状态 | 还需补充 |
| --- | --- | --- |
| mock mapping 原地更新后用于下一次 dispatch | 已有 Python 端到端 UT | 增加 GPU/CUDA Graph 版本 |
| logical heat 与 physical EP load 分离 | 已有 deterministic/random UT | 增加多 EP distributed case |
| invalid stats IDs 被忽略 | 已覆盖 | mapping helper 尚未覆盖 invalid ID |
| EP1 本地计数守恒 | 现有 JSON 样本通过 | 不能替代 EP>1 验证 |
| TP 去重公式 | 静态契约成立 | `TP=2,DP=2` 多进程测试 |
| root-only 指标发布 | 代码已修复 | mock reporter 断言次数、tags、values |
| predicted 与 observed dispatch 一致 | 未实现 | 对接各 router receive counts |
| reload 后 weights 与 mapping 同时生效 | mapping 有 mock UT | mock loader + 数值输出对比 + plan version |
| loader 单 rank 失败 | 未覆盖 | abort/rollback/timeout fault injection |
| 析构发生在 loading 中 | 未覆盖 | worker lifetime test |
| CUDA Graph padding 不计数 | 未覆盖 | capture/replay 不同 batch 测试 |
| reload 峰值显存 | 未测 | 大模型 layer reload memory benchmark |

推荐最小的集成验证配置：

```text
TP=2, DP=2, EP=4
logical experts=4, physical experts=8, top_k=2
mock log2phy 让每个 logical expert 的两个副本跨 EP ranks 放置
DP0/DP1 使用不同的确定性 logical routes
```

应一次性验证：

1. 同 DP group 的两个 TP ranks 产生相同 physical routes；
2. 两个 DP groups 因 `dp_rank` offset 选择不同副本；
3. DeepEP observed receive count 等于 predicted EP column；
4. world sum / TP 后等于 DP0+DP1 的真实 logical top-K；
5. 只有 world0 reporter 发布 4 组 EP tags；
6. mock 新 plan 原地 apply 后，下一 step dispatch 和统计同时切换；
7. 注入一个 rank load failure 时，所有 ranks 保留旧 plan，且无 hang。

## 11. 建议修复顺序

1. 先修 detached worker、异常传播、FAILED/abort/timeout 和析构 join，建立 reload 的安全
   状态机。
2. 将 plan creator 改为唯一 world root，并修复 Controller 快照读取。
3. 增加 plan version/hash/结构校验和全 rank two-phase commit，再讨论旧 weights 的明确
   rollback 策略。
4. 将统计改成周期聚合上报，增加 observed receive counts 与 predicted/observed 差值。
5. 补多进程 TP/DP/EP 测试和 CUDA Graph padding 测试。
6. 最后优化 log2phy fused kernel、INT64 counter 和 reload 峰值显存。

只有当“mock mapping 进入 dispatcher”“observed receive 与 predicted load 一致”“root 上报
值等于去重后的全局统计”“reload 全 rank 原子提交”四条同时成立，才能认为从 top-K 到
EPLB 新布局生效的端到端闭环被完整验证。
