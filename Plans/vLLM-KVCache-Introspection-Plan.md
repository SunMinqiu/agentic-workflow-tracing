# 直接读取 vLLM 前缀缓存内容：实施计划

2026-08-09 起草。**尚未实施**，本文只是方案。目标是把 `realized`（实际复用）从"猜"变成"看"，并据此判定 `logical`（模拟出来的上限）是否还有存在必要。

## 1. 要解决什么

现在 `realized` 有两个来源，都不理想。

`usage.prompt_tokens_details.cached_tokens` 恒为 `null`，这是 vLLM V1 的已知缺陷（[issue #44961](https://github.com/vllm-project/vllm/issues/44961)，2025-04 至今未修完）：引擎内部已经把逐请求的命中量算在 `RequestOutput.num_cached_tokens` 里，只是 OpenAI 兼容层构造 `UsageInfo` 时没有把它填进去。`--enable-prompt-tokens-details` 在 V1 上无效。

`/metrics` 的 Prometheus 计数器是精确的 token 级记账，但只有累加值，没有请求粒度。

两者都只能回答"用了多少"，不能回答"缓存里是什么"。后者需要读引擎进程内部的数据结构。

## 2. 已经确认的事实

以下来自用户运行中的服务器和 vLLM 官方文档，不是推断。

| 项 | 值 | 来源 |
| --- | --- | --- |
| vLLM 版本 | 0.26.0 | `/version` |
| 缓存内容数据结构 | `BlockPool.cached_block_hash_to_block`，块哈希 → `KVCacheBlock` | [官方 API 文档](https://docs.vllm.ai/en/stable/api/vllm/v1/core/block_pool.html) |
| 哈希算法 | sha256 | `cache_config_info.prefix_caching_hash_algo` |
| 前缀匹配粒度 | 16 token | `cache_config_info.prefix_match_unit` |
| 块大小 | 784（用户指定 16，被运行时改写） | `cache_config_info.block_size` |
| GPU 块数 | 1760 | `cache_config_info.num_gpu_blocks` |
| 缓存容量 | 1,325,785 token | `cache_config_info.kv_cache_size_tokens` |
| HTTP 接口 | 25 个路由，无任何缓存查询接口 | `/openapi.json` |
| 部署方式 | podman-hpc 容器，TP=4，挂载 `$VLLM_ROOT:/workspace` | 用户启动脚本 |

服务器还暴露 `/tokenize` 和 `/v1/chat/completions/render` 两个端点，能给出服务器眼中真实的 prompt 文本和 token 序列。这是第 3 阶段的关键输入。

## 3. 尚未确认、必须先查清的四件事

这四条决定方案可行与否，全部只需读源码，不改任何东西。**在它们有答案之前不要写实施代码。**

**块哈希是怎么构造的。** vLLM 的块哈希是**链式**的，每块的哈希由上一块的哈希加本块 token 共同决定（`vllm/v1/core/kv_cache_utils.py` 的 `hash_block_tokens`）。如果确实如此，就不能孤立地计算某一块的哈希，必须从序列开头一路链下来。这直接决定第 3 阶段能不能把缓存里的哈希映射回我们自己的 prompt。还要确认哈希是否掺入了多模态数据哈希、LoRA id、cache salt 等额外输入——SciLink 的带图调用会碰到第一项。

**字典的键是什么粒度。** `block_size` 是 784 而 `prefix_match_unit` 是 16，两者差 49 倍。`cached_block_hash_to_block` 到底按哪个粒度建键，决定了我们能分辨到多细。

**TP=4 时 `KVCacheManager` 在哪个进程。** 推测调度器只有一个实例、位于 engine core 进程，与 TP rank 无关，但没有验证。如果每个 rank 各有一份，读取方案要重做。

**784 这个值是怎么来的。** 假设是混合 mamba 模型把 attention 块大小上调对齐到 mamba page size（配置里有 `mamba_block_size=16`、`is_attention_free=False`）。这条不影响可行性，但影响我们怎么解释对齐损失。

## 4. 分阶段实施

### 阶段 0：源码调研

回答第 3 节的四个问题。产出一份简短记录，写明哈希函数签名、字典键的类型、`KVCacheManager` 的持有者。不动服务器，不动本仓库代码。

**通过条件**：四个问题都有明确答案，且哈希可以从 token 序列复现。

### 阶段 1：离线原型

在一台小机器上用 `LLM(...)` 以库的方式起一个极小模型，直接访问 `llm_engine` 内部拿到 `BlockPool`，验证两件事：能否遍历 `cached_block_hash_to_block`，以及能否用自己算的哈希在字典里命中。

选择离线验证而不是直接改服务器，是因为这一步失败率最高，而在自己机器上失败不影响任何正在跑的实验。

**通过条件**：发一个已知 prompt，能在字典里找到它对应的块。

### 阶段 2：服务端导出

在容器里加一个导出机制。两种做法，优先第一种。

**写文件**：起一个后台线程，收到信号或按固定间隔把当前常驻块哈希集合序列化到 `/workspace` 下。容器已经挂载了这个卷，主机侧直接可读，不需要改动 HTTP 表面，也不需要改端口或网络配置。1760 个块的哈希集合极小，导出成本可忽略。

**加路由**：给 api_server 挂一个 `/debug/kv_cache` 路由。更方便按需触发，但改动了服务器的对外接口，升级 vLLM 时更容易冲突。

无论哪种，都以挂载文件覆盖容器内源文件的方式实现，不重新构建镜像。

**通过条件**：主机侧能读到一份带时间戳的常驻块哈希清单。

**风险**：读取时机与调度器写入存在竞态。可接受的处理是承认最终一致——在两次 LLM 调用之间的空档导出，此时没有请求在跑（`num_requests_running=0`）。

### 阶段 3：与我们的 trace 对齐

对每一次 LLM 调用，需要拿到**服务器眼中的 token 序列**，而不是我们用 tiktoken 猜的。`/tokenize` 和 `/v1/chat/completions/render` 正好提供这个。拿到 token 序列后按阶段 0 确认的规则算出链式块哈希，与该次调用**之前**的那份导出清单求交集，交集大小就是这次调用真实可命中的 token 数。

这一步顺带修掉一个已知的独立缺陷：我们现在统计的 `our_tokens` 与服务器的 `input` 相差极大——纯文本调用因为漏算 tools schema（8189 token）而偏低到 0.33 倍，带图调用因为把 base64 当文本而偏高 34 到 125 倍。用服务器自己的分词结果就没有这个偏差。

**通过条件**：逐次调用的命中量算出来后，其总和与 `/metrics` 的 `prompt_tokens_cached_total` 差值一致。这是一个强校验——两条完全独立的路径得出同一个数。

### 阶段 4：判定 logical 的去留

有了逐次真值，`logical` 的误差就能直接量化。当前已知它在三个 cell 上都违反了自己的上界定义（`realized > logical`），说明现有实现有问题。

判定标准：如果真值与 `logical` 的偏差主要来自阶段 3 修掉的分词问题，那 `logical` 修好之后仍有价值，因为它是后端无关的，能覆盖没有 `/metrics` 的场景。如果偏差另有来源，再决定是否保留。

需要记住的前提是：当前这台服务器的缓存容量 132 万 token、实际用量 16 万、抢占 0 次，**从未发生驱逐**。在这个配置下 `realized` 本身就近似等于理想上限，`logical` 提供不了额外信息。它重新变得有意义的条件是缓存受限——更大的模型、更长的上下文、或多个 workload 共享服务器。

## 5. 兼容性约束

本仓库现有的 OpenAI 和 FreeInference 路径**不得受影响**。那两个后端正常返回 `cached_tokens`，`realized` 一直是有效的（现有结果中 GenoMAS 的 29% / 51% / 49% 即来自此）。新增能力只在厂商不上报时补位，判据已经写在 `analysis/kvcache/report.py`：仅当 `realized_frac` 为空时才读服务端数据。

## 6. 风险

**版本脆弱。** 方案依赖 vLLM 内部数据结构，不是公开 API。升级 vLLM 必须重新验证阶段 0 的四个结论。补丁应以挂载方式存在，并在文件头写明针对的版本号。

**哈希链的完整性。** 如果哈希掺入了我们无法复现的输入（多模态哈希、cache salt），阶段 3 的映射会失败。这是整个方案最可能的失败点，所以阶段 0 必须先确认。

**多模态调用。** SciLink 的图像调用在服务端如何参与哈希，需要单独验证。带图的调用占该 workload 的多数，不能只在纯文本上验证就下结论。

**上游可能先修好。** issue #44961 一旦合入，`cached_tokens` 就直接可用，阶段 2 和 3 的大部分工作就不必要了。开工前先看一眼上游状态。

## 7. 明确不做的事

不修改现有分析代码的行为，不重跑已有结果，不动 `logical` 的实现——直到阶段 3 拿到真值、阶段 4 有了判定依据为止。
