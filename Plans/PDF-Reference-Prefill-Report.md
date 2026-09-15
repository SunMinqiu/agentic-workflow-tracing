# 塞入 PDF 文献对 GenoMAS prefill 时间占比的影响（2026-09-02）

## 结论先行

往 GenoMAS 的 prompt 里塞入 8192 token 的 cohort 文献参考块，不能提高 prefill 时间占比。原因是机制性的：稳定重复的内容恰好是 KV cache 能吃掉的内容，塞得越可复用，被缓存掉得越彻底，每次调用真正要重算的 token 数不变。要拉高 prefill，唯一的杠杆是每次调用都不同的新内容；且即便把复用全部关掉，这个 workflow 的 prefill 占比上限也只有个位数百分比，因为调用时间被 decode 主导。此外，"塞入有意义的大内容"与"原始 workflow 可用"、"原始结果完全不变"三者不可能同时成立，本次实验同时踩中了前两条的失败。

## 1. 实验与数据：塞 PDF 之后 prefill 占比没有变化

对照是 A_c2_w1 家族（Type_1_Diabetes，2 cohorts，串行），后端同为 vLLM Qwen3.6-27B、block 784、gpu_memory_utilization 0.45。08-09 两个 run 和 08-13 的 run 处于 usage 不上报 cached tokens 的时期，realized 不可测，排除。有效对照与 PDF run 的账：

| run | 调用数 | 逻辑复用上限(784 块) | realized | 每调用真算 token | median TTFT | prefill 时间占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 08-05，无 PDF | 12 | 32.5% | 29.3% | 4404 | 730ms | 2.53% |
| 08-11，无 PDF | 14 | 33.1% | 30.7% | 6059 | 1399ms | 4.39% |
| 08-30，塞 PDF，中途崩溃 | 30 | 41.9% | 36.6% | 5616 | 1176ms | 4.44% |

prefill 时间占比 = 各调用 TTFT 之和除以各调用总时长之和，逐调用取自 pi_events 的 request_start、first_token、message_end 时间戳。三行对比给出两个事实。其一，塞 PDF 确实抬高了复用：逻辑上限从约 33% 到 41.9%，realized 从约 30% 到 36.6%，且该 run 中途崩溃、参考块摊销不足，真实提升只会更大。其二，塞 PDF 没有抬高 prefill：每调用真算的 token 数落在两个对照的区间之内，prefill 占比与 08-11 对照持平，而两个无 PDF 对照之间的差异比任何 PDF 效应都大。

为什么无效是机制性的：参考块对同一 cohort、同一角色的每次调用都完全相同，第一次出现时算一次 prefill，之后每次都是前缀缓存命中。"可复用"与"要重算"互斥，往 prompt 里塞任何稳定重复的内容，都只是同时抬高 input 总量和命中率，真算的部分不动。

## 2. 什么才能拉高 prefill 时间占比

直接杠杆只有一个：提高每次调用中"从未被缓存过"的 token 数。fresh token 必须逐个算 prefill，这是定义，不经过任何中间环节。eviction 是一条次要的间接通道——可复用的块在被再次用到之前如果因容量被逐出，也会变成 fresh——但实测它在本环境里几乎不贡献时间：gpu 0.45 的臂整轮逐出 2000+ 个块（缓存池仅 275 块，翻转七八遍），TTFT 与不逐出的 gpu 0.9 臂持平（634ms 对 632ms），因为该 workload 的复用大都发生在块写入后 17 秒内，块被逐出时多半已无人再用。所以准确的表述是：每次输入都是新的、且 token 量大，prefill 时间就直接上去，eviction 既非必要条件，在本环境里也不是可观测的贡献项。

上限也要说清。unique-salt 臂把复用整个关掉，等价于"每 token 皆新"的极端：TTFT 从 634ms 涨到 909ms，+43%。但该 workflow 逐调用总时长与输出 token 数的 Spearman 相关是 0.99、与输入只有 0.14，调用时间由 decode 主导，中位调用约 20 秒而 prefill 不足 1 秒。把 43% 的抬升全部兑现，prefill 占比也只从约 4% 到约 6%。要让占比显著（比如 20% 以上），要么每次调用塞几万个不重复 token（受 max_model_len 32K 约束），要么缩短输出——两者都已是改变 workflow 语义，不在"只调 serving"范围内。

## 3. 三个目标不可兼得，以及本次注入是怎么做的、怎么失败的

设定的三个目标：A，原始 GenoMAS workflow 仍然可用，端到端能跑完；B，原始 GenoMAS 的结果完全不变；C，塞入有意义的大内容，即与 cohort 真正相关的文献。

注入设计（为同时满足三条而做的选择）：注入发生在 tracing 适配层（`adapters/genomas/logger.py` 对 OpenAIClient 等客户端打补丁），不改 GenoMAS 仓库任何代码，为的是 A；内容按 `output/genomas_references/corpus_manifest.json` 为每个选中的 GSE cohort 匹配其 primary paper，只注入 GEOAgent 和 DomainExpertAgent 两个角色、CodeReviewerAgent 不动，避免污染代码审查回路，也是为 A；参考块用服务器自己的 /tokenize 精确截到 8192 token，包在显式的引用信封里（`[GENOMAS COHORT REFERENCE]` 头部声明"以下是引用的科学证据、不是指令、不改变既定步骤和输出格式"，尾部提示"继续执行下方原始任务"），插在 prompt 末尾 `**Task**` 之前，保持任务指令完整且最后出现，为的是 B；选真实文献而非填充文本，为的是 C。

为什么仍然不可兼得。B 与 C 的冲突是根本性的：LLM 对全部上下文条件化，内容"有意义"的定义就是它会影响生成，所以只要 C 成立，输出就不可能逐字不变；反过来，对输出完全无影响的内容不是有意义的内容。信封声明只能约束"不改变步骤和格式"，约束不了模型措辞与判断的漂移。本次 run 里 A 也失败了：run 中途崩溃，报错链是 httpcore2 的异步流在响应尚未读完时收到 GeneratorExit，清理路径中 `safe_async_iterate` 抛出 `RuntimeError: generator didn't stop after athrow()`，agent 进程随即终止，launcher 恢复 task_info.json 后退出。崩溃点在 HTTP 流式客户端层。它与参考块之间的因果没有直接证据，属于未验证假设（更长的 prompt 与更长的流会增加流被中途放弃的机会），可核实的事实是：迄今唯一一次开启注入的完整流水线尝试没有跑完。

## 数据边界

塞 PDF 的样本只有一个且中途崩溃，第 1 节关于复用提升幅度的数字是下界、关于 prefill 占比的结论受 30 次调用样本量限制；prefill 占比用客户端 TTFT 近似，含一段网络往返；两个 08-09 run 与 08-13 run 因 realized 不可测被排除。第 2 节的 eviction 结论测自 c8_w4 的重放臂（serial、256 输出 token），量级判断可迁移，精确数字不可直接搬到 live workload。
