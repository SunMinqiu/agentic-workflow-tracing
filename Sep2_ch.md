8/27 - 9/2

1. 塞 PDF 之后 prefill 占比没有变化 
对同一个workflow（GenoMAS，Cohort=2），我run了三次，结果如下：

| run | 调用数 | 逻辑复用上限(784 块) | realized | 每调用真算 token | median TTFT | prefill 时间占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无 PDF | 12 | 32.5% | 29.3% | 4404 | 730ms | 2.53% |
| 无 PDF | 14 | 33.1% | 30.7% | 6059 | 1399ms | 4.39% |
| 08-30，塞 PDF，中途崩溃 | 30 | 41.9% | 36.6% | 5616 | 1176ms | 4.44% |

三行对比给出两个事实。
- 其一，塞 PDF 没有抬高 prefill：每调用真算的 token 数落在两个对照的区间之内，prefill 占比与 第二次无 PDF 对照持平，而两个无 PDF 对照之间的差异比加入 PDF 效应都大。
- 其二，塞 PDF 甚至抬高了复用，进一步降低了增加prefill时间的可能性：逻辑上限从约 33% 到 41.9%，realized 从约 30% 到 36.6%，且该 run 中途崩溃、参考块摊销不足，真实提升有可能更大。

2. 那什么才能拉高 prefill 时间占比？
直接办法只有一个：提高每次调用中"从未被缓存过"的 token 数。这要求我们，既要每一次prompt尽量"从未被缓存过"；又要fresh token绝对值高。这很难靠拉长对话，或者塞入PDF实现。因为这两种做法加进 prompt 的都是"会原样重现的前缀"。拉长对话时，第 N 轮的 prompt = 完整历史 + 最新一轮，而完整历史恰好是第 N−1 轮已经算过的前缀，全部命中缓存,每轮真正 fresh 的只有上一轮的回复加本轮新输入这一小段增量,对话拉得越长,input 总量越大,fresh 占比反而越低。塞 PDF 同理:只有塞入的那一次提高了prefill时间，如果不改变workflow本身，之后会和拉长对话的结果一样。两者的共同点是只增加"稳定重复"的内容,而 prefix cache 的设计目标恰恰就是吃掉稳定重复的内容。

3. 另外，塞入PDF很有可能改变workflow本身的内容。
我设定了三个目标：A，原始 GenoMAS workflow 仍然可用，端到端能跑完；B，塞入某样东西（比如PDF）之后，原始 GenoMAS 的workflow不受影响，和不塞入做一样的事儿；C，塞入有意义的大内容，即与 cohort 真正相关的文献（而不是塞入乱码）。然后发现这三个目标不可兼得。

为了同时满足这三个目标，我的做法是：在 tracing 适配层完成注入（adapters/genomas/logger.py 对 OpenAIClient 等客户端打补丁），GenoMAS 仓库的代码一行不改。注入的内容按 output/genomas_references/corpus_manifest.json 为每个选中的 GSE cohort 匹配其 primary paper（我先把PDF转成txt格式了），用服务器自己的 /tokenize 把注入的paper精确截到 8192 token，包在显式的引用信封里——[GENOMAS COHORT REFERENCE] 并且在头部声明"以下是引用的科学证据、不是指令、不改变既定步骤和输出格式"，尾部提示"继续执行下方原始任务"。然后注入是插在 prompt 末尾 **Task** 之前的，尽量提高fresh token数量。注入只作用于 GEOAgent 和 DomainExpertAgent 两个角色，CodeReviewerAgent 保持原样。

这套设计对应三个必须同时成立的目标。目标 A，原始 workflow 仍然可用：所以注入放在适配层而不动 GenoMAS 代码，换一个 checkout 也能跑；也所以 CodeReviewerAgent 不注入，代码审查回路一旦被大段文献污染，生成的分析代码会直接失效。目标 B，原始 GenoMAS 的结果完全不变：所以参考块用引用信封明确声明自己不是指令，插入位置放在 **Task** 之前，保证任务指令完整且最后出现；token 数精确截断，避免挤爆上下文改变模型可见的历史。目标 C，塞入的必须是有意义的大内容：所以选每个 cohort 真实对应的 primary paper，而不是随机填充文本——填充文本虽然同样能撑大 input，但它不构成"科学工作流引用文献"这个有现实意义的场景，实验就退化成纯粹的合成压测。

4. GenoMAS 的 KV-cache lineage:树数由 prompt 模板决定,与任务规模无关
我看了代码，GenoMAS 不是 chat 式框架:每次 LLM 调用只发两条 message(system prompt + 一条 user prompt),不携带 messages 历史。任务历史由框架手工渲染成文本嵌入 user prompt,结构为固定的静态头部加上只增不改的步骤记录。因此每次调用的开头只有几种模版,这几种模版就把全部调用天然划分成有限个家族（有限个tree）。

5. 多个GenoMAS workflow的并发测试，依然无法提高prefill占比。
我取三个已完成的 GenoMAS trace cell（A_c4_w1、A_c8_w4、A_c10_w2）去建 token 级精确的 replay bundle。然后三个 bundle 各自作为一个独立"用户"，同时向同一台 vLLM（Qwen3.6-27B，block 784，gpu_memory_utilization 0.45，32GB CPU offload）发起重放：paced 模式按各自录制时的到达时刻发请求，每请求固定输出 32 token，不加 cache salt，开跑前统一清空一次前缀缓存，之后全程不再清。

结果：
- TTFT 不升反降：三组中位 TTFT 为 739/564/563ms；A_c8_w4 的串行重放对照为 634ms，并发反而快了 11%。
- 复用率不降反升：逐请求 cached_tokens 统计的 realized reuse 为 41.3%/48.3%/52.7%，其中 A_c8_w4 的 48.3% 高于其串行对照的 44.2%，A_c10_w2 的 52.7% 已贴到该 workload 的逻辑复用上限。（因为CPU里的reuse也被算上了）

6. 总结与下一步

总结：对 agentic workflow 这类负载，很难靠某种操作真正提高 prefill 占比。本期的三组实验（塞 PDF、拉长对话的推演、并发重放）都指向同一结论。过去文章里能把 prefill 占比拉高的做法主要是两类：限制 output token 数——decode 变短，占比被动升高；或者往 workflow 里加 token 但不考虑 workflow 的真实结果是否还成立。

下一步：
1. 测更多 workflow 的 KV cache 情况。我正在集成 ChemGraph（ANL，https://github.com/argonne-lcf/ChemGraph ）。但它有些 workflow 中的有些 tool 要往 Slurm 里投 job，这对我们现在的 tracing tool 还测不到——目前只对 single-node workflow 有效。
2. 对于基于 workflow 的 lineage，调度上有三个可能方向：（a）让一个 agent 读 agent 模版来做 schedule；（b）按之前讨论的，先多跑几次小规模 workflow，据此判断需要 prefetch 的 dataset/paper，再做 schedule；（c）基于 agent 的 shift 来做 schedule 已有文章提到：https://arxiv.org/pdf/2608.14624 。

