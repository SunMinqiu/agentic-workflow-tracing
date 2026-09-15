# GenoMAS 分批文献实验第一阶段

2026-09-09。复用 `output/genomas_references/corpus_manifest.json` 的六篇本地正文，每次目标调用消费新的正文片段。第一阶段使用调用适配入口做可控实验；在线检索和原生阅读工具留到下一阶段。

## 已实现

- GEOAgent 与 DomainExpertAgent 按 cohort、角色分别维护正文游标，每次调用消费未读片段。正文保持原文件顺序，不按角色重新排列章节。一个文档用完后继续同 cohort 的下一个文档，不循环。
- 默认参考块上限 16384 tokens，包含引用说明。前置位于 system/developer 消息之后、其他消息之前。后置插在最后一条 user 消息的最后一个 `**Task**` 前；没有该标记时追加到该消息末尾。两种方式都保留原任务文本。
- 修复旧入口依赖 `**Task**` 的限制。GEOAgent 的规划调用可以从当前 agent 的 `setups` 获取 cohort，不依赖共享客户端或上一调用的 cohort。
- 从 vLLM `/v1/models` 读取模型上下文上限，用服务端 tokenizer 校验完整消息。保留请求已有的输出预算；没有显式输出上限时预留 8192 tokens，不修改生成参数。空间不足只缩减参考块。
- 相同正文文件按内容哈希合并。不同 cohort 或角色的游标独立，因此跨 cohort、跨角色的相同正文不代表全局新内容。
- `reference_schedule.jsonl` 记录投放和跳过原因；`cohort_references/` 保存实际参考块；`messages.jsonl` 的 `cohort_reference` 记录片段范围、哈希、位置、模型、完整 prompt token 数及准备耗时。
- 删除旧的固定参考块生成和角色章节排序实现。配置为零时不启动文献调度。

## 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `GENOMAS_REFERENCE_TOKENS` | 16384 | 参考块 token 上限，0 为基线 |
| `GENOMAS_REFERENCE_POSITION` | front | front 或 back |
| `GENOMAS_REFERENCE_OUTPUT_RESERVE` | 8192 | 请求未给输出上限时预留的上下文空间 |
| `GENOMAS_REFERENCE_ROLES` | GEOAgent,DomainExpertAgent | 投放对象 |
| `GENOMAS_REFERENCE_MANIFEST` | 现有本地清单 | 按 cohort 筛选正文 |
| `GENOMAS_MAX_TIME` | 空，沿用上游 420 秒 | 用户确认本次文献试验使用 1800 秒 |

在已同步新代码、服务恢复后，使用现有运行入口。明确设置 `VLLM_URL` 为工作节点能够访问的服务地址，并固定模型、cohort、并发和缓存初始状态。以下命令分别启动三组，前一组完成后再启动下一组：

```bash
source cloudlab_env.sh
export VLLM_URL=http://127.0.0.1:18080
export GENOMAS_MAX_TIME=1800
GENOMAS_REFERENCE_TOKENS=0 bash scripts/launch_run.sh genomas A_c2_w1
GENOMAS_REFERENCE_TOKENS=16384 GENOMAS_REFERENCE_POSITION=back bash scripts/launch_run.sh genomas A_c2_w1
GENOMAS_REFERENCE_TOKENS=16384 GENOMAS_REFERENCE_POSITION=front bash scripts/launch_run.sh genomas A_c2_w1
```

三个完整运行都通过后，再重复有效配置；24K 档使用 `GENOMAS_REFERENCE_TOKENS=24576`。原任务和历史很长时，实际投放量可能低于上限。不要为达到文献上限缩短原任务输出预算。

## 验证与边界

旧 20260830_212953 运行中，GEOAgent 的 15 次请求都没有 `**Task**` 标记，其中 12 次请求文本没有可唯一识别的 cohort。远端 GenoMAS 的 `MultiStepProgrammingAgent.set_path_config()` 将当前 cohort 的设置写入 `self.setups`，新入口从对应 agent 的调用栈读取该字段。

当前两个 smoke cohorts 为 GSE123086 和 GSE123088，共用 Gawel 2019。之前模型记录的正文长度约为 27K tokens。16K 档通常只够每个 cohort、角色投放约两次；实际次数随完整请求可用空间变化。耗尽后记录 `exhausted`，不声称剩余调用持续获得新文献。

单元测试覆盖前后布局、逐次消费、耗尽、角色与 cohort 独立性、上下文不足不消费正文、输出预算优先、无 Task 标记及规划调用的运行时 cohort。性能结论必须等待真实服务实验。

2026-09-09 服务恢复后，确认 Mac 的 18000 端口提供 Qwen3.6-27B，上下文上限为 131072 tokens；补建 CloudLab 18080 到 Mac 18000 的反向隧道。已定向同步本次代码，保留远端被替换文件的 `.pre_pdf_rotation_20260909` 备份。真实 tokenizer 验证前置、后置均消费相同片段：参考块分别为 16353 和 10772 tokens，然后耗尽。之前的流式客户端异常尚未复现，不能报告已修复。

初次无文献运行 `GenoMAS_A_c2_w1_20260909_110619` 在首个 cohort 触发上游默认 420 秒超时，随后人工停止该无效实验，保留日志并恢复任务元数据。随后曾启动 1800 秒基线 `GenoMAS_A_c2_w1_20260909_111534`，用户指出应先定位超时原因，已于同次会话停止并恢复任务元数据。1800 秒配置不作为后续方案；保留原 420 秒预算，先诊断无文献基线。两次运行都未加入论文，沿用 A_c2_w1 的两 cohort 预处理模式，不含完整回归阶段。

首个超时 cohort 的 11 次调用合计输入 93929 tokens、输出 28651 tokens。服务端首 token 时间合计 7.558 秒，生成时间合计 410.433 秒；客户端调用总时间 430.940 秒，首次请求到最后响应跨度 435.204 秒。服务端生成占首 token 与生成时间之和的 98.19%。首 token 时间包含排队等开销，不等于纯 prefill。最后一次 DomainExpertAgent 调用计 8727 个输出 tokens，服务端生成耗时 124.845 秒，但保存正文经同一模型 tokenizer 计数只有 126 tokens。当前记录不足以精确拆分隐藏推理等额外输出。该结果说明无文献基线主要受 decode 支配，不能归因为论文注入。

用户随后确认允许 1800 秒预算，并要求复用已有基线。已直接启动 16K 前置文献试验，日志为 `logs/genomas_A_c2_w1_20260909_113522.log`。当前优先用同日同服务的无文献记录 `GenoMAS_A_c2_w1_20260909_110619` 比较单次调用时间，8 月 5 日、11 日作为历史参考。该无文献记录超时且被停止，只能作为已完成调用的时间参考，不能据此比较完整任务 E2E 或声称同预算成功率改善。本轮不再重跑基线。

验收同时报告任务完成情况、重试、实际新增正文量、缓存命中、服务端 prefill、客户端 TTFT、decode 和 workflow E2E。准备文献的 tokenization 时间属于 workflow 开销，不计为模型 prefill。前后两组使用相同选择规则；live 轨迹可能变化，比较实际文献序列和输出量后再归因。累计请求 prefill 除以 E2E 不能解释为并发运行的墙钟占比。
