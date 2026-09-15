# KVCache 报告维护

`results/index.html` 是实验汇总，`results/all-experiments.html` 是完整指标索引。回放结果和 eviction 证据写入来源 task 已有的 `results/<run>/<task>/kvcache_report.html`，位于“Fixed-input replay experiments”一节。

汇总的 Workflow results 表保留每个任务的 IO 和 KVCache 直接入口。KVCache 对照表每行只保留一个 KVCache 入口，直接进入对应运行结果或 CPU swap eviction 证据。多次重复实验的入口先展示各次结果表。原始执行与回放分别标明，未采集事件的运行不生成 eviction 节。物理驱逐与 CPU swap 统计先展示，完整事件记录默认折叠。传输字节数和 worker 耗时来自指标窗口，p50 / p95 来自成功且请求匹配的 worker 传输记录。物理驱逐事件缺少字节大小及操作起止时间，显示未记录；缓存闲置时长单独展示。打开深链会展开目标所在的折叠区。

浅蓝色表示 CPU swap 实验且有 KVCache eviction 数据；黄色斜纹表示该项缺少测量。采集完整性单独显示。历史逻辑移除与已确认的物理驱逐分别命名。完整采集且零事件显示零，未采集显示状态。

来源由 `summary.bundle → bundle.source_cell` 确定，链接和颜色由 `analysis/kvcache/catalog.py` 统一生成。没有可确认来源的运行显示在完整指标索引，不按任务名称猜测归属。符号链接的重复引用只渲染一份运行内容。

容量与精度对照定义在 `config/kv_report_comparisons.json`，每组显式列出运行和基线。汇总与详情使用相同的表格渲染。完整逐请求缓存数据优先；引擎计数仅在可归属窗口中作为备用，并明确口径。CPU swap 历史运行的逐请求命中率是 49.56%，引擎计数为 42.55%，两者在证据区分别保留。

重建和检查：

```sh
PYTHONPATH=src python3 tools/rebuild_kv_reports.py
PYTHONPATH=src python3 tools/check_kv_report_links.py
PYTHONPATH=src python3 -m pytest tests/test_kv_report_integration.py tests/test_kv_reports.py tests/test_results_index.py tests/test_replay_sweep.py
```

`rebuild_kv_reports.py` 准备派生数据、刷新 task 页与汇总，并迁移旧报告链接。迁移清单位于 `output/kv_report_integration_manifest.json`。已替代的独立 sweep 和 block HTML 会被删除，原始 summary、requests、事件、服务配置和被引用图表保留。

报告模块只读派生分析。修改原始记录后先运行统一重建，重新准备失效的分析，再刷新报告。原 task 报告生成命令、回放命令和物理 trace 分析入口均接入同一报告结构。

历史实验事实与证据保留在 `Plans/KVCache-Causal-Validation-20260911.md`、`Plans/KVCache-Live-Validation-20260910.md` 和 `Plans/VLLM-Physical-Trace-Deployment.md`。已完成的报告整合计划和被替代的旧页面重构计划已删除。
