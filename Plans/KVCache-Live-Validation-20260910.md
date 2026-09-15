# KVCache 真实验证结果：2026-09-10

本次沿用原 C2-12 与 C8 bundle，在 vLLM 0.26.0、Qwen3.6-27B、GPU blocks 274、CPU offload 32 GiB 配置下执行。C2-12 保持 packed、输出上限 32；C8 保持 serial、输出上限 256。每轮独立 salt，轮内共享 salt。所有主实验的首请求 cached tokens 均为 0。

| Workload | 重复 | 请求数 | Runtime 秒 | 逻辑 removal | 逻辑 return | CPU 副本候选 return | 采集状态 |
|---|---|---:|---:|---:|---:|---:|---|
| C2-12 | rep0 | 12 | 14.216 | 0 | 0 | 0 | unconfirmed |
| C2-12 | rep1 | 12 | 13.774 | 0 | 0 | 0 | unconfirmed |
| C2-12 | rep2 | 12 | 13.644 | 0 | 0 | 0 | unconfirmed |
| C8 | rep0 | 108 | 468.44 | 1767 | 220 | 148 | unconfirmed |
| C8 | rep1 | 108 | 468.38 | 1767 | 220 | 148 | unconfirmed |
| C8 | rep2 | 108 | 465.906 | 1767 | 220 | 148 | partial |

C2-12 runtime 中位数为 13.774 秒，C8 为 468.380 秒。C8 第三轮记录了一次 replay 超时，最终没有发现序号缺口，计数与前两轮一致；错误仍保留，状态为 partial。全部运行都没有部署后的发布边界确认，不能标完整。

C2-12 每轮有 34 次服务端 histogram 驱逐观测，但逻辑 removal 为 0。histogram 槽位观测与逻辑内容移除不是同一统计对象。C8 的 CPU 候选数没有传输完成证据，不能当成确认 reload 数。

## 受控序列

A → A 在无预热时 cached tokens 为 0、2976；预热后两个正式请求均为 2976。压力验证使用 A → 96 个首 token 不同的请求 → A，两个 A 均报告 0 cached tokens。被追踪的 A hash 在 group 0 中依次出现 GPU store、CPU store、GPU removal、CPU removal、GPU store、CPU store，验证了移除与再次出现，也说明 CPU 副本并未连续保留。

压力序列详情见 [逐事件核对](../replay/kv_controls_20260910/pressure/rep0/pressure_sequence_check.json)。它验证的是逻辑事件路径，没有直接测得物理槽位 eviction、重算 token 区间或实际 reload。

reset 后复用相同 salt 的两个 A 都仍有 2352 cached tokens。reset 请求成功不能单独证明全冷启动，代码已将 cold_by_reset 的展示改为 reset requested。原始 summary 保留；新记录使用 reset_requested。

## 采集开销

相同 salt 的初次对照因跨轮 cached tokens 不一致而排除，保留原始数据及排除原因。下面六次运行改用每轮独立 salt，每轮首请求 cached tokens 为 0，总 cached tokens 均为 24304。

| 配对 | OFF 秒 | ON 秒 | ON / OFF | ON KV JSON MB |
|---|---:|---:|---:|---:|
| 0 | 13.641 | 13.911 | 1.0198 | 107.2 |
| 1 | 13.777 | 13.834 | 1.0041 | 109.8 |
| 2 | 13.852 | 13.606 | 0.9822 | 117.6 |

配对 runtime 比值中位数为 1.0041。三个样本的差异跨越零，不能据此给出精确开销结论。这里测量客户端采集期间的请求运行时间，不包含采集初始化、完整报告生成、CPU 成本或尚未部署的物理 hook 开销。JSON 体积包括起始 replay 历史和派生数据，后续需要测量初始回放成本，并在保持可追溯性的前提下减少跨轮重复存储。

## 验收和剩余工作

- 132 项相关测试通过。46 个页面的 1020 个本地链接及锚点通过检查；8 组页面与视口检查、ledger 筛选及六个 pilot 排除标记通过验证。
- 物理 hook、publisher barrier、精确请求关联、transfer 完成解析及源码校验和回滚工具已准备。CloudLab 暂存目录为 `/users/Minqiu/kv-trace-handoff-20260910`，包内 8 个文件 SHA-256 校验通过。
- 尚未核对部署进程的四个 vLLM 源文件，也未安装或重启该 GPU 服务。需要 Perlmutter 可用 shell 与启动脚本后完成真实插桩验收、内部 request ID 对账和每个 worker 的日志覆盖确认。
- 三档 GPU 容量、CPU offload 开关和物理 hook 开关矩阵尚未执行；实际重算区间、GPU 完成时间与可归属等待仍未测得。下一步必须在部署源文件核对后补齐这些记录，再进入主矩阵。

入口：[实验总表](../results/index.html)、[C8 配置报告](../results/GenoMAS_A_c8_w4_20260818_153414/A_c8_w4/kvcache_report.html#replay-experiments)、[部署操作](VLLM-Physical-Trace-Deployment.md)。
