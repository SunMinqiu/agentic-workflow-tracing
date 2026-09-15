# vLLM 物理 trace 部署与验收

2026-09-11 已核对 Perlmutter 容器中的 vLLM 0.26.0 源码并部署。模型为 Qwen3.6-27B，镜像 ID 为 `da9136512c30`。七个补丁目标的原始 SHA-256 保存在 `config/vllm_trace_v026_source.json`，源码审计保存在 `output/trace-audit-20260911/source`。部署使用只读源码挂载，原镜像和用户启动脚本保留。

实际运行短控制、C2/C8、快照导入和报告检查的完整步骤见 [用户手册第 11 节](../agentic_workflow_manual.md#11-实测-kvcache-eviction-与-swap)。本文件负责插桩安装、事件语义和回滚。

## 安装

在启动 vLLM 的 Python 环境内定位源码根目录。将本仓库放到该环境可读的位置，进入仓库后执行：

```sh
export KV_VLLM_SOURCE="$(python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent.parent)')"
python tools/prepare_vllm_trace_patch.py --source-root "$KV_VLLM_SOURCE" --output-dir output/deployed-vllm-trace
```

工具先核对七个部署文件的 SHA-256，再生成补丁。任何文件不匹配都会停止；需要审查差异并更新源码审计，不能跳过校验。核对生成的 `source_verification.json` 与 `vllm-kv-trace.patch` 后应用：

```sh
python tools/prepare_vllm_trace_patch.py --source-root "$KV_VLLM_SOURCE" --output-dir output/deployed-vllm-trace --apply
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export KV_TRACE_DIR="/dev/shm/kv-physical-validation"
export KV_TRACE_SESSION="kv-validation-$(date -u +%Y%m%dT%H%M%SZ)"
```

随后使用现有启动脚本重启服务，确保这些环境变量传入所有引擎和 worker 进程。保留原模型、dtype、block size、TP、DP、采样比例和 KV event 配置；容量实验只按实验单元变更 GPU 容量或 CPU offload。`KV_TRACE_DIR` 未设置时物理日志 hook 不写记录；publication barrier 补丁仍生效。

## 日志语义

- `PhysicalEviction`：分配路径复用一个 ref count 为零且仍缓存内容的物理槽位。包含 pool ID、block ID、allocation generation、全部缓存 hash 别名和驱逐前状态。
- `CacheInvalidated`：显式失效等其他清除路径。不能并入容量驱逐计数。
- `BlockAllocated`、`BlockAccessed`、`BlockReleased`、`BlockHashAssigned`：分别记录分配、touch、引用降至零和内容 hash 关联。最后释放到驱逐的间隔与最后 store 到 removal 的间隔分别统计。
- `RequestCacheLookup` 和 `BlockLookup`：记录实际查找路径及所选 block；`RequestAllocation` 仅说明调度分配，不能作为完成的重算 token 或耗时。
- `TransferSubmitted` 和 `TransferCompleted`：记录原 offload worker 的提交返回值与完成通知。完成记录保留 worker 报告的 bytes 和 transfer time，同时记录采集到完成通知的时刻。提交到通知的墙钟间隔不当成 DMA 时间。
- publication barrier：由 ZMQ publisher 线程排空当时已入队的事件并返回 publisher epoch、最后序号和服务端时间。它确认已发布的事件边界，不能证明 GPU 工作完成或所有待生成事件已经入队。

物理身份包含进程 epoch 和 pool ID。同一个数字 block ID 跨进程、重启或 allocation generation 不能直接合并。TP worker 的传输记录分开保留。`RequestRegistered` 通过客户端 `vllm_xargs.kv_trace_request_id` 关联内部请求。`TransferPlan` 记录内容、CPU 槽位及代次、GPU 目标和预期 worker 数；`TransferAllWorkersComplete` 记录调度器收齐 worker 完成的时刻。`RequestComputeScheduled` 与 `RequestComputeOutputObserved` 区分已调度和已收到计算结果。`RequestLoadBlocked`、`RequestLoadCompletionObserved`、`RequestLoadUnblocked` 记录调度器观测到的等待区间，不等于纯 DMA 时间。

## 真实验收

先执行 A → A、reset → A 和压力后再次请求 A。逐项核对请求 ID 是否与服务内部 ID 精确一致、物理槽位复用、hash 别名、transfer job 的提交与完成，以及 publication barrier 的开始和结束 epoch。每个参与进程的日志必须收齐；没有尾部确认时保留完整性未确认。

日志写入节点内存文件系统，必须在结束 GPU allocation 前复制出来。先确认请求已完成且传输任务已收齐，再从服务端复制原始日志并导入对应 repetition：

```sh
python -m agent_io_tracing.serving.kv_physical replay/EXPERIMENT/ARM/rep0 output/physical-traces/kv-physical-*.jsonl
python tools/rebuild_kv_reports.py
python tools/check_kv_report_links.py
```

导入保留原始文件的 SHA-256，通过 `RequestRegistered` 中显式记录的客户端请求标记匹配内部 ID，再将对应事件计入该 repetition。未归属的后台 store transfer 不凭时间邻近强行归到某个请求。报告列出排除数量和序号缺口。该导入路径本身不会将日志标成完整采集。

## 回滚

停止服务后执行以下命令，再用原脚本启动。工具核对补丁后的文件及备份 hash；若安装后另有修改，会停止以保留这些修改。

```sh
python tools/prepare_vllm_trace_patch.py --source-root "$KV_VLLM_SOURCE" --output-dir output/deployed-vllm-trace --restore
unset KV_TRACE_DIR KV_TRACE_SESSION
```

## 已验证与限制

2026-09-11 的短序列已经确认一次 CPU → GPU 任务：4 个 worker 均成功，合计 308,084,736 bytes，恢复 2,352 prompt tokens。GPU 重复请求命中 2,976 tokens，首次请求计算 2,991 tokens。压力控制观察到 1,004 次物理容量驱逐，196 个写出任务全部完成，最后请求 A 的 cached tokens 为零。具体原始数据和新一轮验证见本轮结果文件。

第一次补丁验证因 `remove_skipped_blocks` 接口读取错误失败，已修复并排除该次运行。第二版的 load 内容已关联 CPU 槽位，但请求状态中的 GPU block 列表为空；第三版改为读取传输任务的实际目标映射。第二版没有补写不存在的目标证据。

已有 Prometheus 总量按 `transfer_type` 分别解析，来自 worker 的时间相加不能当作请求阻塞时间。逻辑 CPU 副本候选、确认传输任务和物理内容记录分别统计。完整进程起止覆盖仍需独立确认。
