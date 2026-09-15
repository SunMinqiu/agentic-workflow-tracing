# 固定输入并发重放实验记录（2026-09-02，第一次尝试，失败）

## 目的

用固定输入（token-exact replay bundle）测并发对 prefill 时间占比的影响。三个不同的已录制 GenoMAS cell 各重放一遍，并发臂三组同时跑，对照臂串行跑（每组前清缓存 / 只开头清一次两种），比较 TTFT、queue/generation 分段和缓存命中。

## 环境

- vLLM 服务器：新机器（非 Perlmutter），经 Mac 隧道 127.0.0.1:18000 访问。
- 模型：Qwen/Qwen3.6-27B，served name `Qwen3.6-27B`（重启后已对齐旧写法），`--max-model-len 32768`，`VLLM_SERVER_DEV_MODE=1`，`--enable-prompt-tokens-details`。
- Bundle 来源 cell：`GenoMAS_3tasks_20260818_113547/A_c4_w1`（59 请求）、`GenoMAS_A_c8_w4_20260818_153414/A_c8_w4`（108 请求）、`GenoMAS_A_c10_w2_20260818_130924/A_c10_w2`（146 请求）。录制时最长 prompt 18,651 token。

## 时间线

1. 三组 runner `--mode paced` 同时启动，开跑前手动 reset 过一次 prefix cache。
2. A 组（A_c4_w1）完成 19/59 后，一个请求 300 秒收不到首 token，超时，进程退出。
3. B 组（A_c8_w4）同样超时退出（log 中即 `request c8e5702f... timed out`）。
4. C 组（A_c10_w2）存活但极慢（42/146），且 A、B 退场后它处于独占服务器状态，窗口混杂，手动停止。

## 根因

- runner 的每请求超时写死为 300 秒（`replay/runner.py` main 里 `timeout_s=300.0`），三组 paced 并发把上万 token 的 prefill 堆进队列，单请求排队超过 5 分钟即触发超时。
- runner 任何一个请求失败就放弃整个 bundle，且所有测量只在 bundle 全部完成后一次性写盘，中途失败则内存中已完成请求的数据全部丢失。

## 数据状态

本次尝试没有产生任何可用测量：out/conc 下仅有三个进度 log，无 requests.jsonl、无 metrics 快照、无 summary。三组全部需要重跑，串行基线两臂尚未开始。

## 重跑前的修改（待确认后实施）

1. runner 增加 `--timeout` 参数替代写死的 300 秒，并发臂用 1800 秒级别，把大时延如实测下来。
2. 逐请求结果改为完成一条追加写一条（流式落盘 requests.jsonl），中途失败时已完成部分保留，"截取三组共存窗口"的事后分析成为可能。

## 重跑清单

1. 应用上述修改。
2. 并发臂：reset 一次 → 三组 `--mode paced` 同时跑。
3. 串行冷基线：每组 `--reset-before` 依次跑。
4. 串行 warm 基线：开头 reset 一次，三组依次跑。
5. 对比各臂 summary 的 median_ttft_ms、server_timing（queue/generation）、per_request_cache.hit_rate；并发臂 Prometheus 窗口不可归属，逐请求 cached_tokens 可用。
