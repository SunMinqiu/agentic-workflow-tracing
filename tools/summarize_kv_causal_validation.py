"""Write a source-linked Chinese explanation of the completed causal validation."""
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent_io_tracing.serving.kv_report_data import prepare_report_data,read_rows
from agent_io_tracing.analysis.kvcache.catalog import Catalog


def report_link(rep, label, kind='run'):
    catalog=Catalog(ROOT)
    row=catalog.by_rep[rep.resolve()]
    return f'[{label}]({catalog.href(row,ROOT/"Plans",kind)})'


def main():
    base=ROOT/'replay/kv_physical_c8_20260911/gpu274_cpu32'
    records=[]
    for rep in sorted(base.glob('rep*')):
        if not (rep/'physical_traces').is_dir():continue
        data=prepare_report_data(rep)
        metadata=json.loads((rep/'summary.json').read_text())
        waits=read_rows(rep/'request_load_waits.jsonl')
        records.append((rep,data,metadata,waits))
    if len(records)!=3:raise RuntimeError('This explanation requires all three C8 repetitions')
    first=records[0][1];causal=first['causal_trace']
    counts=causal['candidate_reconciliation']
    for rep,data,_,_ in records:
        if data['causal_trace']['candidate_reconciliation'] != counts:
            raise RuntimeError('Candidate classifications differ across repetitions; update the explanation')
        candidates=read_rows(rep/'candidate_reconciliation.jsonl')
        if sum(bool((r.get('cpu_store_boundary_evidence') or {}).get('boundary_inside_completed_chunk')) for r in candidates)!=counts['computation_observed']:
            raise RuntimeError('Checkpoint boundary evidence differs; update the explanation')
        if data['causal_trace']['reload_records_with_stable_gpu_allocation']!=data['causal_trace']['confirmed_reload_content_records']:
            raise RuntimeError('Some GPU allocation identities remain unconfirmed; update the explanation')
    recomputed=[r for r in read_rows(base/'rep0/candidate_reconciliation.jsonl') if r['status']=='computation_observed']
    observed_tokens=sum(r.get('completed_content_tokens',0) for r in recomputed)
    text=['# KVCache eviction 与 swap：2026-09-11 真实验证',
          '', '本次 KVCache 测量与报告目标已完成：记录实际 eviction、双向传输、完成确认、后续计算和请求等待，并提供可追溯的图表与明细。Mamba 数值状态验证作为附带发现保留，不属于本次验收条件；采集窗口和缺失状态仍按实际证据标注。',
          '',f'原 C8 bundle 的三次重复都得到同一结果：148 个“CPU 副本候选”中，{counts["confirmed_load"]} 次通过 CPU 搬回，{counts["computation_observed"]} 次重新计算，未解决的候选为 {counts["unresolved"]}。这些是新补丁采集的新运行，历史日志没有补写确认字段。',
          '', '**先分清这几个数字**', '',
          '| 每轮数量 | 它数的是什么 |', '| ---: | --- |',
          f'| {first["physical_trace"]["physical_evictions"]:,} | 分配新空间时，确实把仍有缓存内容的 GPU 物理槽位腾给其他用途的事件。 |',
          '| 1,804 | GPU 的 BlockRemoved 中记录的 hash 移除次数，包括内容别名和其他 hash 更新。 |',
          '| 1,767 | 合并重复移除后，开始等待同一内容再次出现的区间。 |',
          '| 220 | 在观察窗口中等到了再次出现的区间。 |',
          '| 148 | 再次出现时，逻辑事件中仍能连续追踪到 CPU 副本的候选。 |',
          f'| {causal["confirmed_load_jobs"]} | 确认完成的 CPU→GPU 传输任务。一个任务可以带回多个缓存组的内容。 |',
          f'| {causal["confirmed_reload_content_records"]} | 这些任务包含的内容恢复次数，涉及 {causal["unique_loaded_contents"]} 份不同内容。重复恢复同一份内容会再次计数。 |',
          '', '物理槽位事件与逻辑 hash 事件是两种计数。一个槽位可以关联多个 hash，hash 更新也不一定腾出物理槽位。',
          '', '写到 CPU 也不等于发生 GPU 驱逐。vLLM 的 native offload 会主动写出可保存的缓存：本次 C2 每轮向 CPU 写出 8,009,023,488 bytes，约 8.01 GB，但物理缓存驱逐为 0，也没有 CPU→GPU 搬回。因此需要分别记录写出、驱逐和恢复。',
          '', 'vLLM eviction histogram 每轮有 2,076 次观测。源码在检查槽位是否还有缓存内容之前，就记录被跟踪槽位的生命周期结束。因此它包含没有缓存内容的槽位等情况，不能用 2,076 代替上表中的物理容量驱逐数。',
          '', '148 个候选跨越不同缓存组，拆开后关系如下：', '',
          '| 缓存内容 | CPU 副本候选 | 确认搬回 | 确认重新计算 |', '| --- | ---: | ---: | ---: |',
          '| attention 组 3 | 94 | 73 | 21 |', '| Mamba 组 0、1、2 合计 | 54 | 54 | 0 |', '| 合计 | 148 | 127 | 21 |',
          '', '18 个搬回任务各恢复三个 Mamba 组的状态，共 54 份状态；还恢复了 73 个 attention 内容块，共计 127 条内容恢复记录。73 × 784 = 57,232，正好对应调度器计入外部缓存的 prompt tokens。三个 Mamba 状态服务于同一段前缀，不能把它们再各乘 784 加进 token 数。',
          '', '**为什么这次能确认搬回**', '',
          '每个确认的传输都连上了这条记录链：请求 ID → 调度器传输计划 → CPU 槽位及代次、内容 hash、GPU 目标槽位 → 4 个 worker 成功完成 → 调度器收齐完成通知 → 搬回的槽位出现在该请求后续完成的计算步骤中。',
          '', 'GPU 目标槽位还核对了分配代次：三轮各 127 条恢复记录，从传输计划到后续计算完成之间，目标槽位都没有重新分配。这排除了“block ID 相同，但已经属于另一轮分配”的误配。',
          '', '每轮 280 个双向传输任务都收齐了完成记录。其中 262 个写出任务产生 1,048 个 worker 样本，18 个搬回任务产生 72 个 worker 样本。逐任务汇总的字节数、样本数和时间与 Prometheus 两个方向的总量一致。',
          '', '请求也做了独立核算：调度器将 347,296 个 prompt tokens 计入 GPU 本地缓存，将 57,232 个计入外部缓存，另外 411,760 个完成了 prompt 计算，相加等于全部 816,288 个输入 tokens。三轮各 108 个请求全部对得上。这是分配与计算数量的核算；各缓存组的数值状态是否正确，还需要单独验证。',
          '', '**那 21 个候选为什么没有从 CPU 恢复**', '',
          '21 个候选都出现了相同的查找结果：同一前缀的 CPU full-attention 缓存组命中，但 CPU Mamba 组 0 未命中。模型需要不同层各自的缓存，只有 attention 副本不能让整个前缀跳过计算。vLLM 的查找逻辑在一个必需组没有命中时返回零可恢复 tokens。',
          '', '以请求 `155d40686b8a4017908304cc2ccd2604` 为例，输入有 2,873 tokens。GPU 已有前 1,568 tokens；CPU attention 副本覆盖接下来的 784 tokens，但对应的 Mamba 组没有找到。实际完成的计算区间如下，区间右端不包含在内：',
          '', '| 计算区间 | token 数 | 含义 |', '| --- | ---: | --- |',
          '| [1568, 2352) | 784 | CPU attention 上有副本，但这次仍重新计算的内容。 |',
          '| [2352, 2864) | 512 | 后续 prompt 计算。 |',
          '| [2864, 2873) | 9 | prompt 末尾。 |',
          '', '进一步逐条检查三轮记录后，三轮各 21 个候选在查找未命中之前，都没有对应 Mamba 内容的 CPU 分配或写完记录。这个例子的第一次 Mamba 写出出现在本次请求计算之后。现在能证明的是“恢复所需的缓存组不齐”，还不能把原因归为 CPU 淘汰。这里的“没有记录”限定于已经采集的 trace。下一步应检查不同缓存组的保存边界和写出条件。',
          '', '继续追溯上一次写出 attention 副本的请求，三轮各 21 个候选所需的前缀边界都位于它先前计算块的中间。上例之前的一步计算为 [1568, 3136)，后来需要的检查点 2352 位于中间。attention 的分块内容可以保留下来，而当前源码说明 Mamba 状态在计算块结束处形成。这个机制与“attention 有副本，Mamba 没有写出记录”的观察吻合。每条候选的原明细现在都列出所需边界、先前计算区间和写出请求。',
          '', '确认各缓存组状态正确后，优化实验可以比较：在可能复用的前缀边界结束计算块，能增加多少完整缓存组的恢复机会，以及额外分段会增加多少运行时间。单独调整 CPU 淘汰顺序不能补出尚未生成的检查点。',
          '', f'21 个候选的返回内容都逐 token 匹配了原请求输入，并且整段都处于已完成的计算区间。每段 784 tokens，共 {observed_tokens:,} tokens。这是这 21 个候选对应的重算量，不代表整个 C8 的全部重算量。',
          '', '**CPU 自身淘汰了多少内容**', '',
          '原三轮 C8 每轮各记录 1,230 次 CPU 槽位分配、1,230 次内容标记可用和 1,230 次 CPU 容量淘汰。每轮被淘汰的内容中，562 份由本轮请求保存，668 份来自此前其他请求，来源未知为 0。这是 CPU 池中的槽位代次计数，按四个 GPU worker 共同使用的逻辑 CPU 槽位计一次，与 GPU 的 1,697 次物理驱逐分别统计。',
          '', '从 CPU 内容被标记可用，到其被淘汰的间隔，中位数三轮分别为 221.825、242.345、244.033 秒。第二轮最大值约 730.720 秒，可以超过本轮约 456.557 秒的运行时间，因为该内容可能在上一轮已经存入 CPU。原始记录逐条保留 CPU 池、槽位、代次、内容 hash、写出请求和触发淘汰的请求。',
          '', '**传输了多少，花了多久**', '',
          '每轮 GPU→CPU 传输 63,146,164,224 bytes，约 63.15 GB；CPU→GPU 传输 6,521,749,504 bytes，约 6.52 GB。GB 按十亿字节计算，CPU 配置容量 32 GiB 使用另一种单位。',
          '', '| 轮次 | 整轮运行秒数 | GPU→CPU 的 worker 时间合计，秒 | CPU→GPU 的 worker 时间合计，秒 | 18 次请求等待合计，毫秒 |',
          '| --- | ---: | ---: | ---: | ---: |']
    for rep,data,metadata,waits in records:
        directions=data['aggregate_transfers']['directions']
        text.append(f'| {rep.name} | {metadata["wall_s"]:.3f} | {directions["GPU_to_CPU"]["worker_time_s"]:.6f} | {directions["CPU_to_GPU"]["worker_time_s"]:.6f} | {sum(r["observed_blocked_to_unblocked_s"] for r in waits)*1000:.3f} |')
    text += ['', 'worker 时间由各 GPU 的传输计时相加，包含并行工作。请求等待从调度器被观测为等待缓存，到允许请求继续执行，包含调度延迟。两列不是同一个时间，也不能从整轮运行时间中直接减掉它们来预测关闭 swap 的速度。',
             '', '传输计时来自各 GPU 上的 CUDA 起止事件。源码先让传输流等待所需计算和前一项传输，再记录计时起点；完成通知则由后续轮询观察。因而，表中的 CUDA 传输时间没有包含起点之前的流等待，通知等待由另一组记录描述。参见 [已核对的传输实现](../output/trace-audit-20260911/source/vllm/v1/kv_offload/cpu/gpu_worker.py)。',
             '', '整轮为什么仍要约 457 秒？原 C8 第一轮的 108 个请求串行执行，每个请求固定生成 256 tokens。服务端记录的“首个输出 token 之后继续生成”的时间合计约 390.20 秒；等到首个 token 的时间合计约 45.70 秒，后者包含排队和 prompt 处理。这两项没有覆盖全部客户端端到端时间。缓存复用可以减少 prompt 计算，但后续输出生成仍占了整轮的大部分时间，所以计算 token 数明显下降时，整轮时间也可能只小幅变化。',
             '', '**缓存被驱逐前放了多久**', '',
             '这里统计 C8 第一轮实际发生的物理缓存驱逐。每条记录都以驱逐时刻为终点，起点不同就代表不同的问题。p50 是中位数，p95 表示 95% 的已观测间隔不超过该值。', '',
             '| 从什么时刻开始计时 | 有记录的驱逐次数 | p50，秒 | p95，秒 |', '| --- | ---: | ---: | ---: |']
    labels={'time_since_last_release_s':'最后一次引用数归零，不再被请求占用', 'allocation_age_s':'这一次物理槽位分配', 'time_since_last_access_s':'最后一次记录到前缀缓存命中操作'}
    for key, values in first['physical_trace']['duration_statistics'].items():
        text.append(f'| {labels[key]} | {values["count"]:,} | {values["p50"]:.3f} | {values["p95"]:.3f} |')
    text += ['', '例如，“引用数归零后 55 秒被驱逐”表示该槽位在这段时间没有被请求占用，直到被分配器回收。前缀缓存命中操作只覆盖有该类记录的子集，不能当成 GPU 最后一次读取这块内存的时间。页面中的分布图与这张表使用同一批物理驱逐记录。',
             '', '**短请求检查给出了更直观的例子**', '',
             '| 同一个 2,991-token 请求 | GPU 缓存提供 | CPU 恢复 | 完成的 prompt 计算 |',
             '| --- | ---: | ---: | ---: |', '| 首次请求 | 0 | 0 | 2,991 |', '| 直接重复 | 2,976 | 0 | 15 |', '| 清掉 GPU 缓存后重复 | 0 | 2,352 | 639 |',
             '', '三次请求生成的 8 个输出 token 完全相同。第三次由 4 个 worker 合计搬回 308,084,736 bytes；worker 传输时间合计 12.819 毫秒，调度器观测的请求等待为 14.473 毫秒。',
             '', '**采集质量与适用范围**', '',
             '三轮 C8 的物理日志均包含引擎和 4 个 worker，序号没有缺口。所有传输任务和请求 token 核算均闭合。完整进程起止覆盖仍保留未确认，不能把请求核算通过当成进程整个生命周期的证明。',
             '', 'rep2 的网络事件采集记录过一次超时。重试后，确认的起止发布边界之间的 464 个 batch 序号全部在文件中，没有缺号。报告保留采集器的 partial 状态，同时单独显示这个序号核查结果。',
             '', '历史 C8 的 148 个候选仍保持历史证据级别。本次三个新运行都出现相同的 127 与 21 分类；不能把新实验的 trace 冒充历史运行的 trace。',
             '', '**已经修正的实现和报告**', '',
             '- 核对并挂载七个 vLLM 源码补丁目标，增加 CPU 槽位代次、传输计划、全部 worker 完成、请求计算完成与等待记录。',
             '- 使用显式客户端请求标记关联内部请求 ID，拒绝错误流和未收到结束标记的响应。',
             '- 分方向解析原有 Prometheus 传输总量，并与逐任务数据核对。',
             '- 原 block ledger 直接显示候选对应的请求、确认搬回或重算结论；总表和配置页链接到同一份明细。',
             '- 增加一次确认搬回的时间图，分别标注 worker 通知间隔、传输计时和请求等待。没有 CPU offload 的页面不生成 CPU 搬回零行明细表。',
             '- 日志改写节点内存文件系统，使用完整 JSONL 记录和 SHA-256 快照拷回，避免在运行中直接复制不断增长的文件。',
             '', '**开销对照和后续实验**', '',
             '同一节点、GPU 274、CPU 32 GiB 下，C2 关闭物理记录三轮中位数为 12.848 秒，开启为 12.845 秒；六轮 GPU 缓存命中量均为 24,304 tokens。两组顺序为先 OFF 后 ON，未交错；CPU 资源占用没有采样。这个小实验没有显示明显额外时长，不能据此给出精确的开销上限。',
             '', capacity_status(),
             '', same_node_status(),
             '', '**直接查看证据**', '',
             '- [实验总表](../results/index.html)',
             '- ' + report_link(base/'rep0','C8 配置及各轮结果','config'),
             '- ' + report_link(base/'rep0','C8 第一轮 eviction 明细','eviction'),
             '- [148 个候选的逐条解释](../replay/kv_physical_c8_20260911/gpu274_cpu32/rep0/candidate_reconciliation.jsonl)',
             '- [逐请求 token 核算](../replay/kv_physical_c8_20260911/gpu274_cpu32/rep0/causal_requests.jsonl)',
             '- [确认搬回的内容](../replay/kv_physical_c8_20260911/gpu274_cpu32/rep0/confirmed_reloads.jsonl)',
             '- [逐任务传输及完成记录](../replay/kv_physical_c8_20260911/gpu274_cpu32/rep0/causal_transfers.jsonl)',
             '- [CPU 淘汰内容及保存时长](../replay/kv_physical_c8_20260911/gpu274_cpu32/rep0/cpu_evictions.jsonl)',
             '- [调度器观测的请求等待](../replay/kv_physical_c8_20260911/gpu274_cpu32/rep0/request_load_waits.jsonl)',
             '- [记录开关对照](../replay/kv_trace_overhead_20260911/overhead_findings.json)',
             '- [C8 跨节点输出序列对比](../output/kv-causal-validation-20260911/c8-output-equivalence-cross-node.json)',
             '- [C2 跨节点输出序列对比](../output/kv-causal-validation-20260911/c2-output-equivalence-cross-node.json)',
             '- [C2 各缓存组的 GPU 前缀复用核查](../output/kv-causal-validation-20260911/c2-hybrid-prefix-audit.json)',
             '- [容量实验顺序](../output/kv-capacity-matrix-20260911/schedule.json)',
             '- [各轮 bundle 原文件指纹核查](../output/kv-causal-validation-20260911/bundle-provenance.json)',
             '- [vLLM 源码指纹](../config/vllm_trace_v026_source.json)', '']
    target=ROOT/'Plans/KVCache-Causal-Validation-20260911.md'
    target.write_text('\n'.join(text))
    print(target,flush=True)


def capacity_status():
    directory=ROOT/'output/kv-capacity-matrix-20260911'
    receipts=[json.loads(path.read_text()) for path in sorted(directory.glob('gpu*-rep*.json'))]
    lines=[f'关闭 CPU offload 的 GPU 192、274、548 三档实验沿用原 C2 和 C8 bundle，每档计划三轮。当前完成 {len(receipts)}/9 个容量与轮次组合，每个组合包含两个 bundle。下表只包含完成且保存物理日志的组合。', '', '“完成的 prompt 计算”同时包含首次计算和重新计算。它回答实际执行了多少输入 token 的计算，不能全部称为驱逐造成的重算。各列表按 rep 编号排列。']
    for workload in ['c2-12','c8']:
        lines += ['', f'**{workload.upper()}：关闭 CPU offload 的容量对比**', '', '| GPU block 数 | 完成轮数 | 每轮运行秒数 | 每轮完成的 prompt 计算 tokens | 每轮物理驱逐次数 |', '| ---: | ---: | --- | --- | --- |']
        for capacity in [192,274,548]:
            measured=[]
            for receipt in sorted(receipts,key=lambda r:r['rep']):
                if receipt['gpu_blocks']!=capacity:continue
                rep=ROOT/f'replay/kv_capacity_physical_{workload}_20260911'/f'gpu{capacity}_cpu0'/f'rep{receipt["rep"]}'
                raw=json.loads((rep/'summary.json').read_text());data=prepare_report_data(rep)
                measured.append((raw['wall_s'],data['causal_trace']['completed_prefill_tokens'],data['physical_trace']['physical_evictions']))
            if measured:
                values=[' / '.join(f'{r[i]:.3f}' if i==0 else f'{r[i]:,}' for r in measured) for i in range(3)]
                lines.append(f'| {capacity} | {len(measured)}/3 | '+ ' | '.join(values)+' |')
        sample=next((ROOT/f'replay/kv_capacity_physical_{workload}_20260911').glob('*/rep*/summary.json'),None)
        if sample:
            lines += ['', report_link(sample.parent,workload.upper()+' 容量实验与逐轮明细','config')]
    lines += ['', 'GPU 192 的 rep0 与原 CPU offload 开启实验在 nid001181 上运行；续跑使用 nid001101，两台节点均为 4 张 A100-SXM4-40GB。节点与 GPU UUID 已保存，跨节点的运行时间差不能全部归因于缓存配置。', '', '本次部署源码在启用 KV connector 且模型含 Mamba 层时，按缓存组查找 GPU 命中，并使用各组命中长度的最大值；没有 connector 时走普通的共享前缀查找路径。参见 [已核对的调度器源码](../output/trace-audit-20260911/source/vllm/v1/core/sched/scheduler.py)。因此，开启 native offload 会同时改变 GPU 前缀查找路径。C2 在 GPU 274 下没有 CPU→GPU 搬回，但开启 offload 的原运行仍比关闭后的 rep0 少完成 2,352 个 prompt tokens 的计算。这个差额来自三个请求各多计入 784 个 GPU 本地缓存 tokens，不能记作 CPU 恢复收益。', '', '固定 temperature 和 seed 没有建立跨配置输出完全相同的证明。原 CPU offload 开启的三轮 C2、C8 各自输出 token 序列完全一致；与新节点关闭 offload 的 GPU 274 rep0 比较，分别有 7/12 和 92/108 个请求的输出序列不同，其中也包括没有缓存命中的请求。节点和执行路径同时变化，目前不能定位差异原因。短控制实验的 8-token 输出相同，只证明该短控制的结果。缓存组缺失的原因分析仍需检查保存边界和写出条件。']
    lines += ['', '**附带发现：模型缓存状态一致性**', '', '三个 C2 请求的记录进一步显示：native offload 路径只触碰了 attention 组的已缓存 GPU block，其他三个组使用新分配的槽位，外部缓存分配为 0。第一个请求从 token 784 开始计算，另外两个从 2352 开始；在调度记录的其他组 block 表中，前缀位置是空占位。参见 [逐条证据](../output/kv-causal-validation-20260911/c2-hybrid-prefix-audit.json)。这构成需要检查状态补齐路径的正确性疑点，当前采集没有数值级 KV 状态校验，不能直接判定输出错误。', '', '在把 offload 的少算量解释为有效加速之前，应先核查各组 GPU 命中长度不一致时，native connector 是否补齐了缺失的 Mamba 状态。当前源码中取各组命中长度最大值的分支，注释说明依赖 NIXL 无条件恢复 Mamba 状态，但分支条件覆盖所有 connector；native 路径在外部命中 tokens 为 0 时会直接返回。该问题与 CPU 检查点缺失分别记录，后续先验证状态正确性，再评估检查点保存策略。']
    lines += ['', '进一步只读检查状态复制实现后，按这些调度记录可以推导出更具体的路径：新请求的上一状态列为 `(已计算 tokens - 1) // 784`。第一个样本得到列 0，该列在 Mamba block 表中是空占位 block 0；另外两个样本得到列 2，同样是空占位。状态复制代码使用该列作为源，而 metadata 仅根据已计算 tokens 大于 0 标记存在初始状态。这还需要运行时数值验证，但已经给出了明确的检查位置。对应源码为 [Mamba 状态复制](../output/trace-audit-20260911/extra-source/vllm/v1/worker/mamba_utils.py)、[V2 初始状态索引](../output/trace-audit-20260911/extra-source/vllm/v1/worker/gpu/model_states/mamba_hybrid.py) 和 [Mamba attention metadata](../output/trace-audit-20260911/extra-source/vllm/v1/attention/backends/mamba_attn.py)。']
    lines += ['', '重新启动服务后的同配置重复也没有输出完全一致：同在 nid001101、GPU 274、关闭 offload 的 C8，rep1 和 rep2 相对 rep0 分别有 62/108 和 61/108 个请求输出不同；C2 分别为 3/12 和 5/12。驱逐数和完成的 prompt 计算量仍相同。这说明输出差异并非只在开关 offload 时出现；需要数值状态或其他确定性检查来定位缓存正确性问题。参见 [同配置输出重复性核查](../output/kv-causal-validation-20260911/capacity-output-repeatability.json)。']
    return '\n'.join(lines)


def same_node_status():
    directory=ROOT/'output/kv-capacity-matrix-20260911'
    receipts=[json.loads(p.read_text()) for p in sorted(directory.glob('same-node-offload-rep*.json'))]
    lines=['**同节点开启 CPU offload 的补充对照**', '']
    if not receipts:
        return '\n'.join(lines+['尚无完成并保存物理日志的补充轮次。'])
    from check_kv_output_equivalence import compare
    lines += [f'在 nid001101 上完成 {len(receipts)} 轮补充对照，GPU 274、CPU 32 GiB。每轮重新启动服务，先跑 C2，再跑 C8，与关闭 offload 的容量实验顺序相同。原三轮 offload 实验另行保留，不合并计算中位数。', '', '| bundle | 补充轮次 | 整轮秒数 | 完成的 prompt 计算 tokens | GPU 物理驱逐 | 确认搬回内容次数 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    comparisons=[]
    cpu_notes=[]
    for workload in ['c2-12','c8']:
        for receipt in receipts:
            index=receipt['rep']
            rep=ROOT/f'replay/kv_same_node_physical_{workload}_20260911/gpu274_cpu32/rep{index}'
            data=prepare_report_data(rep); raw=json.loads((rep/'summary.json').read_text())
            causal=data['causal_trace']
            lines.append(f'| {workload.upper()} | {index} | {raw["wall_s"]:.3f} | {causal["completed_prefill_tokens"]:,} | {data["physical_trace"]["physical_evictions"]:,} | {causal["confirmed_reload_content_records"]} |')
            if workload=='c8':
                cpu=causal['cpu_cache'];counts=causal['candidate_reconciliation']
                cpu_notes.append(f'C8 补充 rep{index} 仍得到 {counts["confirmed_load"]} 次确认搬回、{counts["computation_observed"]} 次确认重算，未解决候选为 {counts["unresolved"]}。CPU 淘汰为 {cpu["eviction_events"]} 次，其中 {cpu["victims_created_in_run"]} 份由本轮 C8 保存，{cpu["victims_from_other_requests"]} 份来自此前请求。')
            for label,reference in [('same-node-off',ROOT/f'replay/kv_capacity_physical_{workload}_20260911/gpu274_cpu0/rep0'),('original-on',ROOT/f'replay/kv_physical_{workload}_20260911/gpu274_cpu32/rep0')]:
                result=compare(reference,rep)
                output=ROOT/f'output/kv-causal-validation-20260911/{workload}-{label}-vs-supplement-rep{index}.json'
                output.write_text(json.dumps(result,indent=2)+'\n')
                comparisons.append((workload,index,label,result,output))
    for note in cpu_notes:lines += ['',note]
    lines += ['', '输出序列逐请求比较如下。“不同”表示至少一个输出 token 不同，不直接判定哪一份输出正确。', '', '| bundle | 补充轮次 | 比较对象 | 输出不同的请求数 / 全部请求 |', '| --- | ---: | --- | ---: |']
    for workload,index,label,result,output in comparisons:
        name='同节点关闭 offload 的 rep0' if label=='same-node-off' else '原节点开启 offload 的 rep0'
        lines.append(f'| {workload.upper()} | {index} | [{name}](../{output.relative_to(ROOT)}) | {result["different_outputs"]} / {result["requests"]} |')
    lines += ['', '同节点比较去掉了节点变化这一项，但开启 connector 仍会改变前缀查找和执行路径。输出对比不能替代各缓存组的数值状态校验。原实验的 CPU 池还保留了更早请求的内容，新服务只经历本轮 C2 预热，因此 CPU 淘汰数量需要连同被淘汰内容的来源一起解释。']
    audit=ROOT/'output/kv-causal-validation-20260911/c2-hybrid-prefix-same-node-audit.json'
    if audit.exists():
        records=json.loads(audit.read_text())['records']
        if len(records)==3 and all(r['mamba_blocks_at_previous_state_column']==[0,0,0] for r in records):
            lines += ['', '新节点的 C2 再次出现同样三个样本：GPU 本地分配长度分别为 784、2352、2352，外部恢复为 0；按源码推导的上一 Mamba 状态列，在三个组中仍全部指向空占位 block 0。疑点已在补充运行中复现，数值状态验证仍未完成。参见 [同节点逐条记录](../output/kv-causal-validation-20260911/c2-hybrid-prefix-same-node-audit.json)。']
    for workload in ['c2-12','c8']:
        lines += ['', report_link(ROOT/f'replay/kv_same_node_physical_{workload}_20260911/gpu274_cpu32/rep0',workload.upper()+' 补充对照与逐轮证据','config')]
    return '\n'.join(lines)


if __name__=='__main__':main()
