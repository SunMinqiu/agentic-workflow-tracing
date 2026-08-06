# Local-vLLM KV-Cache Sweep: Code Change Plan

Revision 2, 2026-08-04. Rewritten after review. Revision 1 was rejected on method: it re-ran a live workflow per arm, which compares agent trajectories rather than cache configurations. Capture-and-replay is now the spine of the plan, not an afterthought.

Target: a self-hosted vLLM under `podman-hpc` on the HPC node. Unblocks Phase 2 groups A, B, and C from [KVCache-Plan-What-and-Why.md](KVCache-Plan-What-and-Why.md).

---

## 1. What is shared, and what is not

A cache arm is a property of the **environment**, not of the **workflow**. That rule survives review and still decides where code goes. The test for whether new code sits in the right place: **would it have to be written again for the next workflow?**

But the shared layer is **capture, arm manifest, replay, and result schema** — not a common client. GenoMAS reaches the model through the OpenAI SDK, SciLink through litellm. Both already work against an OpenAI-compatible endpoint, so both already work against vLLM. Forcing them onto one gateway would buy nothing and cost two integrations. The common substrate is the pair of files the loggers already emit, `messages.jsonl` and `pi_events.jsonl`; replay consumes that schema, not the client.

| Layer | Knows about arms | Knows about a workflow |
| --- | --- | --- |
| `serving/` (new) | Yes, entirely | No |
| `replay/` (new) | Consumes the arm stamp | No — reads the common trace schema |
| `adapters/llm_trace.py` | Only that a stamp exists | No |
| `adapters/<wf>/logger.py` | No | Yes — the only per-workflow code |
| `scripts/lib_results.sh` | Cell-naming convention only | No |
| `analysis/kvcache/*` | Reads the stamp | No |

1000genome and Montage have no LLM and stay out of the sweep entirely.

---

## 2. The spine: capture, replay, live

Three stages, in order. Skipping to stage 3 is what revision 1 did wrong.

**Stage 1, capture.** Run each workflow once, live, and record a replay bundle: the full request sequence with token IDs, generation parameters, per-request arrival timestamps, the concurrency structure (which requests were in flight together), and the recorded responses. Token IDs matter — replaying IDs through the completions endpoint rather than text through the chat endpoint removes chat-template variance from every downstream comparison, so an arm difference cannot be a templating difference in disguise.

**Stage 2, replay.** Fire the identical bundle at each arm. This is where groups A, B, and C are actually measured. Inputs are byte-identical across arms by construction, so any difference in TTFT, hit rate, or I/O is attributable to the arm.

**Stage 3, live validation.** Re-run the live workflow at a small number of arms — baseline, the best predicted arm, and one arm expected to be bad — and report end-to-end effect. Live runs are expensive and irreproducible, so they validate a conclusion rather than search for one.

### Replay has a time problem; decide it explicitly

H1 claims retention binds before capacity, and its evidence is that minutes of non-LLM computation sit between consecutive calls. That gap structure is a property of the **live** run. Replaying back-to-back to save wall clock destroys exactly the thing H1 studies.

So replay has two modes, and each arm declares which it needs:

- `paced` — preserve original inter-arrival gaps and in-flight overlap. Required for every retention or flush arm.
- `packed` — back-to-back, preserving only sequence and concurrency width. Valid for pure mechanism and pure capacity arms, where only the reference sequence matters.

Never compare a `paced` cell against a `packed` one.

Paced replay costs the original wall clock, which at current run sizes is cheap: the cells in `phase4_20260729_124254` span 4 to 9 minutes each, so three flush intervals across two cells is under an hour. Re-check this before scaling to `fullpipeline` or `A_c8_w4`, where the cost grows with the run.

### What replay must reproduce, and what it must not

Reproduce: the prompt token IDs, the request order, the arrival timing under `paced`, the concurrency width, and the sampling parameters.

Do not reproduce: the generated content. Pin `max_tokens` to the recorded output length so decode cost stays comparable across arms, and keep the generated text for comparison.

Expect some output divergence even between arms that are supposed to be numerically equivalent. A prefix-cache hit changes how the prefill is chunked and batched, which changes floating-point reduction order, which occasionally changes a token. So a small divergence between `A0` and `A1` is not automatically a bug, and equally it is not automatically fine — measure the baseline divergence rate between two runs of the *same* arm first, and treat that as the noise floor any later quality claim has to clear.

---

## 3. Analysis fixes: measure the right quantity

### 3.1 Reuse distance, not working set

Revision 1 claimed trie-node creations equal the ideal cache's working set. That is wrong. A trie node is a distinct prefix position; a vLLM cache entry is a block of `block_size` tokens keyed by its prefix hash. The counts are related but not equal, and more importantly **neither one predicts the capacity bend point**.

The quantity that does is the **reuse-distance distribution**, measured in unique blocks. For each block reference, the stack distance is the number of *distinct* blocks referenced since that block was last touched. Under LRU with capacity C blocks, a reference hits if and only if its stack distance is below C. So a single pass over the trace yields a stack-distance histogram, and that histogram yields the miss-ratio curve **for every capacity at once**.

Multiply by bytes per block to put the curve in the same units as the capacity knob:

```
bytes_per_block = block_size × 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

Every term comes from the pinned model manifest in section 8, so the conversion is exact rather than estimated.

This changes the capacity experiment from a blind five-point sweep into a prediction validated at three or four points chosen around the predicted bend. Both outcomes are publishable: if measurement tracks prediction, report the full curve; if it does not, the deviation is a finding about vLLM's eviction, which is not pure LRU — it will not evict blocks referenced by running requests, and blocks form a prefix tree whose eviction order favours leaves.

Emit three quantities per cell, replacing the single bad one:

| Quantity | Definition | Used for |
| --- | --- | --- |
| `cumulative_unique_blocks` | Distinct aligned cacheable blocks over the whole trace | Upper bound on demand; the denominator for capacity multiples |
| `reuse_distance_blocks` | Stack-distance histogram over block references | Predicts the miss-ratio curve |
| `peak_resident_blocks` | Peak simultaneous residency under a stated retention rule | Only meaningful with a policy attached; unbounded LRU makes it equal to cumulative |

### 3.2 Alignment: record both units

[`logical.py:52`](../src/agent_io_tracing/analysis/kvcache/logical.py#L52) hard-codes `OPENAI_BLOCK = 128`. Two separate problems.

First, the constant must come from the arm rather than a module literal. Thread `block_size` through `analyze_cell_logical`, rename `logical_128_frac` to `logical_aligned_frac`, and keep the old key as an alias for one release so `results/**/kvcache_logical.json` stays readable.

Second, and less obvious: the physical KV block size and the granularity at which a prefix hit is decided are not necessarily the same number in current vLLM. Logical hit rate must be aligned to the **prefix-matching unit**, not the physical block size. The arm manifest records both, and the analysis takes the matching unit. Which flag reports it is version-dependent and gets pinned in section 8 — do not take a flag name in this document on faith.

### 3.3 Tokenizer

[`logical.py:55-60`](../src/agent_io_tracing/analysis/kvcache/logical.py#L55-L60) falls back to tiktoken `o200k_base` for anything tiktoken does not recognize, so serving Qwen or Llama computes every logical number in the wrong tokenizer. `logical%` is the plan's headline backend-independent quantity; computed in a tokenizer the backend does not use, it is not backend-independent.

Resolve per model: vLLM's `/tokenize` endpoint when a local server is configured, then a local HuggingFace tokenizer, then tiktoken. Keep the resolved name in `summary["tokenizer"]` so a mixed-tokenizer comparison is visible rather than silent.

Capture stores token IDs anyway (section 2), which makes this mostly moot for replayed cells — but the live cells and every existing trace still need it.

---

## 4. `serving/` — remote endpoint control

The HPC node already owns the `podman-hpc` lifecycle. Local code does not launch, stop, or reconfigure the container. It connects to the active endpoint and records which arm the operator started.

**`vllm_endpoint.py`** — client for the already-running server. It checks `/health` and `/v1/models`, reads `/metrics`, and exposes an explicit `reset_prefix_cache()` operation. It never starts or stops the HPC container.

**`arms.py`** — the arm table as data: name, knob dict, replay mode, and which hypothesis it probes. The only file that changes when an arm is added.

**`metrics.py`** — parse `/metrics` into a flat dict; sample on a timer, not once per cell boundary.

### One local run targets one active HPC arm

The local driver does not assume it can change server flags. Before a run, it receives an arm manifest containing the image digest, vLLM version, model, and resolved cache geometry. It stamps that manifest onto every call and checks that `/v1/models` matches. Changing an arm means restarting vLLM on the HPC node, then starting another local run with the new manifest.

Capacity arms remain normalized per replay bundle because `1×` differs between GenoMAS and SciLink. The HPC launch command therefore uses the capacity selected for that bundle. Mechanism arms with identical absolute configuration can reuse one active server across multiple replay bundles.

Between cells sharing one server, call `/reset_prefix_cache`, or cell N inherits cell N−1's warm cache.

---

## 5. Metrics: what each source can and cannot say

Revision 1 over-claimed here. Three sources, non-interchangeable.

| Source | Grain | Valid use | Invalid use |
| --- | --- | --- | --- |
| Response `usage.prompt_tokens_details.cached_tokens` | Per request | Realized reuse per call; the existing join key | — |
| `/metrics` prefix-cache queries and hits | Aggregate counters | Cross-check total reuse per isolated cell | Per-call attribution; direct equality with `cached_tokens`, whose unit and semantics differ |
| `/metrics` KV usage gauge | Instantaneous | **KV block occupancy** | GPU memory footprint — it is a fraction of allocated blocks, not bytes of HBM |
| `/metrics` TTFT histogram | Aggregate distribution | Distribution-level cross-check of client timing | Any per-call TTFT; client-side stream timing stays authoritative |
| Node GPU telemetry | Sampled | Actual HBM footprint | — |

Sample periodically. Occupancy, queue depth, and eviction pressure are time-varying, and the interesting dynamics happen mid-cell, not at its boundaries.

Record alongside every sample: configured KV bytes, requests running, requests waiting, and **preemption count**.

### Preemption is a validity gate, not a monitoring nicety

Squeeze KV capacity far enough and vLLM preempts and recomputes running requests. TTFT then moves for scheduling reasons that have nothing to do with prefix hit rate, and the low-capacity end of the curve stops meaning what the axis label claims.

Rule: any capacity point with nonzero preemption is either flagged in the figure or dropped from the curve fit. Without this the bend point is unfalsifiable.

---

## 6. Server-side eBPF for the tier arms

Revision 1 assumed the existing tracer would see KV traffic. It will not. The tracer wraps the workflow client; vLLM runs in a container, possibly on a different node. Client-side tracing sees none of the server's file I/O.

Phase 6 therefore needs a second collection point:

- Run the tracer on the node hosting vLLM, resolving PIDs through the `podman-hpc` container's namespace to the actual worker processes.
- Record the KV offload paths, backing filesystem, read and write bytes, latency, and thread.
- Align server-side events to client-side cells **by request id**, not by clock. Cross-node sub-second clock alignment is fragile; [`load_joined_calls`](../src/agent_io_tracing/analysis/kvcache/logical.py#L91) already stores `provider_request_id` per call, and vLLM logs the same id. The join already exists.

The offload connector, its target directory, its capacity, and a positive control proving events actually appear must all be verified before the tier arms are treated as measurements. "The eBPF tooling is the thing other groups cannot do" is only true if it is pointed at the right node.

---

## 7. Arm table

Every flag here is version-dependent. Pin the image first (section 8), then confirm each flag against that image. Do not trust this table.

| Arm | Knobs | Replay mode | What it prices |
| --- | --- | --- | --- |
| `A0_nocache_engine` | prefix caching disabled | packed | True rung-0 floor |
| `A0_nocache_tag` | tag-based cache busting | packed | The hosted-vendor floor, kept for comparability with existing OpenAI and FreeInference cells |
| `A1_prefix` | defaults, caching on | packed | Rung 1, the real baseline |
| `A2_cpu_offload` | KV offload to host DRAM | packed | Whether spilling to host memory buys back hit rate |
| `A3_fs_tier` | KV offload to NVMe, then Lustre | packed | **H5**, and the arm that needs section 6 |
| `B_cap_*` | KV cache memory bytes, set from the predicted curve | packed | Validates the miss-ratio curve from 3.1 |
| `B_align_*` | block size and matching unit | packed | Alignment granularity |
| `C_flush_*` | timed global prefix-cache reset, swept downward | **paced** | Locates the retention threshold this workload needs |

Four notes.

**Capacity knob.** Use the explicit KV-cache-bytes setting, not the GPU-block-count override — the latter is documented as a preemption-testing aid, and bytes map directly onto the curve from section 3.1.

**`C_flush_*` is not a TTL, and its purpose has changed.** A TTL expires entries individually by age; a periodic global reset invalidates everything at once. The name in code is `global_flush_interval`, and it must never be written up as if vLLM had a TTL.

The reason to run it is no longer "confirm that retention binds", because the existing traces say it does not at these timescales. Prefix ages reach 248s in `A_c2_w2` and 127s in `A_c4_w2`, so the long gaps H1 describes are real — but token capture does not decay across them. `A_c4_w2` captures 82%, 77%, 80%, 95%, 87% across the five age bins, and `A_c2_w2` captures 93%, 96%, 95%, 44%, 82%, where the 44% bin holds four calls and 4,992 tokens and is noise. Both vendors retain entries past four minutes without visible loss.

So sweep the interval **downward** — 60s, 30s, 15s — and report the threshold at which this workload's gap structure starts losing hits. The output is "this workload needs retention of at least X seconds", which transfers to any serving system, rather than "the vendor we happened to use was good enough".

**FP8 is deferred out of the first pass**, by decision, not oversight. KV quantization changes numerics, so the main plan requires a quality axis alongside it. Building that axis is real work — exact token match rate, semantic equivalence on divergent cases, then task-level correctness on live runs — and it should not block the capacity curve. Replay makes it tractable when its turn comes, since inputs are identical across arms by construction, so revisit after step 5. Reporting an FP8 speedup without the quality axis is a number that cannot recommend anything, which is why the arm is out rather than in-but-unmeasured.

**Group D** (replicas, cache-aware routing) stays out of the first pass. The existing `A_c4_w2` and `A_c4_w4` cells already generate concurrent in-flight requests against one cache, which is the contention half of the question.

---

## 8. Phasing

1. **Pin the environment.** Image digest, vLLM version, model revision, chat template, tokenizer, tensor-parallel degree, physical block size, prefix-matching unit, and the actual KV bytes the server allocates. Everything downstream is expressed relative to these; without them no result is reproducible and the bytes-per-block conversion in 3.1 cannot be computed.
2. **Build capture and replay.** GenoMAS and SciLink are the first two adapters, not the design centre. Output is a replay bundle in the common schema.
3. **Fix the analysis** — block-aligned reuse, reuse-distance histogram, working-set curve, tokenizer resolution. Runs on existing traces, so it proceeds in parallel with step 2 and produces the predicted capacity curve that tells step 5 which points to measure.
4. **A0 versus A1 under replay.** Validate per-request `cached_tokens` against the server's aggregate counters in an isolated cell. If they disagree, stop and resolve before sweeping anything.
5. **Capacity curve** via the KV-bytes knob, at points chosen from step 3's prediction, with the preemption gate from section 5 enforced.
6. **CPU and filesystem tiers**, with eBPF running on the vLLM node per section 6.
7. **Live workflow runs** at a few selected arms; report performance, I/O, and quality together.

---

## 9. Day-0 verification

Run against the served endpoint before building anything. Substitute host and port.

The local workflow environment can be configured without LiteLLM as a gateway:

```zsh
export VLLM_URL=http://hpc-node:8000
export VLLM_SERVED_MODEL=Qwen3.6-27B
source config/config_vllm_endpoint.env
agent-trace-vllm-check
```

GenoMAS then uses the served model name directly. SciLink keeps its existing LiteLLM client and uses the same endpoint through the `openai/` model prefix. `local-vllm` is only a nonempty SDK placeholder when the server has no authentication; set `VLLM_API_KEY` when vLLM was launched with `--api-key`.

```zsh
curl -s "$VLLM_URL/v1/models" | python3 -m json.tool
```

```zsh
curl -s "$VLLM_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$VLLM_MODEL"'","messages":[{"role":"user","content":"hello"}],"max_tokens":8}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
```

```zsh
curl -s "$VLLM_URL/metrics" | grep -E 'prefix_cache|cache_usage|preempt|time_to_first_token|num_requests'
```

```zsh
curl -s -X POST "$VLLM_URL/reset_prefix_cache"
```

The second command is the one that can invalidate the design. It decides whether `prompt_tokens_details.cached_tokens` is populated per request, which the entire per-call join in `load_joined_calls` depends on. If it is absent, realized reuse degrades to a per-cell aggregate from server counters and the join has to change — find out before writing the replay layer, not after.

Run it twice with an identical long prompt: the second call should report nonzero `cached_tokens`. If not, prefix caching is off or the prompt is under the minimum length, and that blocks step 4.

Also confirm from inside the container that the token-IDs completion path accepts a prompt of integers, since stage-1 capture is built on it.
