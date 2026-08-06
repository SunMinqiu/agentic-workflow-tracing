# Phase 1: Logical KV-Cache Demand of Agentic Workflows

## Implementation status

Audited against the repository on 2026-08-04. Status reflects executable code and tests, not statements elsewhere in this plan. Completed items have been removed; what remains is open work.

Legend: 🟡 partially implemented or awaiting real instrumentation · ⬜ not implemented.

### Phase 1 instrumentation and analysis

| Item | Status | Current implementation or missing work |
| --- | --- | --- |
| Q4 source decomposition | ⬜ | Messages are persisted, but reusable tokens are not attributed to system instructions, conversation history, tool output, handoff, retry, or data-derived spans. Phase labels are heuristic and are not token-source accounting. |
| Q6 workflow ranking | 🟡 | `build_report` sorts cells by total input volume and displays realized, logical, and gap percentages, but there is no explicit combined ranking score for high volume and low realized capture. |
| TTFT and TPOT analysis | 🟡 | Done for GenoMAS: `_install_openai_stream_timing` replaces the OpenAI-compatible call with a `stream_options.include_usage` stream, so first-token and last-token timestamps are real, and three cells in `phase4_20260729_124254` carry measured TTFT and TPOT. The two remaining cells in that run were traced before the patch and print the unavailable notice. SciLink reads `completion_start_time` from litellm and is covered by a test, but no SciLink run with `stream=True` exists yet, so its reports still print the unavailable notice. |

### Phase 2 experiment infrastructure

| Planned factor or experiment | Status | Current implementation or missing work |
| --- | --- | --- |
| A. Rung 0 to rung 1, no cache versus exact prefix caching | 🟡 | Runnable through vendor APIs. `apply_nocache_tag` in `adapters/llm_trace.py` prepends a per-call random tag so no provider prefix cache can hit, `GENOMAS_NOCACHE` and `SCILINK_NOCACHE` select the arm, and the untagged prompt is what gets logged so logical reuse stays comparable across arms. `phase4_20260729_124254` holds matched cached and no-cache cells on both OpenAI and FreeInference; the no-cache cells score realized 0% as designed. Missing: pairing arms at equal call counts, since the cells currently differ in length. |
| A. Multi-tier offload, approximate reuse, compression, and eviction ladder | ⬜ | Rungs above exact prefix caching still need a local serving stack. None is connected. |
| B. Capacity, TTL, eviction policy, and block-size sweeps | ⬜ | Prefix interval and a fixed 128-token alignment are analyzed offline, but no cache simulator or controlled sweep exists. |
| C. HBM, DRAM, NVMe, and shared-storage tiers | ⬜ | eBPF traces general file I/O, but no KV-cache tier is deployed or identified in the trace. |
| D. Workflow concurrency | 🟡 | GenoMAS includes matched `A_c4_w2` and `A_c4_w4` cells, but this changes agent workers rather than serving replicas or cache routing. |
| D. Replica count and cache-aware routing | ⬜ | No multi-replica serving harness or routing policy exists. |
| E. Context-construction variants | ⬜ | No controlled history, placement, normalization, retry, or summary-granularity variants are implemented. |
| Capability ladder experiment | ⬜ | No runnable line A sweep exists. |
| Sensitivity curve experiment | ⬜ | No capacity or TTL sweep exists. |
| Cross-layer attribution experiment | ⬜ | Context construction has not been crossed with cache mechanism or storage tier. |

### Phase 2 response variables

| Response | Status | Current implementation or missing work |
| --- | --- | --- |
| TTFT | 🟡 | Real for GenoMAS. `plot_ttft_vs_fresh_input`, `plot_latency_breakdown`, and `plot_ttft_vs_prefix_age` are emitted whenever `has_stream_timing` holds, and three cells in `phase4_20260729_124254` report median TTFT of 0.71s to 1.66s with median TPOT of 4.6ms to 29ms per token. Those three are all no-cache cells; the cached cells predate the patch and must be re-run before a cached-versus-uncached TTFT delta exists. SciLink needs one streaming run. |
| Prefill compute saved | 🟡 | Reused and fresh token counts provide a token proxy, but no model-side prefill compute counters are collected. |
| KV-related I/O bytes and bandwidth | ⬜ | General eBPF I/O exists, but KV objects, cache loads, and cache writes are not labeled or separated. |
| GPU memory footprint | ⬜ | No GPU memory sampler is connected. |
| Output quality | ⬜ | No task-quality or answer-equivalence evaluation is tied to cache configurations. |

### Hypothesis readiness

| Hypothesis | Status | What can be tested now and what remains |
| --- | --- | --- |
| H1 retention binds before capacity | 🟡, and currently **not supported** | Measured, not just measurable. `_temporal_metrics` bins every call by age since the latest compatible prefix and reports capture rate per bin. The measurement contradicts half the hypothesis: the long gaps are real, up to 248s, but capture does not decay across them on either vendor. See Early findings. What remains is to sweep flush interval downward on a local stack to find where retention *starts* to bind. |
| H2 multi-agent prompt assembly discards sharing | 🟡 | `cross_agent_bonus_frac` is exactly 0.0 in all five cells of `phase4_20260729_124254`, spanning two vendors, two models, and two cohort counts. The zero is no longer a single-run observation. Prompt-assembly normalization experiments that would show the sharing is recoverable are still missing. |
| H3 avoidable construction choices cause misses | ⬜ | No construction-choice sweep or token-source attribution exists. |
| H4 mid-context insertion limits exact prefix reuse | ⬜ | Exact-prefix opportunity is measured, but content-level reuse and insertion-position metrics are not. |
| H5 cache capacity turns compute savings into I/O | ⬜ | No tiered KV cache, KV-specific I/O attribution, or TTFT-versus-tier experiment exists. |

### Group E and cross-layer metrics

| Planned item | Status | Current implementation or missing work |
| --- | --- | --- |
| Tool-output-to-prompt mapping | ⬜ | Tool executions and prompts are logged separately, but output spans are not linked to the prompt tokens they create. |
| Conversion-layer tokens per data read | ⬜ | No file-byte or tool-output span is mapped to tokens. |
| Submission-layer repetitions of data-derived spans | ⬜ | Prefix matching is global text matching and does not identify data provenance. |
| Data amplification factor across all three layers | ⬜ | The disk denominator exists, but conversion and submission numerators are missing. |
| Data-summary granularity sweep | ⬜ | Not implemented. |
| Summary-determinism normalization sweep | ⬜ | Not implemented. |
| History-retention sweep | ⬜ | Not implemented. |
| Tool-output-placement sweep | ⬜ | Not implemented. |
| Retry and failure-handling sweep | ⬜ | Failures and heuristic phases are logged, but no controlled retry-policy variants or retry-token accounting exist. |
| Dataset-size sweep | 🟡 | GenoMAS has cohort and trait-count cells, and traditional workflows have scale cells, but there is no analysis that holds workflow logic fixed and attributes prompt length, call count, retries, and reuse directly to dataset bytes. |

### Prerequisites

| Prerequisite | Status | Current implementation or missing work |
| --- | --- | --- |
| Deterministic response replay | 🟡 | GenoMAS can replay exact cached responses from `llm_cache.jsonl` and fail on strict misses. SciLink has no equivalent. |
| Replay prompts against local vLLM or SGLang | ⬜ | No local serving launcher, model fixture, or cache-parameter interface exists. |
| Traditional workflow comparison | 🟡 | 1000 Genomes and Montage share reusable I/O tracing infrastructure, but they have no LLM context and therefore contribute only the storage-side comparison. |

---

## What we measure


| Question                                            | Metric                                            | Why it matters                                                     |
| --------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| **Q1 How long is each call's prompt?**              | per-call input tokens                             | Prompt tokens are the KV-cache unit; sets the scale of one context |
| **Q2 How does context grow across the workflow?**   | input tokens over call order, per-role slope      | Shows whether we cache a fixed prefix or a growing history         |
| **Q3 How much prefix is reused between calls?**     | backend cached_tokens; raw-message prefix overlap | Quantifies how much we can save                                    |
| **Q4 Where does the context come from?**            | tokens from tool output, history, handoff, retry  | Locates repeated context                                           |
| **Q5 How much context is re-submitted over a run?** | Σ cached_tokens; verbatim re-submissions          | Global waste                                                       |
| **Q6 Which workflow needs prefix caching most?**    | rank by input-token volume and low realized reuse | Picks the highest                                                  |


---



## Early findings

**GenoMAS** — the main target, a gene-expression analysis agent

- Per-call prompt stays at 10k–13k tokens, peaks at 20k–30k; scale comes from more calls, not a bigger single prompt → cache the fixed prefix that recurs across calls.

- Logical reuse sits at 60% to 67% of input tokens across every cell measured so far, and it barely moves between vendors, models, or cohort counts. That stability is the argument that logical reuse is a workload property while `cached_tokens` is a vendor artifact.
- The realized-versus-logical gap is where vendors differ: 4% on OpenAI `gpt-4o-mini`, 11% on FreeInference `qwen3.6-35b`, for the same workflow at similar scale. Same demand, different capture.
- Reuse gaps are long but retention is not costing us anything yet. The age of the latest compatible prefix reaches 248s in `A_c2_w2` and 127s in `A_c4_w2`, confirming that minutes of non-LLM computation do sit between reusable calls. Capture rate across the five age bins is nonetheless flat — 82%, 77%, 80%, 95%, 87% on FreeInference and 93%, 96%, 95%, 44%, 82% on OpenAI, where the 44% bin holds four calls and 4,992 tokens and is noise. This is a negative result for H1 as originally stated and it reframes the retention question as a threshold to locate rather than a bottleneck to demonstrate.

**SciLink** — a reference point, not the target

- On an OpenAI backend with automatic prefix caching, ~18% of prompt tokens are reclaimed with zero changes.
- The later 4-cell run inverts the sign: realized 5% to 12% against logical 1% to 2%, a negative gap. SciLink's prompts carry large images and long single-shot contexts, so the vendor hits on structure that the token-level prefix matcher does not count as reusable. Worth explaining before quoting either number, because a negative gap means the two metrics are measuring different things rather than one bounding the other.

---



## Remaining priorities

1. Re-run the cached arm of `phase4` under the current logger so both arms carry stream timing, then fit the TTFT slope per 1k uncached tokens. Cheapest remaining item and it unblocks every seconds-denominated result.
2. Run SciLink once with `stream=True` to populate its TTFT and TPOT, and explain its negative realized-minus-logical gap.
3. Implement Q4 token-source decomposition and tool-output-to-prompt provenance.
4. Add deterministic SciLink replay and local vLLM or SGLang prompt replay.
5. Build the controlled Phase 2 cache mechanism, retention, capacity, tier, routing, and context-construction sweeps.
6. Add KV-specific I/O attribution, GPU memory sampling, and output-quality evaluation.

---

# Phase 2: Comparing Cache Methods, Parameters, and Environments

## Framing

Cache configurations are probes, not systems under evaluation. Each configuration change produces a delta, and that delta prices one structural property of the workload. A factor earns a place in the matrix only if it can falsify a stated hypothesis about agentic scientific workflows.

`cached_tokens` is not a workload property. It is what one backend happened to hit under one set of parameters, subject to that vendor's minimum-length threshold, alignment granularity, and TTL. The backend-independent quantities — how much context a run logically re-submits, in what structure, at what interval — are the object of study. The gap between logical reuse and realized reuse is the reportable number, because it names both the opportunity and the technology that failed to capture it.

## Factors

| Group | Levels | Hypothesis it addresses |
| --- | --- | --- |
| **A. Reuse mechanism** | exact prefix caching → prefix + multi-tier offload → non-prefix approximate reuse → plus KV compression/eviction | Which rung must a workflow reach before reuse pays off |
| **B. Capacity and retention** | cache size as a multiple of working set (0.25× / 0.5× / 1× / 2× / 4×), TTL, eviction policy, block size | Where the response curve bends |
| **C. Storage tier** | HBM only → +CPU DRAM → +local NVMe → +shared storage | Whether cache misses convert into I/O bottlenecks |
| **D. Concurrency and routing** | replica count, random vs cache-aware routing, fan-out width, neighbor contention | Whether structural sharing survives deployment |
| **E. Context construction** | history policy, tool-output placement, summary determinism, retry policy, data-summary granularity | Whether the framework or the serving layer holds the larger lever |

Attention architecture (MHA/GQA/MLA) and KV quantization width do not need their own dimension. Both change exactly one quantity — bytes of KV per token — which folds into the working-set multiple in group B.

Express capacity as a multiple of the run's working set rather than in absolute GB, so the bend point transfers across models and hardware. Normalize latency across hardware as TTFT slope per 1k uncached tokens.

## What the current vendor APIs actually expose

Groups A through D are not reachable through OpenAI or FreeInference. Both are hosted OpenAI-compatible endpoints that enable prefix caching automatically and expose no capacity, TTL, tier, eviction, or routing controls. What remains adjustable through the API is the request-side surface that decides whether a prefix qualifies for a hit.

| Knob | OpenAI | FreeInference | What it lets us vary |
| --- | --- | --- | --- |
| Cache enable/disable | No API flag, but defeatable request-side | Same | Implemented. A per-call random tag in front of the first token makes every prefix unique, which forces a 0% hit rate on any prefix cache without a local serving stack. Verified: every `_nocache` cell reports realized 0%. |
| `prompt_cache_key` | Available; groups requests so they land on the same cache | Unverified; OpenAI-compatible endpoints usually ignore unknown fields | Approximates group D routing without owning the router |
| Minimum cacheable prefix and alignment | ~1024-token minimum, 128-token increments | Backend-defined, likely vLLM or SGLang block size | Sets the alignment constant used in `logical.py` |
| Retention | Vendor-controlled, minutes of inactivity | Undocumented | Nothing directly; only observable through prefix-interval bins |
| Prompt structure | Fully controllable | Fully controllable | All of group E |
| Model choice | Changes KV bytes per token and cache eligibility | Changes backend entirely | Confounds vendor comparison; never mix vendors in one cell |
| `cached_tokens` readout | `usage.prompt_tokens_details.cached_tokens` | Verified present | Realized reuse only |

The consequence for the plan: with vendor APIs alone, group E is a real experiment, the rung 0 to rung 1 step of group A is now a real experiment through the no-cache tag, and the retention question in H1 is observable through the prefix-age capture table but not sweepable. The higher rungs of A and all of B, C, and D require the local vLLM or SGLang replay harness listed under prerequisites. Confirm `prompt_cache_key` support on FreeInference empirically before relying on it, and verify the current OpenAI minimum and increment against live documentation rather than this table.

## Baseline positioning

A no-cache configuration is an instrument, not a comparison arm. Every serving stack has enabled prefix caching by default since 2023, so reporting a speedup against no-cache is a straw man. It is retained for two things nothing else provides: the denominator for logical reuse, and the TTFT-per-1k-uncached-token slope that converts every token-denominated result into seconds. It belongs in the methodology section; the comparison arms start at exact prefix caching.

The arm is now built and run. `phase4_20260729_124254` pairs cached and no-cache cells on both vendors, and the no-cache cells land at realized 0% against logical 61% to 67%, which is the intended behaviour of the tag: the prompt still logically repeats itself, and the cache is simply forbidden to notice. The matched cached cells realize 48% on FreeInference and 55% on OpenAI. The gap between logical and realized reuse is now bracketed by a measured floor rather than inferred from a single arm.

The seconds-per-uncached-token conversion is not yet fittable from this run. Only the three no-cache cells carry stream timing, because the two cached cells were traced before the streaming patch landed. Re-running the cached arm under the current logger is the smallest step that unlocks a paired TTFT delta.

## Experiment lines

Not a full factorial. Three lines.

| Line | Sweep | Output |
| --- | --- | --- |
| Capability ladder | Vary A, defaults elsewhere | Which rung agentic workflows require; where marginal gain vanishes |
| Sensitivity curve | Fix A at exact prefix caching, sweep B capacity and TTL | Bend point — the minimum capacity and retention the workload demands |
| Cross-layer attribution | E crossed with A or C | Whether changing context construction beats changing serving technology |

The third line is the one that yields an actionable conclusion. If a one-line change in prompt assembly outperforms an entire serving-layer upgrade, the characterization points directly at how agent frameworks should be written.

## Response variables

Hit rate in tokens, TTFT, end-to-end wall clock, prefill compute saved, KV-related I/O bytes and bandwidth via eBPF, GPU memory footprint, and output quality. Quality is mandatory once the sweep reaches non-prefix approximate reuse or KV quantization, since those rungs trade accuracy for hit rate and a comparison without a quality axis is meaningless.

## Hypotheses

**H1 — Retention, not capacity, is the binding constraint.** Stated premise: reuse distance has a long tail exceeding commercial cache TTLs, because minutes of non-LLM computation sit between consecutive calls.

The premise holds and the conclusion does not, on the evidence so far. The tail is real — prefix ages reach 248s — but capture rate is flat across every age bin on both OpenAI and FreeInference, so neither vendor's retention is losing us anything at this run length. Restate the hypothesis as a threshold question rather than a claim: **how much retention does this workload require?** Sweep a global flush interval downward on the local stack until capture degrades, and report the interval at which it does. That number transfers to any serving system; "the vendor we used was good enough" does not.

**H2 — Multi-agent systems discard structural sharing at assembly time.** `x-agent% = 0` in every GenoMAS cell measured so far, across both vendors and both cohort counts: GEOAgent, CodeReviewerAgent, and DomainExpertAgent are prefix-isolated, and all reuse is intra-agent history accumulation. These agents share task background and data schema and should share prefix; the framework gives each its own assembled system prompt and throws that away. The zero is now robust enough to build on, so the open question moves from "is it real" to "is it recoverable": normalize the assembled system prompts so the shared background sits at the head of every agent's prefix, and measure how much of the 0 turns into a hit. That result indicts framework design, which is stronger than showing cache-aware routing helps.

**H3 — Most misses trace to avoidable construction choices.** Sweep E against the A ladder and compare the two effect sizes.

**H4 — Mid-context insertion caps exact prefix caching early.** Tool outputs and interleaved agent messages land inside the history rather than at its tail, so content-level reuse outruns prefix-level reuse. Read the jump from rung 2 to rung 4 of the ladder.

**H5 — Beyond a working-set threshold the cache stops saving compute and starts buying I/O.** Sweep C and check whether the TTFT reduction is consumed by KV load waits. This is the hypothesis to prioritize: it sits at the intersection of KV-cache research and the eBPF tooling, and it is the measurement other groups cannot make.

## Group E in detail: context construction in scientific workflows

Part of the prompt in a scientific workflow is not written by a human — it is derived from data on disk. GEOAgent reads an expression matrix and injects column names, sample counts, head rows, and summary statistics. Once that link exists, I/O and KV cache are two segments of one data path rather than two topics. The mapping from tool output to prompt tokens is not currently recorded and is the first field to add.

### Data amplification factor

How many times one byte of raw data is ultimately submitted to the LLM over a run, measured at three layers.

| Layer | Quantity | Instrument |
| --- | --- | --- |
| Disk | Opens, reads, and bytes for a file within one run | eBPF, already available |
| Conversion | Tokens each read contributes to a prompt | New: tool-output-to-prompt mapping |
| Submission | Occurrences of the same data-derived span across all prompts | Prompt sequence matching |

The factor spans storage and inference, which is why LLM-systems work does not measure it and I/O work does not think to. A high factor means the same data is both re-read from disk and re-submitted in prompts, two layers of waste compounding. The mitigation is likewise cross-layer: caching the summary eliminates both the disk read and the prefill.

### Factors to sweep

**(a) Data summary granularity** — full table vs head-N rows vs schema and statistics only vs a file path the agent may choose to read. The last level is the important one: it converts prompt tokens into a file read, an explicit token-for-I/O exchange that is natural here because the data already lives on disk. The exchange rate is directly measurable — X tokens saved against Y bytes read and Z milliseconds of wait — and which side wins depends on data size and cache state.

**(b) Summary determinism** — absolute paths, timestamps, randomly sampled rows, and floating-point jitter in tool output break byte-exact prefixes. Run once with raw tool output and once with output normalized for sampling, timestamps, and paths. A large difference means a substantial share of prefix invalidation is an engineering artifact rather than intrinsic to agentic workloads, which is a valuable negative result.

**(c) History retention policy** — full history vs sliding window vs summary compaction vs explicit state object. Compaction rewrites history, so every compaction is a global cache invalidation and appears as a cliff on the time axis. The tokens compaction saves and the hits it destroys are both measurable, and on long workflows the balance may run counter to intuition.

**(d) Tool-output placement** — appended at the tail vs inlined into history vs hoisted into the system prompt. This sets the mid-context insertion ratio and therefore the ceiling on exact prefix caching.

**(e) Retry and failure handling** — scientific tools fail far more often than conversational ones: malformed data, missing dependencies, script errors, download timeouts. Appending the error preserves the prefix; rebuilding the context destroys it, and a high failure rate multiplies the difference. CodeReviewerAgent grows context at +1,558 tok/call against +379 and +383 for the other two roles, plausibly from error-and-revision loops; confirm before building on it.

**(f) Dataset size as an independent variable** — run the same workflow over inputs of different size and observe prompt length, call count, and retry count. This connects scientific data scale to KV-cache pressure, a dimension conversational workloads do not have.

### Priority

(a) and (b) first. (a) is cross-layer, unique to this tooling, and produces a new metric. (b) is nearly free and may overturn the assumption that agentic workloads are inherently hard to cache. (c) through (f) follow, with (e) the closest to the real pain of scientific workflows.

## Prerequisites

**Replay.** Controlling variables requires detaching prompt sequences from the live agent, since no two agent runs are identical. Persist the full message sequence per call and replay it against a local vLLM or SGLang instance. Without replay, only single-point observation is possible, not parameter study.

**Multiple workflows.** One workflow yields a case study. GenoMAS, SciLink, 1000genome, and Montage together support the actual claim: how context reuse structure differs systematically between agentic and traditional workflows.
