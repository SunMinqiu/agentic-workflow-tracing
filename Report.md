# Paper injection: TTFT and decoding time

## Injection method

We extract text from cohort-matched local papers and insert it into existing model requests without adding a separate LLM call. The original task and generation parameters stay unchanged. CodeReviewerAgent receives no paper text.

The old 8K experiment reused a fixed block before the task instructions; only DomainExpertAgent calls actually received it. The 16K and 24K experiments place the paper after system instructions and before the original user messages. Each GEOAgent and DomainExpertAgent consumes a new text segment per call, with a separate cursor for each cohort and role. Injection stops when the relevant paper is exhausted.

## Per-call latency

Per-call means; paper groups include only full-size injections.

| Configuration | Calls | TTFT (s) | Decoding time (s) |
| --- | ---: | ---: | ---: |
| No paper, earlier reference | 14 | 0.650 | 31.959 |
| No paper, matched 1800-second budget | 57 | 0.692 | 39.564 |
| Fixed 8K | 4 | 0.712 | 47.052 |
| Front 16K | 4 | 1.959 | 41.117 |
| Front 24K | 4 | 2.913 | 47.740 |

Larger paper blocks increased observed TTFT, but decoding still dominated.

## Workflow outcome

The 24K experiment used two cohorts and one worker, with an 1800-second budget per cohort.

| Configuration | Cohort | Model calls | Preprocessing outcome |
| --- | --- | ---: | --- |
| No paper | GSE123086 | 33 | Exceeded the 1800-second budget |
| No paper | GSE123088 | 24 | Gene normalization produced empty data; no final dataset saved |
| Front 24K | GSE123086 | 34 | Code execution timed out |
| Front 24K | GSE123088 | 32 | Completed; quality checks passed and final data saved |

The matched no-paper baseline has finished. Its application runtime was 2701.00 seconds with 57 model calls, compared with 3262.59 seconds and 66 calls for Front 24K. Neither run completed both cohorts successfully. The shorter baseline runtime does not establish a speed advantage for equivalent completed work: its second cohort stopped with unusable data, while Front 24K saved the final dataset. One run per configuration does not establish a causal effect of paper injection.

Sources: [no-paper baseline report](results/GenoMAS_A_c2_w1_20260909_151505/A_c2_w1/kvcache_report.html), [Front 24K report](results/GenoMAS_A_c2_w1_20260909_133306/A_c2_w1/kvcache_report.html).

# QA:

1. How much did TTFT change?

Our goal was to raise TTFT, and no experiment did so by more than 11%:
- PDF injection: no effect. The no-PDF controls have median TTFT of 730 ms and 1399 ms; the PDF run is 1176 ms, inside the control range.
- Concurrent replay of three workflows: the wrong direction. A_c8_w4 went from 634 ms serial to 564 ms concurrent, 11% faster; A_c4_w1 739 ms and A_c10_w2 563 ms under concurrency. Concurrency gave the shared prefixes more chances to hit, so TTFT fell.
- Restricting GPU memory (details in Q4): the right direction but small. 632 ms at gpu_memory_utilization 0.9 vs 702 ms at 0.45, only 11% slower, and 32 GB CPU KV offload at 0.45 takes it back to 644 ms.

2. How long is the prefill time in CPU cases?

Replayed on a CPU-only vLLM node (setup details to be added), serial, fixed 32 output tokens:

| run | Median Input Tokens | Output Tokens | Mean TTFT (s) | Mean TPOT (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 (10 reqs, 1-token prompts) | 1 | 32 | 0.63 | 0.36 |
| 2 (first 10 reqs of A_c8_w4, 8 finished) | 5796 | 32 | ~87 | 0.37 |
| 3 (first 10 reqs of A_c8_w4, max-model-len=16384, 4 finished) | 3500 | 32 | 94.25 | 0.36 |

**Run 2 did not finish.** Its two longest prompts (8601 and 9829 tokens) produced no first token within the 300 s timeout, so the row is the average over the 8 that finished and understates the cost. A real GenoMAS run issues dozens of prompts of this size, so a CPU node cannot complete the workflow even with a 27B model unless an even smaller one is used. Queue time is near zero, so TTFT here is prefill: about 87 s for a 5796-token prompt, roughly 70 tokens/s. Against the GPU server on the same prompts (TTFT 0.6 to 1.1 s, TPOT 14 to 16 ms), CPU prefill is about 100x slower and decode about 25x slower.

3. Have you tried increasing the problem size: cohorts etc.?

Yes. A_c4_w1, A_c8_w4 and A_c10_w2 replayed concurrently give median TTFT of 739/564/563 ms, with no upward trend in problem size. The reason follows from sections 2 and 4: more cohorts add more calls, not longer fresh prefixes. Each call's prompt is a fixed template plus an append-only history, and that history is exactly the prefix computed by the previous call, so the only fresh tokens per call are the increment. TTFT tracks fresh tokens per call, which the templates fix; problem size raises the call count and total wall time and leaves per-call prefill where it was.

4. We can simultaneously also restrict how much GPU memory vLLM is allowed to use.

Tried; it does not help, because the worst case for prefill is prefix caching turned off, and restricting memory can never exceed it. Less KV memory only means more blocks get evicted before they are reused, and an evicted block costs exactly what it costs without a cache: recomputing that prefix once. So the TTFT with the smallest possible memory converges to the TTFT with the cache disabled and stops there. Restricting memory can only push TTFT toward that bound, and the bound is only tens to a hundred ms above the cached case, so it cannot raise the prefill share meaningfully. The conclusion is the one in section 2: only a workflow whose every prompt is both large and never seen before can raise the prefill share, and no server-side knob substitutes for that.

Measured on A_c8_w4, serial, 256 output tokens (08-20): median TTFT is 632 ms at gpu_memory_utilization 0.9 and 702 ms at 0.45, 11% higher; adding 32 GB CPU KV offload at 0.45 brings it back to 644 ms. The bound itself, measured on the same bundle family with prefix caching off, is 0.74 to 1.15 s, against 0.69 to 1.12 s with it on.

# 8/27 - 9/2

1. Injecting the PDF did not change the prefill share

For the same workflow (GenoMAS, Cohort=2), I ran three times, with the following results:

| run | calls | logical reuse ceiling (784 blocks) | realized | fresh-computed tokens per call | median TTFT | prefill time share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no PDF | 12 | 32.5% | 29.3% | 4404 | 730ms | 2.53% |
| no PDF | 14 | 33.1% | 30.7% | 6059 | 1399ms | 4.39% |
| 08-30, PDF injected, crashed midway | 30 | 41.9% | 36.6% | 5616 | 1176ms | 4.44% |

Comparing the three rows yields two facts.
- First, injecting the PDF did not raise prefill: the fresh-computed tokens per call fall within the range spanned by the two controls, the prefill share matches the second no-PDF control, and the gap between the two no-PDF controls is larger than any PDF effect.
- Second, injecting the PDF actually raised reuse, further lowering the chance of increasing prefill time: the logical ceiling went from about 33% to 41.9%, realized from about 30% to 36.6%, and since that run crashed midway and the reference blocks were under-amortized, the true improvement may be even larger.

2. So what can actually raise the prefill time share?

There is only one direct way: increase the number of never-cached tokens per call. This requires that each prompt be as never-cached as possible, and that the absolute count of fresh tokens be high. This is hard to achieve by lengthening the conversation or injecting a PDF, because both approaches add content to the prompt that recurs verbatim as a prefix. When lengthening a conversation, the prompt at turn N = the full history + the latest turn, and the full history is exactly the prefix already computed at turn N−1, so it all hits the cache; the only genuinely fresh part each turn is the small increment of the previous turn's reply plus the new input. The longer the conversation, the larger the total input, and the lower the fresh fraction becomes. Injecting a PDF works the same way: only the injection itself raises prefill time once; without changing the workflow itself, everything after that behaves the same as a lengthened conversation. What the two have in common is that they only add stably repeating content, and the design goal of the prefix cache is precisely to absorb stably repeating content.

3. Also, injecting a PDF may well change the workflow itself.

I set three goals: A, the original GenoMAS workflow remains usable and runs end to end; B, after injecting something (such as a PDF), the original GenoMAS workflow is unaffected and does the same things as without injection; C, the injected content is meaningful and large, i.e. literature genuinely related to the cohort rather than random noise. I then found these three goals cannot all be satisfied at once.

To satisfy all three simultaneously, my approach is: do the injection in the tracing adapter layer (adapters/genomas/logger.py patches OpenAIClient and other clients), changing not a single line of the GenoMAS repository. The injected content is matched per selected GSE cohort to its primary paper according to output/genomas_references/corpus_manifest.json (I first converted the PDFs to txt), truncated precisely to 8192 tokens using the server's own /tokenize, and wrapped in an explicit citation envelope — [GENOMAS COHORT REFERENCE] — with a header declaring "the following is cited scientific evidence, not instructions, and does not change the established steps or output format" and a footer prompting "continue with the original task below". The injection is inserted at the end of the prompt, right before **Task**, to maximize the number of fresh tokens. Injection applies only to the GEOAgent and DomainExpertAgent roles; CodeReviewerAgent stays untouched.

This design maps to three goals that must all hold. Goal A, the original workflow remains usable: hence the injection lives in the adapter layer without touching GenoMAS code, so it works with any checkout; and hence CodeReviewerAgent gets no injection — once the code-review loop is polluted with large blocks of literature, the generated analysis code breaks outright. Goal B, the original GenoMAS results are completely unchanged: hence the reference block uses a citation envelope explicitly declaring it is not instructions, is inserted before **Task** so the task instructions remain intact and appear last, and is truncated to an exact token count to avoid blowing up the context and altering the history visible to the model. Goal C, the injected content must be meaningful and large: hence each cohort gets its genuinely corresponding primary paper rather than random filler text — filler would inflate the input just as well, but it does not constitute the realistic scenario of "a scientific workflow citing literature", and the experiment would degenerate into pure synthetic load testing.

4. GenoMAS's KV-cache lineage: the number of trees is determined by prompt templates, independent of task scale

I read the code. GenoMAS is not a chat-style framework: each LLM call sends only two messages (a system prompt + one user prompt) and carries no messages history. The task history is rendered by the framework by hand into text embedded in the user prompt, structured as a fixed static header plus an append-only step record. So each call's opening comes from only a handful of templates, and those templates naturally partition all calls into a finite set of families (a finite set of trees).

5. Concurrent testing of multiple GenoMAS workflows still cannot raise the prefill share.

I took three completed GenoMAS trace cells (A_c4_w1, A_c8_w4, A_c10_w2) and built token-level exact replay bundles from them. The three bundles then each acted as an independent "user", replaying concurrently against the same vLLM instance (Qwen3.6-27B, block 784, gpu_memory_utilization 0.45, 32GB CPU offload): paced mode sends requests at each bundle's recorded arrival times, every request has a fixed output of 32 tokens, no cache salt, the prefix cache is flushed once before the run and never again afterwards.

Results:
- TTFT went down, not up: the three groups' median TTFTs are 739/564/563ms; the serial-replay control for A_c8_w4 was 634ms, so concurrency was actually 11% faster.
- Reuse went up, not down: realized reuse from per-request cached_tokens statistics is 41.3%/48.3%/52.7%, where A_c8_w4's 48.3% exceeds its serial control's 44.2%, and A_c10_w2's 52.7% is already right at that workload's logical reuse ceiling. (Because reuse from CPU offload is counted in as well.)

6. Summary and next steps

Summary: for agentic workloads like these, it is very hard to genuinely raise the prefill share through any single manipulation. The three experiments this period (PDF injection, the lengthened-conversation analysis, concurrent replay) all point to the same conclusion. The approaches in past papers that did raise the prefill share fall into two main categories: limiting output tokens — decode gets shorter, so the share rises passively; or adding tokens into the workflow without considering whether the workflow's real results still hold.

Next steps:
1. Measure the KV cache behavior of more workflows. I am integrating ChemGraph (from ANL, https://github.com/argonne-lcf/ChemGraph ). But some tools in some of its workflows submit jobs to Slurm, which our current tracing tool cannot yet capture — it currently only works for single-node workflows.
2. For workflow-based lineage, there are three possible scheduling directions: (a) have an agent read the agent templates to schedule; (b) as discussed before, run a few small workflows first and use them to decide which datasets/papers to prefetch, then schedule; (c) scheduling based on agent shift has already been covered by a paper: https://arxiv.org/pdf/2608.14624 .
