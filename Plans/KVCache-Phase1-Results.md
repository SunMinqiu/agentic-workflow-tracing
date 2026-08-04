# Phase 1 Results: KV-Cache Data from Local Runs

These are the only three local runs that carry **real** `cached_tokens` — i.e. measured KV-cache reuse from the serving backend. All three use OpenAI `gpt-4o-mini`, so token counts and cache rates are directly comparable.

---

## Exp 1 — GenoMAS, 2 cohorts (A_c2_w2)


| Metric                                                          | Value            |
| --------------------------------------------------------------- | ---------------- |
| LLM calls                                                       | 76               |
| Per-call input tokens (median / max)                            | 5,388 / 24,251   |
| Output / input ratio                                            | 5.4%             |
| Total input tokens                                              | 575,841          |
| Context growth, tokens/call (GEO / CodeReviewer / DomainExpert) | 928 / 1,716 / 34 |
| **Realized KV reuse (cacheRead %)**                             | **41%**          |
| cacheRead tokens                                                | 235,008          |
| Verbatim re-submission %                                        | 6%               |


---



## Exp 2 — GenoMAS, 3 cohorts (A_c3_w2)


| Metric                                                          | Value           |
| --------------------------------------------------------------- | --------------- |
| LLM calls                                                       | 128             |
| Per-call input tokens (median / max)                            | 12,069 / 19,330 |
| Output / input ratio                                            | 2.2%            |
| Total input tokens                                              | 1,428,416       |
| Context growth, tokens/call (GEO / CodeReviewer / DomainExpert) | 189 / 368 / 155 |
| **Realized KV reuse (cacheRead %)**                             | **65%**         |
| cacheRead tokens                                                | 924,928         |
| Verbatim re-submission %                                        | 1%              |


---



## Exp 3 — SciLink, grain segmentation (polycrystalline_grains_basic)


| Metric                                  | Value           |
| --------------------------------------- | --------------- |
| LLM calls                               | 37              |
| Per-call input tokens (median / max)    | 10,528 / 45,446 |
| Output / input ratio                    | 4.9%            |
| Total input tokens                      | 442,576         |
| Context growth, tokens/call (all calls) | 76              |
| **Realized KV reuse (cacheRead %)**     | **18%**         |
| cacheRead tokens                        | 77,696          |
| Verbatim re-submission %                | 0%              |


---

