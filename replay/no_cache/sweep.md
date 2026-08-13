# Fixed-input Test — no_cache

_Generated 2026-08-12 · 4 arm(s) · prefill-only replay._

## Summary — tokens

| arm | calls | total input | total output | cache queries | cache hits | realized reuse | repetitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block 784 · auto · prefix off · GPU 0.90 · 08-12 11:59 | 12 | 74,798 | 384 | n/a | n/a | n/a | 1 |
| block 784 · auto · prefix off · GPU 0.90 · 08-12 12:00 | 17 | 165,818 | 544 | n/a | n/a | n/a | 1 |
| block 784 · auto · prefix off · GPU 0.90 · 08-12 12:02 | 11 | 96,386 | 352 | n/a | n/a | n/a | 1 |
| block 784 · auto · prefix off · GPU 0.90 · 08-12 12:03 | 14 | 122,460 | 448 | n/a | n/a | n/a | 1 |

## Summary — time and configuration

| arm | median TTFT | median TPOT | median request latency | wall | vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| block 784 · auto · prefix off · GPU 0.90 · 08-12 11:59 | 0.74s | 16.25ms/token | 1.24s | 17.49s | baseline |
| block 784 · auto · prefix off · GPU 0.90 · 08-12 12:00 | 1.15s | 16.01ms/token | 1.61s | 27.28s | +55.75% TTFT |
| block 784 · auto · prefix off · GPU 0.90 · 08-12 12:02 | 1.10s | 16.02ms/token | 1.59s | 16.97s | +49.05% TTFT |
| block 784 · auto · prefix off · GPU 0.90 · 08-12 12:03 | 1.08s | 15.99ms/token | 1.57s | 21.06s | +46.31% TTFT |

12 calls, 74,798 total input tokens, output fixed at 32 tokens per request, replayed in `packed` mode from `GenoMAS_A_c2_w1_20260805_152115_A_c2_w1.json`.

**Not comparable** — different bundles: GenoMAS_A_c2_w1_20260805_152115_A_c2_w1.json, GenoMAS_A_c2_w1_20260809_193357_A_c2_w1.json, GenoMAS_A_c2_w1_20260809_211004_A_c2_w1.json, GenoMAS_A_c2_w1_20260811_111925_A_c2_w1.json; different call counts: 11, 12, 14, 17; different total input: 122460, 165818, 74798, 96386.

Every arm reports the same serving config; the server was not restarted with a different knob between them, so any difference above is run-to-run noise, not a knob effect.

![Realized reuse and TTFT by arm](visualizations/sweep_comparison.png)


## Per arm

### block 784 · auto · prefix off · GPU 0.90 · 08-12 11:59

- Bundle: `GenoMAS_A_c2_w1_20260805_152115_A_c2_w1.json` · mode: `packed` · cache: `cold_by_reset` · repetitions: 1

**Server config**

| key | value |
| --- | --- |
| `_block_size_resolved` | `True` |
| `block_size` | `784` |
| `cache_dtype` | `auto` |
| `calculate_kv_scales` | `False` |
| `enable_prefix_caching` | `False` |
| `engine` | `0` |
| `gpu_memory_utilization` | `0.9` |
| `is_attention_free` | `False` |
| `kv_cache_dtype_skip_layers` | `[]` |
| `kv_cache_max_concurrency` | `10.292397660818713` |
| `kv_cache_memory_bytes` | `None` |
| `kv_cache_size_tokens` | `1349045` |
| `kv_offloading_backend` | `native` |
| `kv_offloading_size` | `None` |
| `kv_sharing_fast_prefill` | `False` |
| `mamba_block_size` | `131072` |
| `mamba_cache_dtype` | `auto` |
| `mamba_cache_mode` | `none` |
| `mamba_page_size_padded` | `None` |
| `mamba_ssm_cache_dtype` | `float32` |
| `num_cpu_blocks` | `None` |
| `num_gpu_blocks` | `1760` |
| `num_gpu_blocks_override` | `None` |
| `prefix_caching_hash_algo` | `sha256` |
| `prefix_match_unit` | `None` |
| `skip_page_size_padded` | `None` |
| `sliding_window` | `None` |
| `user_specified_block_size` | `True` |
| `user_specified_mamba_block_size` | `False` |

**Artifacts**

- repetition 1: [summary](block784_dtype-auto_prefix-off_gpu0.9_20260812T115914/rep0/summary.json) · [requests](block784_dtype-auto_prefix-off_gpu0.9_20260812T115914/rep0/requests.jsonl) · [metrics before](block784_dtype-auto_prefix-off_gpu0.9_20260812T115914/rep0/metrics_before.prom) · [metrics after](block784_dtype-auto_prefix-off_gpu0.9_20260812T115914/rep0/metrics_after.prom)

---

### block 784 · auto · prefix off · GPU 0.90 · 08-12 12:00

- Bundle: `GenoMAS_A_c2_w1_20260809_193357_A_c2_w1.json` · mode: `packed` · cache: `cold_by_reset` · repetitions: 1

**Server config**

| key | value |
| --- | --- |
| `_block_size_resolved` | `True` |
| `block_size` | `784` |
| `cache_dtype` | `auto` |
| `calculate_kv_scales` | `False` |
| `enable_prefix_caching` | `False` |
| `engine` | `0` |
| `gpu_memory_utilization` | `0.9` |
| `is_attention_free` | `False` |
| `kv_cache_dtype_skip_layers` | `[]` |
| `kv_cache_max_concurrency` | `10.292397660818713` |
| `kv_cache_memory_bytes` | `None` |
| `kv_cache_size_tokens` | `1349045` |
| `kv_offloading_backend` | `native` |
| `kv_offloading_size` | `None` |
| `kv_sharing_fast_prefill` | `False` |
| `mamba_block_size` | `131072` |
| `mamba_cache_dtype` | `auto` |
| `mamba_cache_mode` | `none` |
| `mamba_page_size_padded` | `None` |
| `mamba_ssm_cache_dtype` | `float32` |
| `num_cpu_blocks` | `None` |
| `num_gpu_blocks` | `1760` |
| `num_gpu_blocks_override` | `None` |
| `prefix_caching_hash_algo` | `sha256` |
| `prefix_match_unit` | `None` |
| `skip_page_size_padded` | `None` |
| `sliding_window` | `None` |
| `user_specified_block_size` | `True` |
| `user_specified_mamba_block_size` | `False` |

**Artifacts**

- repetition 1: [summary](block784_dtype-auto_prefix-off_gpu0.9_20260812T120034/rep0/summary.json) · [requests](block784_dtype-auto_prefix-off_gpu0.9_20260812T120034/rep0/requests.jsonl) · [metrics before](block784_dtype-auto_prefix-off_gpu0.9_20260812T120034/rep0/metrics_before.prom) · [metrics after](block784_dtype-auto_prefix-off_gpu0.9_20260812T120034/rep0/metrics_after.prom)

---

### block 784 · auto · prefix off · GPU 0.90 · 08-12 12:02

- Bundle: `GenoMAS_A_c2_w1_20260809_211004_A_c2_w1.json` · mode: `packed` · cache: `cold_by_reset` · repetitions: 1

**Server config**

| key | value |
| --- | --- |
| `_block_size_resolved` | `True` |
| `block_size` | `784` |
| `cache_dtype` | `auto` |
| `calculate_kv_scales` | `False` |
| `enable_prefix_caching` | `False` |
| `engine` | `0` |
| `gpu_memory_utilization` | `0.9` |
| `is_attention_free` | `False` |
| `kv_cache_dtype_skip_layers` | `[]` |
| `kv_cache_max_concurrency` | `10.292397660818713` |
| `kv_cache_memory_bytes` | `None` |
| `kv_cache_size_tokens` | `1349045` |
| `kv_offloading_backend` | `native` |
| `kv_offloading_size` | `None` |
| `kv_sharing_fast_prefill` | `False` |
| `mamba_block_size` | `131072` |
| `mamba_cache_dtype` | `auto` |
| `mamba_cache_mode` | `none` |
| `mamba_page_size_padded` | `None` |
| `mamba_ssm_cache_dtype` | `float32` |
| `num_cpu_blocks` | `None` |
| `num_gpu_blocks` | `1760` |
| `num_gpu_blocks_override` | `None` |
| `prefix_caching_hash_algo` | `sha256` |
| `prefix_match_unit` | `None` |
| `skip_page_size_padded` | `None` |
| `sliding_window` | `None` |
| `user_specified_block_size` | `True` |
| `user_specified_mamba_block_size` | `False` |

**Artifacts**

- repetition 1: [summary](block784_dtype-auto_prefix-off_gpu0.9_20260812T120203/rep0/summary.json) · [requests](block784_dtype-auto_prefix-off_gpu0.9_20260812T120203/rep0/requests.jsonl) · [metrics before](block784_dtype-auto_prefix-off_gpu0.9_20260812T120203/rep0/metrics_before.prom) · [metrics after](block784_dtype-auto_prefix-off_gpu0.9_20260812T120203/rep0/metrics_after.prom)

---

### block 784 · auto · prefix off · GPU 0.90 · 08-12 12:03

- Bundle: `GenoMAS_A_c2_w1_20260811_111925_A_c2_w1.json` · mode: `packed` · cache: `cold_by_reset` · repetitions: 1

**Server config**

| key | value |
| --- | --- |
| `_block_size_resolved` | `True` |
| `block_size` | `784` |
| `cache_dtype` | `auto` |
| `calculate_kv_scales` | `False` |
| `enable_prefix_caching` | `False` |
| `engine` | `0` |
| `gpu_memory_utilization` | `0.9` |
| `is_attention_free` | `False` |
| `kv_cache_dtype_skip_layers` | `[]` |
| `kv_cache_max_concurrency` | `10.292397660818713` |
| `kv_cache_memory_bytes` | `None` |
| `kv_cache_size_tokens` | `1349045` |
| `kv_offloading_backend` | `native` |
| `kv_offloading_size` | `None` |
| `kv_sharing_fast_prefill` | `False` |
| `mamba_block_size` | `131072` |
| `mamba_cache_dtype` | `auto` |
| `mamba_cache_mode` | `none` |
| `mamba_page_size_padded` | `None` |
| `mamba_ssm_cache_dtype` | `float32` |
| `num_cpu_blocks` | `None` |
| `num_gpu_blocks` | `1760` |
| `num_gpu_blocks_override` | `None` |
| `prefix_caching_hash_algo` | `sha256` |
| `prefix_match_unit` | `None` |
| `skip_page_size_padded` | `None` |
| `sliding_window` | `None` |
| `user_specified_block_size` | `True` |
| `user_specified_mamba_block_size` | `False` |

**Artifacts**

- repetition 1: [summary](block784_dtype-auto_prefix-off_gpu0.9_20260812T120323/rep0/summary.json) · [requests](block784_dtype-auto_prefix-off_gpu0.9_20260812T120323/rep0/requests.jsonl) · [metrics before](block784_dtype-auto_prefix-off_gpu0.9_20260812T120323/rep0/metrics_before.prom) · [metrics after](block784_dtype-auto_prefix-off_gpu0.9_20260812T120323/rep0/metrics_after.prom)

---

[Back to experiments index](../../results/index.html)

