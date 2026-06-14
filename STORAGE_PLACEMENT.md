# Storage Placement Standard

How we decide which agent artifacts go on Lustre vs local disk in the SRAgent
+ eBPF tracing harness on CloudLab.

## Decision Rules (in priority order)

For every kind of file the harness or the agent produces or reads, apply
these three rules in order. The first one that fits wins.

1. **Large sequential I/O** — single file > 100 MB, or aggregate workload
   total > 1 GB.
   → **Lustre.** This is exactly what Lustre is built for.

2. **Many small files / high-frequency random reads** — database indexes,
   wheel caches, vector-store sharded files, and similar.
   → **Local disk.** This pattern is Lustre's worst case (metadata-heavy,
   small-IO-bound). Putting it on Lustre will *slow it down*, not speed
   it up — even though intuition says "DB → shared storage".

3. **Everything else** — code, configuration, small intermediate
   artifacts, visualization outputs.
   → **Local disk.** Default unless rule 1 or 2 triggers.

> **Important caveat for rule 2.** Not every file called a "database" lands
> on Lustre. ChromaDB's typical access pattern (many small random reads
> against vector-index shards) makes it a rule-2 case despite the
> "database" label. Only databases that are themselves large monolithic
> files with sequential I/O (e.g. multi-GB Parquet, large fastq files)
> qualify under rule 1.

## Placement Matrix

| Category | Examples | Typical size | Access pattern | Rule | **Lands on** |
|---|---|---|---|---|---|
| Code + venv | `~/SRAgent/`, `.venv/` | ~1 GB | Load at startup | 3 | **Local** |
| ChromaDB (vector store) | Uberon / MONDO ontology indexes | 50–500 MB | Many small random reads | 2 | **Local** ⚠️ |
| Agent state cache | `~/.cache/SRAgent/`, appdirs | <100 MB | Read at startup | 3 | **Local** |
| uv cache | `~/.cache/uv/` | 1–5 GB (lots of small wheels) | Frequent small reads during install | 2 | **Local** |
| **Downloaded papers PDFs** | `papers --output-dir <dir>` | 1–5 MB per paper, aggregate up to GB | Write once, read once | 1 | **Lustre** |
| **Downloaded fastq / SRA data** | `sequences` / fastq-dump output | GB–TB | Large file sequential I/O | 1 | **Lustre** |
| **fastq-dump tempfiles** | `$TMPDIR/<runid>.tmp` | Several GB | Sequential write/read | 1 | **Lustre** (via `TMPDIR`) |
| **Filtered datasets** | `metadata` / `find-datasets` CSV outputs | Variable, can be large | Bulk write + later read | 1 | **Lustre** |
| Trace raw logs | `ebpf_events.log`, `tool_calls.log` | 10 MB – 1 GB | Append during run, full read at parse time | 1 (borderline) | **Local** by default; **Lustre** via `BASE_OUT` override |
| Trace parsed products | `parsed.json`, `pi_summary.json` | <50 MB | Single write, read by visualizer | 3 | **Local** |
| Visualizations | `visualizations/*.html`, `*.png` | <10 MB | Read in browser | 3 | **Local** |

## Three Directory Variables

These map directly into `config_sragent.env`:

| Variable | Default value | Role |
|---|---|---|
| `WORK_DIR` | `/tmp/sragent_work` (local) | Per-workload cwd. ChromaDB, appdirs cache, agent's small outputs land here. |
| `DATA_DIR` | `/mnt/lustrefs/sragent_data` (Lustre) | Per-workload download dir. Papers PDFs, fastq files, fastq-dump tempfiles (via `TMPDIR`), filtered CSV outputs. |
| `BASE_OUT` | `./traces/<timestamp>` (local) | Trace artifacts: raw eBPF log, tool calls, parsed JSON, summary, HTML viz. Override to a Lustre path if you want long-term persistence across CloudLab node refreshes. |

## How Workloads Reference These at Runtime

Inside `WORKLOADS` array entries, use `$DATA` and `$SCRIPT_DIR` as escaped
placeholders. The trace script exports `DATA=$DATA_DIR/<workload_name>` per
workload before evaluating the args, and also sets
`TMPDIR=$DATA/tmp` so fastq-dump's tempfiles automatically route to Lustre.

```bash
WORKLOADS=(
  # Pure HTTP — no large downloads, no flags needed for routing.
  "entrez_basic||entrez|\"Convert GSE121737 to SRX accessions\""

  # Downloads PDFs → route output to Lustre via $DATA.
  "papers_basic||papers|SRX4967527 --output-dir \$DATA"

  # Reads a small CSV fixture from the harness directory.
  "metadata_basic||metadata|\$SCRIPT_DIR/fixtures/metadata_input.csv"
)
```

`\$DATA` is escaped (literal dollar in the array entry) so it expands at
workload-launch time, not at config-source time.

## Why the ChromaDB Exception Matters

Putting ChromaDB on Lustre is a tempting mistake because:

- It's "a database" → intuition says shared/persistent storage
- It's small-medium → seems harmless

But its access pattern is **the canonical bad case for Lustre**:

- Many small files (chroma_db sharded segments, parquet pieces)
- Random reads triggered by every vector query
- Metadata-heavy operations (open/stat per shard per query)

A single ChromaDB query can issue dozens of `openat`/`fstat` calls. On
Lustre this hits MDS hard for a workload that fundamentally doesn't need
shared storage — the same vector index works fine when local on each
client. **Keep ChromaDB local.**

If a future workload demands a genuinely large shared vector store
(multi-GB indexes, distributed query), revisit this decision and benchmark
both placements before committing.

## Trace Logs: Why Default Local Despite Their Size

`ebpf_events.log` can grow to hundreds of MB on long traces. By rule 1 that
arguably qualifies for Lustre. We default to local anyway because:

1. **Append pattern is fine on local SSD** — no benefit from Lustre's
   parallel OST striping for a single-writer append.
2. **Post-processing reads the full file once**, locally, then produces
   small outputs. Pulling raw logs from Lustre adds a network round-trip
   without saving anything.
3. **Persistence across node refreshes** is the only real argument for
   Lustre placement here, and it's a one-line override:
   `BASE_OUT=/mnt/lustrefs/traces/$(date +%Y%m%d_%H%M%S)`.

Long-running production traces or a publishable trace corpus → flip the
override. Day-to-day iteration → keep local.
