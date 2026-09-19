# Tiny Inference Engine

A small LLM inference stack built from scratch on `Qwen/Qwen3-0.6B` (Colab T4).
Each stage adds one serving technique, checks token-identical greedy output against
the previous stage, then measures what changed.

**What it implements:** KV cache · static & continuous batching · paged KV blocks  
**Evidence:** `benchmarks/results.json` + plots in `benchmarks/plots/`

## Highlights

| Stage | Takeaway (Colab T4) |
|-------|---------------------|
| KV cache | Long-prompt decode flattens (~43 ms/step); **1.6×** tok/s vs naive |
| Static batching | Throughput **~34×** from batch 1→256; ~50% slot waste with staggered finishes |
| Continuous batching | At 4 slots: **74.6** vs **52.3** tok/s; wait **1195** vs **3329** ms |
| Paged KV | Same 32-block budget: paged **16/16** completed, reserved **11/16** |

## Quickstart

```bash
pip install -e .
python scripts/run_benchmark.py --milestone 1 2 --max-new-tokens 400
python scripts/run_benchmark.py --milestone 3 --batch-sizes 1 2 4 8 16 32 64 128 256
python scripts/run_benchmark.py --milestone 4 --slot-sizes 2 4 8 16 --n-requests 16 --max-new-tokens 64
python scripts/run_benchmark.py --milestone 5 --block-budgets 16 32 48 64 --n-requests 16 --max-new-tokens 64
```

Colab:

```python
!git clone https://github.com/Jayaprakash-030/tiny-inference-engine.git && pip install -e tiny-inference-engine
from engine import load
rt = load(); rt.describe()
```

## Stages

Numbers below are from Colab T4 runs in `benchmarks/results.json`.

### 1. Naive loop
Full recompute every step. Per-step latency climbs with sequence length → quadratic
total cost.

[Plot: naive vs KV cache](benchmarks/plots/m1_vs_m2.png)

| prompt | tokens | tok/s | median step | last step |
|--------|-------:|------:|------------:|----------:|
| short | 5 | 18.9 | 49.2 ms | 70.7 ms |
| long | 227 | 13.7 | 71.2 ms | 110.0 ms |

### 2. KV cache
Prefill once, decode one token at a time with cached K/V. Greedy output matches
the naive loop exactly.

| prompt | naive tok/s | cached tok/s | speedup | decode median |
|--------|------------:|-------------:|--------:|--------------:|
| short | 18.9 | 21.2 | 1.1× | 43.2 ms |
| long | 13.7 | 21.9 | 1.6× | 43.2 ms |

### 3. Static batching
Amortize the per-step floor across N sequences. Short requests still occupy slots
until the longest finishes (`wasted_slot_pct`).

[Plot: batch sweep](benchmarks/plots/m3_batching.png)

| batch | tok/s | per-seq tok/s | wasted % | peak mem |
|------:|------:|--------------:|---------:|---------:|
| 1 | 22.8 | 22.8 | 0.0 | 1.2 GB |
| 16 | 220.5 | 26.0 | 46.9 | 1.6 GB |
| 64 | 628.1 | 19.3 | 49.2 | 2.9 GB |
| 256 | 783.2 | 6.1 | 49.8 | 8.0 GB |

### 4. Continuous batching
Iteration-level admit / decode / evict on a random-arrival workload; static waves
as the baseline.

[Plot: static vs continuous](benchmarks/plots/m4_continuous.png)

| slots | static tok/s | cont tok/s | static wait | cont wait |
|------:|-------------:|-----------:|------------:|----------:|
| 2 | 33.8 | 44.6 | 5875 ms | 3802 ms |
| 4 | 52.3 | 74.6 | 3329 ms | 1195 ms |
| 8 | 74.3 | 105.6 | 1555 ms | 106 ms |
| 16 | 74.7 | 104.6 | 1420 ms | 60 ms |

### 5. Paged KV cache
Fixed-size blocks + block table. **Reserved** admission pins prompt+max_new up
front; **paged** grows on decode and preempts with `oom_grow` if the pool is empty.

[Plot: reserved vs paged](benchmarks/plots/m5_paged_kv.png)

| n_blocks | reserved done/rejected | paged done/rejected | paged peak blocks |
|---------:|------------------:|---------------:|------------------:|
| 16 | 6 / 10 | 10 / 6 | 16/16 |
| 32 | 11 / 5 | 16 / 0 | 28/32 |
| 48 | 15 / 1 | 16 / 0 | 29/48 |
| 64 | 16 / 0 | 16 / 0 | 29/64 |

## Layout

```
engine/      importable modules — one stage per file
notebooks/   Colab drivers (imports, runs, plots)
scripts/     headless benchmarks
benchmarks/  results.json and plots/
```

Logic lives in `engine/`; notebooks only drive and plot.

## Correctness

Every stage is checked against the previous one before it is timed. Greedy decoding
is deterministic — a broken KV path does not crash; it produces fluent wrong text.
