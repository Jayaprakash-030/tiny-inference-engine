# Tiny Inference Engine

An LLM inference engine built from scratch, one optimization at a time, with each
step measured against the last. Model: `Qwen/Qwen3-0.6B` on a single T4.

The point is not to beat vLLM. It is to build each optimization, measure what it
actually buys, and understand why production engines are shaped the way they are.

## Quickstart

```bash
pip install -e .
python scripts/run_benchmark.py --milestone 1 2 --max-new-tokens 400
```

In Colab:

```python
!git clone <repo-url> && pip install -e tiny-inference-engine
from engine import load
from engine.batched import sweep
rt = load(); rt.describe()
```

## Milestones

### 1. Naive loop — the baseline
Full recompute every step: the entire sequence goes through the model again, and
all of it is discarded. Per-step latency is flat while short, then climbs linearly
with sequence length — meaning total cost grows quadratically.

*Results: TODO*

### 2. KV cache
Keep the keys and values already computed and feed back one token instead of the
whole sequence. Verified token-identical to the naive loop under greedy decoding.
The growth flattens; the fixed per-step floor does not move.

*Results: TODO*

### 3. Static batching
The per-step floor is paid per step, not per token, so running N sequences through
one step costs barely more than one. Throughput climbs until the GPU stops being
launch-bound and becomes compute-bound.

Also measures `wasted_slot_pct`: finished sequences keep occupying their slot until
the longest sequence in the batch ends.

*Results: TODO*

### 4. Continuous batching
Iteration-level scheduling: evict finished sequences mid-flight and admit waiting
ones into free slots. Same arrival workload is served with static waves and with
continuous scheduling; compare tok/s, wait, latency, and slot utilization.

  python scripts/run_benchmark.py --milestone 4 --slot-sizes 2 4 8 16 --n-requests 32

*Results: see benchmarks/results.json (milestone 4 rows) and notebooks/04_continuous_batching.ipynb*

### 5. Paged KV cache
*Not started.* Fixed-size blocks and a block table, instead of one contiguous
allocation per sequence.

### 6. Prefix caching
*Not started.* Reuse cache across requests that share a prompt prefix.

## Layout

```
engine/      importable modules — one per milestone
notebooks/   thin drivers for interactive work
scripts/     headless runners
benchmarks/  results.json and plots
```

Notebooks contain no logic, only imports, calls, and plots. All logic lives in
`engine/`, so the same code runs in Colab and over SSH.

## Correctness

Every optimization is checked against the previous one before it is timed. Greedy
decoding is deterministic, so an optimization must produce byte-identical tokens —
a broken KV cache does not crash, it produces fluent wrong text, quickly.
