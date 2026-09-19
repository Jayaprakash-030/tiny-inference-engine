#!/usr/bin/env python3
"""Headless benchmark runner. No notebook state, so this works over SSH.

  python scripts/run_benchmark.py --milestone 1 2 --max-new-tokens 400
  python scripts/run_benchmark.py --milestone 3 --batch-sizes 1 2 4 8 16 32 64 128 256
  python scripts/run_benchmark.py --milestone 4 --slot-sizes 2 4 8 16 --n-requests 32
  python scripts/run_benchmark.py --milestone 5 --block-budgets 16 32 48 64 --n-requests 16
"""

import argparse

from engine import load
from engine.batched import check_batch_matches_single, sweep
from engine.cached import check_matches_naive, measure_cached
from engine.continuous import (
    check_decode_batch_matches_run_one,
    check_run_one_matches_cached,
    sweep_continuous,
)
from engine.naive import measure_naive, warmup
from engine.paged import check_paged_matches_cached, sweep_block_budget
from engine.prompts import BATCH, LONG, SHORT
from engine.results import report, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch-sizes", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--slot-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--block-budgets", type=int, nargs="+", default=[16, 32, 48, 64])
    ap.add_argument("--n-requests", type=int, default=32)
    ap.add_argument("--arrival-rate-hz", type=float, default=8.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="benchmarks/results.json")
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    rt = load(args.model) if args.model else load()
    rt.describe()
    print()

    warmup(rt, SHORT)
    results = []

    if 1 in args.milestone:
        print("=== milestone 1: naive loop ===")
        for label, prompt in [("short_prompt", SHORT), ("long_prompt", LONG)]:
            results.append(report(measure_naive(rt, label, prompt, args.max_new_tokens)))

    if 2 in args.milestone:
        print("=== milestone 2: kv cache ===")
        if not args.skip_checks:
            assert check_matches_naive(rt, SHORT), "cache diverged from naive"
            assert check_matches_naive(rt, LONG), "cache diverged from naive"
        for label, prompt in [("short_prompt", SHORT), ("long_prompt", LONG)]:
            results.append(report(measure_cached(rt, label, prompt, args.max_new_tokens)))

    if 3 in args.milestone:
        print("=== milestone 3: static batching ===")
        if not args.skip_checks:
            assert check_batch_matches_single(rt, BATCH[:4]), "batching diverged"
        results.extend(sweep(rt, args.batch_sizes, args.max_new_tokens))

    if 4 in args.milestone:
        print("=== milestone 4: continuous batching ===")
        if not args.skip_checks:
            assert check_run_one_matches_cached(rt), "run_one diverged from cached"
            assert check_decode_batch_matches_run_one(rt), "decode_batch diverged"
        m4_tokens = args.max_new_tokens if args.max_new_tokens != 400 else 64
        results.extend(sweep_continuous(
            rt,
            slot_sizes=tuple(args.slot_sizes),
            n_requests=args.n_requests,
            max_new_tokens=m4_tokens,
            arrival_rate_hz=args.arrival_rate_hz,
        ))

    if 5 in args.milestone:
        print("=== milestone 5: paged KV cache ===")
        if not args.skip_checks:
            assert check_paged_matches_cached(rt), "paged diverged from cached"
        m5_tokens = args.max_new_tokens if args.max_new_tokens != 400 else 64
        m5_n = args.n_requests if args.n_requests != 32 else 16
        results.extend(sweep_block_budget(
            rt,
            block_budgets=tuple(args.block_budgets),
            n_requests=m5_n,
            max_new_tokens=m5_tokens,
            arrival_rate_hz=args.arrival_rate_hz,
        ))

    save(results, args.out)


if __name__ == "__main__":
    main()
