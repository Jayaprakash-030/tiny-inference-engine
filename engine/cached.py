"""Milestone 2 — KV cache.

Two changes from the naive loop, and they only work together:

  1. Keep the keys and values the model computes (use_cache=True, capture
     past_key_values).
  2. Feed back ONE token instead of the whole sequence.

Doing (1) without (2) gives you all of the memory cost and none of the speedup.
"""

import statistics
import time

import torch

from engine.model import Runtime
from engine.naive import naive_generate


@torch.inference_mode()
def cached_generate(rt: Runtime, prompt: str, max_new_tokens: int = 400):
    ids = rt.tokenizer(prompt, return_tensors="pt").input_ids.to(rt.device)
    prompt_len = ids.shape[1]
    step_ms = []
    generated = []

    # --- PREFILL: one pass over the whole prompt, cache built from nothing ---
    rt.sync()
    t0 = time.perf_counter()

    out = rt.model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    rt.sync()
    step_ms.append((time.perf_counter() - t0) * 1000)
    generated.append(next_id.item())

    # --- DECODE: one token in, one token out, cache carried forward ---
    for _ in range(max_new_tokens - 1):
        if next_id.item() == rt.tokenizer.eos_token_id:
            break

        rt.sync()
        t0 = time.perf_counter()

        out = rt.model(input_ids=next_id, past_key_values=cache, use_cache=True)
        cache = out.past_key_values  # reassign — it grows by one entry each step
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        rt.sync()
        step_ms.append((time.perf_counter() - t0) * 1000)
        generated.append(next_id.item())

    full = torch.cat([ids, torch.tensor([generated], device=rt.device)], dim=-1)
    return full, prompt_len, step_ms


def check_matches_naive(rt: Runtime, prompt: str, n: int = 32) -> bool:
    """Greedy decoding is deterministic, so cached and naive must agree exactly.

    A broken KV cache rarely crashes — it produces fluent but wrong text, and it
    is fast, because it skips work it should not skip. Without this check a bug
    looks like a success.
    """
    naive_ids, _, _ = naive_generate(rt, prompt, max_new_tokens=n)
    cached_ids, _, _ = cached_generate(rt, prompt, max_new_tokens=n)

    same = torch.equal(naive_ids, cached_ids)
    print("token-identical:", same)
    if not same:
        print("  naive :", rt.tokenizer.decode(naive_ids[0])[:120])
        print("  cached:", rt.tokenizer.decode(cached_ids[0])[:120])
    return same


def measure_cached(rt: Runtime, label: str, prompt: str, max_new_tokens: int = 400) -> dict:
    rt.reset_mem()
    _, prompt_len, step_ms = cached_generate(rt, prompt, max_new_tokens)

    decode_ms = step_ms[1:]  # prefill is different work — keep it separate
    total_s = sum(step_ms) / 1000

    return {
        "milestone": 2,
        "label": label,
        "prompt_tokens": prompt_len,
        "generated_tokens": len(step_ms),
        "prefill_ms": round(step_ms[0], 2),
        "decode_median_ms": round(statistics.median(decode_ms), 2),
        "decode_p95_ms": round(sorted(decode_ms)[int(len(decode_ms) * 0.95)], 2),
        "decode_first_ms": round(decode_ms[0], 2),
        "decode_last_ms": round(decode_ms[-1], 2),
        "tokens_per_sec": round(len(step_ms) / total_s, 2),
        "peak_mem_gb": rt.peak_mem_gb(),
        "curve": [round(t, 1) for t in decode_ms[::16]],
    }
