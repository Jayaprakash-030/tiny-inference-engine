"""Milestone 1 — naive generation loop, no KV cache.

Every step pushes the entire sequence through the model again and throws all of
it away. Step 500 redoes the work of steps 1-499. This is the baseline every
later milestone is measured against.
"""

import statistics
import time

import torch

from engine.model import Runtime


@torch.inference_mode()
def naive_generate(rt: Runtime, prompt: str, max_new_tokens: int = 400):
    ids = rt.tokenizer(prompt, return_tensors="pt").input_ids.to(rt.device)
    prompt_len = ids.shape[1]
    step_ms = []

    for _ in range(max_new_tokens):
        rt.sync()
        t0 = time.perf_counter()

        # use_cache=False is the whole point: no cache, full recompute.
        out = rt.model(input_ids=ids, use_cache=False)

        # Only the last position predicts the next token. The rest is wasted work.
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=-1)

        rt.sync()
        step_ms.append((time.perf_counter() - t0) * 1000)

        if next_id.item() == rt.tokenizer.eos_token_id:
            break

    return ids, prompt_len, step_ms


def measure_naive(rt: Runtime, label: str, prompt: str, max_new_tokens: int = 400) -> dict:
    rt.reset_mem()
    _, prompt_len, step_ms = naive_generate(rt, prompt, max_new_tokens)
    total_s = sum(step_ms) / 1000

    return {
        "milestone": 1,
        "label": label,
        "prompt_tokens": prompt_len,
        "generated_tokens": len(step_ms),
        "first_step_ms": round(step_ms[0], 2),
        "last_step_ms": round(step_ms[-1], 2),
        "median_step_ms": round(statistics.median(step_ms), 2),
        "tokens_per_sec": round(len(step_ms) / total_s, 2),
        "peak_mem_gb": rt.peak_mem_gb(),
        "curve": [round(t, 1) for t in step_ms[::16]],
    }


def warmup(rt: Runtime, prompt: str, steps: int = 20) -> None:
    """First CUDA calls pay one-time setup that would otherwise inflate the baseline."""
    naive_generate(rt, prompt, max_new_tokens=steps)
