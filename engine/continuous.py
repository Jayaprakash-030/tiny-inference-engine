"""Milestone 4 — continuous batching.

Block 1: workload — requests with staggered limits and random arrivals.
Block 2: single-seq prefill/decode (max_slots=1 path) must match cached_generate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from engine.batched import staggered_limits
from engine.cached import cached_generate
from engine.model import Runtime
from engine.prompts import BATCH, batch_of


@dataclass
class Request:
    req_id: int
    prompt: str
    max_new_tokens: int
    arrival_ms: float  # simulated server time when the request shows up


@dataclass
class ActiveSeq:
    """One in-flight sequence with its own B=1 KV cache."""

    req: Request
    cache: object
    next_id: torch.Tensor  # [1, 1]
    n_gen: int
    tokens: list[int] = field(default_factory=list)


def make_workload(
    n_requests: int,
    max_new_tokens: int = 200,
    arrival_rate_hz: float = 8.0,
    seed: int = 0,
    stagger: bool = True,
) -> list[Request]:
    """Build prompts + per-request limits + exponential arrivals.

    stagger=True spreads stop lengths the same way milestone 3 does, then
    shuffles so short limits are not tied to early arrivals.
    """
    rng = random.Random(seed)
    prompts = batch_of(n_requests)

    if stagger and n_requests > 1:
        limits = staggered_limits(n_requests, max_new_tokens)
        rng.shuffle(limits)
    else:
        limits = [max_new_tokens] * n_requests

    requests: list[Request] = []
    t = 0.0
    for i in range(n_requests):
        requests.append(Request(i, prompts[i], limits[i], t))
        # Mean inter-arrival = 1000 / rate ms.
        t += rng.expovariate(arrival_rate_hz) * 1000.0
    return requests


def print_workload(requests: list[Request]) -> None:
    print(f"{'id':>4}  {'arrival_ms':>10}  {'limit':>5}  prompt")
    for r in requests:
        print(
            f"{r.req_id:>4}  {r.arrival_ms:>10.1f}  {r.max_new_tokens:>5}  {r.prompt[:48]!r}"
        )


@torch.inference_mode()
def _prefill_one(rt: Runtime, prompt: str) -> tuple[object, torch.Tensor, int]:
    """Prefill one prompt. Returns cache, next_id [1,1], first token id."""
    ids = rt.tokenizer(prompt, return_tensors="pt").input_ids.to(rt.device)
    out = rt.model(input_ids=ids, use_cache=True)
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return out.past_key_values, next_id, int(next_id.item())


@torch.inference_mode()
def _decode_one(rt: Runtime, seq: ActiveSeq) -> None:
    """One greedy decode step for a B=1 sequence; updates seq in place."""
    out = rt.model(
        input_ids=seq.next_id,
        past_key_values=seq.cache,
        use_cache=True,
    )
    seq.cache = out.past_key_values
    seq.next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tok = int(seq.next_id.item())
    seq.tokens.append(tok)
    seq.n_gen += 1


def _is_finished(seq: ActiveSeq, eos_id: int) -> bool:
    if seq.n_gen >= seq.req.max_new_tokens:
        return True
    if seq.tokens and seq.tokens[-1] == eos_id:
        return True
    return False


@torch.inference_mode()
def run_one(rt: Runtime, req: Request) -> list[int]:
    """Generate for a single request (the max_slots=1 continuous path)."""
    cache, next_id, tok = _prefill_one(rt, req.prompt)
    seq = ActiveSeq(req=req, cache=cache, next_id=next_id, n_gen=1, tokens=[tok])
    eos = rt.tokenizer.eos_token_id
    while not _is_finished(seq, eos):
        _decode_one(rt, seq)
    return seq.tokens


def check_run_one_matches_cached(
    rt: Runtime,
    prompt: str = BATCH[0],
    n: int = 32,
) -> bool:
    """Greedy run_one must match cached_generate token-for-token."""
    req = Request(0, prompt, n, arrival_ms=0.0)
    cont = run_one(rt, req)
    cached_ids, prompt_len, _ = cached_generate(rt, prompt, max_new_tokens=n)
    cached_gen = cached_ids[0, prompt_len:].tolist()
    same = cont == cached_gen
    print("continuous run_one matches cached:", same)
    if not same:
        print("  cont  :", rt.tokenizer.decode(cont)[:120])
        print("  cached:", rt.tokenizer.decode(cached_gen)[:120])
    return same


if __name__ == "__main__":
    print_workload(make_workload(8, max_new_tokens=200, arrival_rate_hz=8.0, seed=0))
    print()
    print("To verify Block 2 on GPU:")
    print("  from engine import load")
    print("  from engine.continuous import check_run_one_matches_cached")
    print("  assert check_run_one_matches_cached(load())")
