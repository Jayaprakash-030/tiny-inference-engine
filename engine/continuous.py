"""Milestone 4 — continuous batching.

Block 1: workload — requests with staggered limits and random arrivals.
Block 2: single-seq prefill/decode (max_slots=1 path) must match cached_generate.
Block 3: batched decode — left-pad/stack B=1 caches, one forward, split back.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import DynamicCache

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


# ---------------------------------------------------------------------------
# Block 2 — single-sequence generate (no packing)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Block 3 — pack several B=1 caches into one decode step
# ---------------------------------------------------------------------------

def _seq_len(cache: DynamicCache) -> int:
    return cache.layers[0].keys.shape[2]


def _clone_cache_slice(cache: DynamicCache, start: int, end: int) -> DynamicCache:
    """B=1 DynamicCache keeping only positions [start:end]."""
    out = DynamicCache()
    for li, layer in enumerate(cache.layers):
        out.update(
            layer.keys[:, :, start:end, :],
            layer.values[:, :, start:end, :],
            li,
        )
    return out


def _stack_caches_left_pad(
    caches: list[DynamicCache],
) -> tuple[DynamicCache, list[int], int]:
    """Stack B=1 caches; left-pad K/V so real tokens align on the right."""
    lengths = [_seq_len(c) for c in caches]
    max_len = max(lengths)
    out = DynamicCache()
    for li in range(len(caches[0].layers)):
        ks, vs = [], []
        for c, L in zip(caches, lengths):
            k = c.layers[li].keys
            v = c.layers[li].values
            pad = max_len - L
            if pad:
                k = F.pad(k, (0, 0, pad, 0))
                v = F.pad(v, (0, 0, pad, 0))
            ks.append(k)
            vs.append(v)
        out.update(torch.cat(ks, dim=0), torch.cat(vs, dim=0), li)
    return out, lengths, max_len


def _split_caches_after_decode(
    batched: DynamicCache,
    lengths: list[int],
    max_len: int,
) -> list[DynamicCache]:
    """After decode, batched length is max_len+1; keep each seq's real suffix."""
    out = []
    for i, L in enumerate(lengths):
        # Left-padded past occupied [0, max_len); new token at max_len.
        # Real content for seq i is the last (L+1) positions.
        start = (max_len + 1) - (L + 1)
        sliced = DynamicCache()
        for li, layer in enumerate(batched.layers):
            sliced.update(
                layer.keys[i : i + 1, :, start:, :],
                layer.values[i : i + 1, :, start:, :],
                li,
            )
        out.append(sliced)
    return out


@torch.inference_mode()
def _decode_batch(rt: Runtime, active: list[ActiveSeq]) -> None:
    """One greedy decode step for all active sequences (possibly different L)."""
    if not active:
        return
    if len(active) == 1:
        _decode_one(rt, active[0])
        return

    B = len(active)
    next_ids = torch.cat([s.next_id for s in active], dim=0)  # [B, 1]
    stacked, lengths, max_len = _stack_caches_left_pad(
        [s.cache for s in active]
    )

    # Mask: 0 = left pad, 1 = real past + new token column.
    mask = torch.zeros(B, max_len + 1, dtype=torch.long, device=rt.device)
    pos = torch.zeros(B, 1, dtype=torch.long, device=rt.device)
    for i, L in enumerate(lengths):
        mask[i, max_len - L :] = 1
        pos[i, 0] = L

    out = rt.model(
        input_ids=next_ids,
        attention_mask=mask,
        position_ids=pos,
        past_key_values=stacked,
        use_cache=True,
    )
    new_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    split = _split_caches_after_decode(out.past_key_values, lengths, max_len)

    for i, s in enumerate(active):
        s.cache = split[i]
        s.next_id = new_ids[i : i + 1]
        s.tokens.append(int(new_ids[i].item()))
        s.n_gen += 1


@torch.inference_mode()
def run_batch(rt: Runtime, reqs: list[Request]) -> list[list[int]]:
    """Prefill each request alone, then decode together until all finish.

    Not yet a continuous scheduler — just exercises packed decode. Each request
    keeps generating until its own limit/EOS; finished ones are dropped from
    the active list (mini-preview of eviction).
    """
    active: list[ActiveSeq] = []
    for req in reqs:
        cache, next_id, tok = _prefill_one(rt, req.prompt)
        active.append(
            ActiveSeq(req=req, cache=cache, next_id=next_id, n_gen=1, tokens=[tok])
        )

    eos = rt.tokenizer.eos_token_id
    finished: dict[int, list[int]] = {}
    while active:
        # Drop anyone already done (after prefill or previous decode).
        still: list[ActiveSeq] = []
        for s in active:
            if _is_finished(s, eos):
                finished[s.req.req_id] = s.tokens
            else:
                still.append(s)
        active = still
        if not active:
            break
        _decode_batch(rt, active)

    return [finished[r.req_id] for r in reqs]


def check_decode_batch_matches_run_one(
    rt: Runtime,
    prompts: list[str] | None = None,
    n: int = 32,
) -> bool:
    """Packed multi-seq decode must match per-prompt run_one (greedy)."""
    prompts = prompts or BATCH[:3]
    reqs = [Request(i, p, n, arrival_ms=0.0) for i, p in enumerate(prompts)]

    batch_tokens = run_batch(rt, reqs)
    all_ok = True
    for req, got in zip(reqs, batch_tokens):
        solo = run_one(rt, req)
        ok = got == solo
        all_ok &= ok
        if not ok:
            print(f"  [{req.req_id}] MISMATCH  {req.prompt[:40]!r}")
            print(f"      batch: {rt.tokenizer.decode(got)[:80]!r}")
            print(f"      solo : {rt.tokenizer.decode(solo)[:80]!r}")
    print("decode_batch matches run_one:", all_ok)
    return all_ok


if __name__ == "__main__":
    print_workload(make_workload(8, max_new_tokens=200, arrival_rate_hz=8.0, seed=0))
    print()
    print("On GPU:")
    print("  from engine import load")
    print("  from engine.continuous import (")
    print("      check_run_one_matches_cached, check_decode_batch_matches_run_one)")
    print("  rt = load()")
    print("  assert check_run_one_matches_cached(rt)")
    print("  assert check_decode_batch_matches_run_one(rt)")
