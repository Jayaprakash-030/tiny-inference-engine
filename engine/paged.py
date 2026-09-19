"""Milestone 5 — paged KV cache.

Block 1: fixed-size block allocator (ids + free list).
Block 2: store real K/V in those blocks; gather → HF forward → scatter new token.
Block 3: shared pool budget — reserved (worst-case) vs paged (grow as you go).

We still call Hugging Face for the math. Paging owns *where* K/V live; before
each decode we gather the sequence's blocks into a temporary contiguous
DynamicCache, run the model, then write only the new token's K/V back into the
pool (allocating a new block when the last one is full).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from engine.cached import cached_generate
from engine.model import Runtime
from engine.prompts import BATCH


# Tokens stored in one physical block (vLLM's common default).
BLOCK_SIZE = 16


def blocks_needed(n_tokens: int, block_size: int = BLOCK_SIZE) -> int:
    """How many blocks are required to hold n_tokens (ceil division)."""
    if n_tokens <= 0:
        return 0
    return math.ceil(n_tokens / block_size)


@dataclass
class BlockPool:
    """Shared pool of physical KV blocks (ids + optional K/V tensor storage)."""

    n_blocks: int
    block_size: int = BLOCK_SIZE
    free: list[int] = field(init=False)
    # Per physical block, once used for real K/V:
    #   keys[bid][layer]  -> [1, n_kv_heads, block_size, head_dim]
    #   values[bid][layer] -> same
    #   filled[bid] -> how many of the block_size slots are used (0..block_size)
    keys: list[list[torch.Tensor] | None] = field(init=False)
    values: list[list[torch.Tensor] | None] = field(init=False)
    filled: list[int] = field(init=False)
    _meta: tuple | None = field(init=False, default=None)  # n_layers, H, D, dtype, device

    def __post_init__(self) -> None:
        if self.n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        self.free = list(range(self.n_blocks))
        self.keys = [None] * self.n_blocks
        self.values = [None] * self.n_blocks
        self.filled = [0] * self.n_blocks
        self._meta = None

    @property
    def n_free(self) -> int:
        return len(self.free)

    @property
    def n_used(self) -> int:
        return self.n_blocks - self.n_free

    def alloc(self, n: int = 1) -> list[int]:
        """Take n free block ids. Raises if the pool does not have enough."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return []
        if n > self.n_free:
            raise RuntimeError(
                f"out of blocks: need {n}, free {self.n_free} / {self.n_blocks}"
            )
        return [self.free.pop() for _ in range(n)]

    def free_blocks(self, ids: list[int]) -> None:
        """Return physical block ids to the free list and drop their tensors."""
        for i in ids:
            if i < 0 or i >= self.n_blocks:
                raise ValueError(f"invalid block id {i}")
            if i in self.free:
                raise ValueError(f"block {i} is already free")
            self.keys[i] = None
            self.values[i] = None
            self.filled[i] = 0
            self.free.append(i)

    def _remember_shape(self, cache: DynamicCache) -> None:
        layer0 = cache.layers[0].keys  # [1, H, L, D]
        self._meta = (
            len(cache.layers),
            layer0.shape[1],
            layer0.shape[3],
            layer0.dtype,
            layer0.device,
        )

    def _alloc_tensor_block(self, bid: int) -> None:
        assert self._meta is not None
        n_layers, H, D, dtype, device = self._meta
        self.keys[bid] = [
            torch.zeros(1, H, self.block_size, D, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.values[bid] = [
            torch.zeros(1, H, self.block_size, D, dtype=dtype, device=device)
            for _ in range(n_layers)
        ]
        self.filled[bid] = 0

    def write_cache_slice(self, bid: int, cache: DynamicCache, start: int, end: int) -> None:
        """Copy cache positions [start:end) into physical block bid (fills from slot 0)."""
        if self._meta is None:
            self._remember_shape(cache)
        if self.keys[bid] is None:
            self._alloc_tensor_block(bid)
        n = end - start
        if n > self.block_size:
            raise ValueError(f"slice length {n} > block_size {self.block_size}")
        for li, layer in enumerate(cache.layers):
            self.keys[bid][li][:, :, :n, :] = layer.keys[:, :, start:end, :]
            self.values[bid][li][:, :, :n, :] = layer.values[:, :, start:end, :]
        self.filled[bid] = n

    def append_token_kv(self, bid: int, cache: DynamicCache, pos: int = -1) -> None:
        """Write one token's K/V (cache position pos) into the next free slot of bid."""
        if self._meta is None:
            self._remember_shape(cache)
        if self.keys[bid] is None:
            self._alloc_tensor_block(bid)
        slot = self.filled[bid]
        if slot >= self.block_size:
            raise RuntimeError(f"block {bid} is full")
        # Important: do NOT use [:, :, -1:0, :] — in Python that slice is empty.
        L = cache.layers[0].keys.shape[2]
        idx = pos if pos >= 0 else L + pos
        for li, layer in enumerate(cache.layers):
            self.keys[bid][li][:, :, slot : slot + 1, :] = layer.keys[:, :, idx : idx + 1, :]
            self.values[bid][li][:, :, slot : slot + 1, :] = layer.values[:, :, idx : idx + 1, :]
        self.filled[bid] = slot + 1

    def gather_to_cache(self, table: list[int], seq_len: int) -> DynamicCache:
        """Concatenate used slots of blocks in table order → contiguous DynamicCache."""
        if seq_len == 0:
            raise ValueError("cannot gather empty sequence")
        assert self._meta is not None
        n_layers = self._meta[0]
        out = DynamicCache()
        for li in range(n_layers):
            pieces_k, pieces_v = [], []
            remaining = seq_len
            for bid in table:
                take = min(self.filled[bid], remaining)
                pieces_k.append(self.keys[bid][li][:, :, :take, :])
                pieces_v.append(self.values[bid][li][:, :, :take, :])
                remaining -= take
                if remaining == 0:
                    break
            if remaining != 0:
                raise RuntimeError("block table shorter than seq_len")
            out.update(torch.cat(pieces_k, dim=2), torch.cat(pieces_v, dim=2), li)
        return out


@dataclass
class BlockTable:
    """Per-sequence map: logical block index → physical block id."""

    pool: BlockPool
    table: list[int] = field(default_factory=list)
    seq_len: int = 0

    @property
    def n_blocks(self) -> int:
        return len(self.table)

    def ensure_capacity(self, n_tokens: int) -> None:
        need = blocks_needed(n_tokens, self.pool.block_size)
        extra = need - len(self.table)
        if extra > 0:
            self.table.extend(self.pool.alloc(extra))

    def append_tokens(self, n: int = 1) -> None:
        """Block-1 helper: advance length only (no K/V). Prefer ingest/append_kv for Block 2."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self.ensure_capacity(self.seq_len + n)
        self.seq_len += n

    def release(self) -> None:
        self.pool.free_blocks(list(self.table))
        self.table.clear()
        self.seq_len = 0

    def waste_slots(self) -> int:
        if self.seq_len == 0:
            return 0
        return self.n_blocks * self.pool.block_size - self.seq_len

    # ----- Block 2: real K/V -----

    def ingest_from_cache(self, cache: DynamicCache) -> None:
        """Split a contiguous HF cache into pool blocks; replace this table."""
        L = cache.layers[0].keys.shape[2]
        self.release()
        self.ensure_capacity(L)
        bs = self.pool.block_size
        for bi, bid in enumerate(self.table):
            start = bi * bs
            end = min(start + bs, L)
            self.pool.write_cache_slice(bid, cache, start, end)
        self.seq_len = L

    def gather_cache(self) -> DynamicCache:
        return self.pool.gather_to_cache(self.table, self.seq_len)

    def append_kv_from_cache(self, cache: DynamicCache) -> None:
        """After a decode step, store the newest token (last position) into the pool."""
        self.ensure_capacity(self.seq_len + 1)
        # Next token lands in logical block index seq_len // block_size — NOT
        # always table[-1]. Reserved mode pre-allocates empty tail blocks, so
        # the last table entry may be an unused future block.
        bi = self.seq_len // self.pool.block_size
        bid = self.table[bi]
        if self.pool.filled[bid] >= self.pool.block_size:
            raise RuntimeError(
                f"block {bid} (logical {bi}) full at seq_len={self.seq_len}; "
                f"table={self.table} filled={self.pool.filled[bid]}"
            )
        self.pool.append_token_kv(bid, cache, pos=-1)
        self.seq_len += 1


def print_pool(pool: BlockPool, label: str = "") -> None:
    prefix = f"{label}: " if label else ""
    print(
        f"{prefix}blocks={pool.n_blocks}  "
        f"free={pool.n_free}  used={pool.n_used}  "
        f"block_size={pool.block_size}"
    )


# ---------------------------------------------------------------------------
# Block 2 — single-sequence paged generate
# ---------------------------------------------------------------------------

@torch.inference_mode()
def paged_run_one(
    rt: Runtime,
    prompt: str,
    max_new_tokens: int = 32,
    n_blocks: int = 64,
    block_size: int = BLOCK_SIZE,
) -> list[int]:
    """Greedy generate using paged KV storage; HF still runs the forward pass."""
    pool = BlockPool(n_blocks=n_blocks, block_size=block_size)
    bt = BlockTable(pool)
    eos = rt.tokenizer.eos_token_id

    # Prefill: one contiguous forward, then split K/V into blocks.
    ids = rt.tokenizer(prompt, return_tensors="pt").input_ids.to(rt.device)
    out = rt.model(input_ids=ids, use_cache=True)
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [int(next_id.item())]
    bt.ingest_from_cache(out.past_key_values)

    # Decode: gather → one-token forward → scatter new K/V into pool.
    for _ in range(max_new_tokens - 1):
        if tokens[-1] == eos:
            break
        gathered = bt.gather_cache()
        out = rt.model(
            input_ids=next_id,
            past_key_values=gathered,
            use_cache=True,
        )
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = int(next_id.item())
        tokens.append(tok)
        # past_key_values now length seq_len+1 (gathered history + new token).
        bt.append_kv_from_cache(out.past_key_values)
        if tok == eos:
            break

    bt.release()
    return tokens


def check_paged_matches_cached(
    rt: Runtime,
    prompt: str = BATCH[0],
    n: int = 32,
) -> bool:
    """Paged greedy decode must match cached_generate token-for-token."""
    paged = paged_run_one(rt, prompt, max_new_tokens=n)
    cached_ids, prompt_len, _ = cached_generate(rt, prompt, max_new_tokens=n)
    cached_gen = cached_ids[0, prompt_len:].tolist()
    same = paged == cached_gen
    print("paged run_one matches cached:", same)
    if not same:
        print("  paged :", rt.tokenizer.decode(paged)[:120])
        print("  cached:", rt.tokenizer.decode(cached_gen)[:120])
        print(f"  lens: paged={len(paged)} cached={len(cached_gen)}")
    return same


# ---------------------------------------------------------------------------
# Block 3 — memory budget: reserved (worst-case) vs paged (grow as you go)
# ---------------------------------------------------------------------------

@dataclass
class _Live:
    """One in-flight request under a shared BlockPool."""

    req: object  # continuous.Request
    bt: BlockTable
    next_id: torch.Tensor
    tokens: list[int]
    n_gen: int
    prompt_len: int


def _prompt_len(rt: Runtime, prompt: str) -> int:
    return rt.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]


def reserved_blocks_needed(prompt_len: int, max_new_tokens: int,
                           block_size: int = BLOCK_SIZE) -> int:
    """Pre-vLLM style: reserve for prompt + full max output up front."""
    return blocks_needed(prompt_len + max_new_tokens, block_size)


def paged_prefill_blocks_needed(prompt_len: int, block_size: int = BLOCK_SIZE) -> int:
    """Paged admit only needs room for the prompt KV (grows later on decode)."""
    return blocks_needed(prompt_len, block_size)


@torch.inference_mode()
def serve_with_block_budget(
    rt: Runtime,
    requests: list,
    n_blocks: int,
    mode: str = "paged",
    block_size: int = BLOCK_SIZE,
    max_slots: int | None = None,
) -> dict:
    """Run continuous-style admit/decode/evict under a fixed block pool.

    mode="reserved": on admit, allocate blocks for prompt+max_new immediately
                     (empty tail held so nobody else can use them).
    mode="paged":    on admit, allocate only what prefill needs; grow one block
                     at a time during decode; free everything on finish.

    Requests that cannot be admitted when they arrive are rejected (not queued
    forever). In paged mode a sequence may also be preempted mid-decode if the
    pool has no free block when its KV needs to grow — that is the tight-budget
    story, not a crash.
    """
    if mode not in ("paged", "reserved"):
        raise ValueError("mode must be 'paged' or 'reserved'")

    from engine.continuous import Request  # local import: avoid cycle at module load

    pool = BlockPool(n_blocks=n_blocks, block_size=block_size)
    eos = rt.tokenizer.eos_token_id
    waiting = sorted(requests, key=lambda r: (r.arrival_ms, r.req_id))
    active: list[_Live] = []
    completed: list[dict] = []
    rejected: list[dict] = []

    server_ms = 0.0
    peak_used = 0
    peak_concurrent = 0
    qi = 0
    slot_cap = max_slots if max_slots is not None else n_blocks  # soft cap

    if waiting:
        server_ms = max(server_ms, waiting[0].arrival_ms)

    def _note_peaks():
        nonlocal peak_used, peak_concurrent
        peak_used = max(peak_used, pool.n_used)
        peak_concurrent = max(peak_concurrent, len(active))

    def _try_admit(req: Request) -> bool:
        """Prefill + place into active, or reject if the budget cannot fit."""
        nonlocal server_ms
        plen = _prompt_len(rt, req.prompt)
        if mode == "reserved":
            need = reserved_blocks_needed(plen, req.max_new_tokens, block_size)
        else:
            need = paged_prefill_blocks_needed(plen, block_size)

        if pool.n_free < need or len(active) >= slot_cap:
            return False

        rt.sync()
        t0 = time.perf_counter()
        ids = rt.tokenizer(req.prompt, return_tensors="pt").input_ids.to(rt.device)
        out = rt.model(input_ids=ids, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        rt.sync()
        server_ms += (time.perf_counter() - t0) * 1000

        bt = BlockTable(pool)
        bt.ingest_from_cache(out.past_key_values)
        # Reserved: pin blocks for the worst-case final length now.
        if mode == "reserved":
            bt.ensure_capacity(plen + req.max_new_tokens)

        live = _Live(
            req=req,
            bt=bt,
            next_id=next_id,
            tokens=[int(next_id.item())],
            n_gen=1,
            prompt_len=plen,
        )
        if live.tokens[-1] == eos or live.n_gen >= req.max_new_tokens:
            completed.append({
                "req_id": req.req_id,
                "gen_tokens": live.n_gen,
                "blocks_used": live.bt.n_blocks,
            })
            live.bt.release()
        else:
            active.append(live)
        _note_peaks()
        return True

    def _can_append_one(live: _Live) -> bool:
        """True if writing one more KV token fits without a new free block, or one is free."""
        need = blocks_needed(live.bt.seq_len + 1, block_size)
        extra = need - live.bt.n_blocks
        return extra <= 0 or pool.n_free >= extra

    def _decode_one_live(live: _Live) -> float:
        rt.sync()
        t0 = time.perf_counter()
        gathered = live.bt.gather_cache()
        out = rt.model(
            input_ids=live.next_id,
            past_key_values=gathered,
            use_cache=True,
        )
        rt.sync()
        dt = (time.perf_counter() - t0) * 1000
        live.next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok = int(live.next_id.item())
        live.tokens.append(tok)
        live.n_gen += 1
        live.bt.append_kv_from_cache(out.past_key_values)
        return dt

    def _finished(live: _Live) -> bool:
        if live.n_gen >= live.req.max_new_tokens:
            return True
        if live.tokens and live.tokens[-1] == eos:
            return True
        return False

    def _preempt(live: _Live, reason: str) -> None:
        rejected.append({
            "req_id": live.req.req_id,
            "arrival_ms": live.req.arrival_ms,
            "limit": live.req.max_new_tokens,
            "reason": reason,
            "gen_tokens": live.n_gen,
            "free_at_reject": pool.n_free,
        })
        live.bt.release()

    while qi < len(waiting) or active:
        # ADMIT
        while qi < len(waiting) and len(active) < slot_cap:
            req = waiting[qi]
            if req.arrival_ms > server_ms:
                break
            qi += 1
            if not _try_admit(req):
                rejected.append({
                    "req_id": req.req_id,
                    "arrival_ms": req.arrival_ms,
                    "limit": req.max_new_tokens,
                    "reason": "oom_blocks_or_slots",
                    "free_at_reject": pool.n_free,
                })

        if not active:
            if qi >= len(waiting):
                break
            server_ms = max(server_ms, waiting[qi].arrival_ms)
            continue

        # DECODE + EVICT (sequential; memory story is what matters here).
        # Paged can over-admit relative to max length; if a seq needs a new
        # block and the pool is empty, preempt it and free its blocks.
        still: list[_Live] = []
        for live in active:
            if not _can_append_one(live):
                _preempt(live, "oom_grow")
                continue
            server_ms += _decode_one_live(live)
            if _finished(live):
                completed.append({
                    "req_id": live.req.req_id,
                    "gen_tokens": live.n_gen,
                    "blocks_used": live.bt.n_blocks,
                    "waste_slots": live.bt.waste_slots(),
                })
                live.bt.release()
            else:
                still.append(live)
        active = still
        _note_peaks()

    # Anything never reached because we stopped? (shouldn't happen)
    return {
        "milestone": 5,
        "label": f"{mode}_blocks_{n_blocks}",
        "mode": mode,
        "n_blocks": n_blocks,
        "block_size": block_size,
        "n_requests": len(requests),
        "completed": len(completed),
        "rejected": len(rejected),
        "peak_blocks_used": peak_used,
        "peak_concurrent": peak_concurrent,
        "total_s": round(server_ms / 1000, 3),
        "completions": completed,
        "rejections": rejected,
    }


def _budget_result_row(d: dict, *, max_new_tokens: int, arrival_rate_hz: float,
                       seed: int, source: str | None = None) -> dict:
    """Compact row for benchmarks/results.json (drop per-request lists)."""
    row = {
        "milestone": 5,
        "label": d["label"],
        "mode": d["mode"],
        "n_blocks": d["n_blocks"],
        "block_size": d["block_size"],
        "n_requests": d["n_requests"],
        "max_new_tokens": max_new_tokens,
        "arrival_rate_hz": arrival_rate_hz,
        "seed": seed,
        "completed": d["completed"],
        "rejected": d["rejected"],
        "peak_concurrent": d["peak_concurrent"],
        "peak_blocks_used": d["peak_blocks_used"],
        "total_s": d["total_s"],
    }
    if source:
        row["source"] = source
    return row


def compare_block_budget(
    rt: Runtime,
    n_blocks: int = 32,
    n_requests: int = 16,
    max_new_tokens: int = 64,
    arrival_rate_hz: float = 8.0,
    seed: int = 0,
    block_size: int = BLOCK_SIZE,
) -> tuple[dict, dict]:
    """Same workload + same pool size: reserved vs paged admission.

    Returns two compact result dicts ready for engine.results.save().
    """
    from engine.continuous import make_workload, print_workload

    workload = make_workload(
        n_requests, max_new_tokens, arrival_rate_hz, seed, stagger=True,
    )
    print(f"budget: {n_blocks} blocks × {block_size} tokens "
          f"= {n_blocks * block_size} token-slots")
    print_workload(workload)

    reserved_raw = serve_with_block_budget(
        rt, workload, n_blocks=n_blocks, mode="reserved", block_size=block_size,
    )
    paged_raw = serve_with_block_budget(
        rt, workload, n_blocks=n_blocks, mode="paged", block_size=block_size,
    )

    reserved = _budget_result_row(
        reserved_raw, max_new_tokens=max_new_tokens,
        arrival_rate_hz=arrival_rate_hz, seed=seed,
    )
    paged = _budget_result_row(
        paged_raw, max_new_tokens=max_new_tokens,
        arrival_rate_hz=arrival_rate_hz, seed=seed,
    )

    def _line(tag: str, d: dict) -> None:
        print(
            f"  {tag:<10}: completed {d['completed']:>2}/{d['n_requests']}  "
            f"rejected {d['rejected']:>2}  "
            f"peak_conc {d['peak_concurrent']:>2}  "
            f"peak_blocks {d['peak_blocks_used']:>3}/{d['n_blocks']}"
        )

    print(f"--- n_blocks={n_blocks}  n_requests={n_requests}  max_new={max_new_tokens} ---")
    _line("reserved", reserved)
    _line("paged", paged)
    return reserved, paged


def sweep_block_budget(
    rt: Runtime,
    block_budgets=(16, 32, 48, 64),
    n_requests: int = 16,
    max_new_tokens: int = 64,
    arrival_rate_hz: float = 8.0,
    seed: int = 0,
    block_size: int = BLOCK_SIZE,
) -> list[dict]:
    """Compare reserved vs paged across several pool sizes; rows for save()."""
    results: list[dict] = []
    for n_blocks in block_budgets:
        reserved, paged = compare_block_budget(
            rt,
            n_blocks=n_blocks,
            n_requests=n_requests,
            max_new_tokens=max_new_tokens,
            arrival_rate_hz=arrival_rate_hz,
            seed=seed,
            block_size=block_size,
        )
        results.extend([reserved, paged])
    return results


if __name__ == "__main__":
    pool = BlockPool(n_blocks=10, block_size=16)
    print_pool(pool, "start")

    seq = BlockTable(pool)
    seq.append_tokens(20)
    print(f"after prefill 20: table={seq.table} seq_len={seq.seq_len} "
          f"waste={seq.waste_slots()}")
    print_pool(pool, "after prefill")

    seq.append_tokens(13)
    print(f"after grow to 33: table={seq.table} seq_len={seq.seq_len} "
          f"waste={seq.waste_slots()}")
    print_pool(pool, "after grow")

    seq.release()
    print(f"after release: table={seq.table} seq_len={seq.seq_len}")
    print_pool(pool, "after release")

    print()
    print("On GPU:")
    print("  from engine import load")
    print("  from engine.paged import check_paged_matches_cached, compare_block_budget")
    print("  from engine.results import save")
    print("  rt = load()")
    print("  assert check_paged_matches_cached(rt)")
    print("  r, p = compare_block_budget(rt, n_blocks=32, n_requests=16, max_new_tokens=64)")
    print("  save([r, p])")
