"""Milestone 5 — paged KV cache.

Block 1: fixed-size block allocator (ids + free list).
Block 2: store real K/V in those blocks; gather → HF forward → scatter new token.

We still call Hugging Face for the math. Paging owns *where* K/V live; before
each decode we gather the sequence's blocks into a temporary contiguous
DynamicCache, run the model, then write only the new token's K/V back into the
pool (allocating a new block when the last one is full).
"""

from __future__ import annotations

import math
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
        last_bid = self.table[-1]
        # If the previous last block was full, ensure_capacity just appended a fresh id.
        if self.pool.filled[last_bid] >= self.pool.block_size:
            raise RuntimeError(
                f"last block {last_bid} still full after ensure_capacity; table={self.table}"
            )
        self.pool.append_token_kv(last_bid, cache, pos=-1)
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
    print("On GPU, verify Block 2:")
    print("  from engine import load")
    print("  from engine.paged import check_paged_matches_cached")
    print("  assert check_paged_matches_cached(load())")
