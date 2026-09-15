#!/usr/bin/env python3
"""Walk one batched generation step and print every intermediate tensor.

This is a teaching script for milestone 3 (static batching). It mirrors the
first prefill + first decode step inside engine/batched.py — deliberately
verbose so you can see *why* padding, masks, and position_ids matter.

Runs fine on CPU; it only generates a couple of tokens.

  python scripts/run_inspect_batch.py
"""

import torch

from engine import load

# Three prompts of different lengths. Left-padding will stretch the short ones
# so every row in the batch has the same length L.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain how a GPU executes a matrix multiplication.",
]


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def show(label: str, value) -> None:
    """Print a tensor (or anything) with a clear label."""
    print(f"\n--- {label} ---")
    print(value)


def main() -> None:
    banner("0. Load model + tokenizer")
    rt = load()
    tok = rt.tokenizer
    print(f"device={rt.device}  pad_token_id={tok.pad_token_id}  "
          f"eos_token_id={tok.eos_token_id}")

    # ------------------------------------------------------------------
    # 1. Tokenize each prompt alone — just to see the raw lengths differ.
    # ------------------------------------------------------------------
    banner("1. Per-prompt tokenization (no padding yet)")
    for i, p in enumerate(PROMPTS):
        ids = tok(p).input_ids
        print(f"[{i}] len={len(ids):>3}  {p!r}")
        print(f"     ids={ids}")

    # ------------------------------------------------------------------
    # 2. Batch tokenize with LEFT padding.
    #
    # Left padding is mandatory for decode: new tokens are appended at the
    # END of each row, so every sequence's "next-token slot" must line up at
    # the same column index (-1).
    # ------------------------------------------------------------------
    banner("2. Batched tokenize (left padding → shape [B, L])")
    enc = tok(PROMPTS, return_tensors="pt", padding=True)
    ids = enc.input_ids.to(rt.device)
    mask = enc.attention_mask.to(rt.device)
    B, L = ids.shape
    print(f"batch_size B={B}, padded_len L={L}")
    show("input_ids  (0 = pad)", ids)
    show("attention_mask  (1 = real token, 0 = pad)", mask)

    # ------------------------------------------------------------------
    # 3. Explicit position_ids.
    #
    # With left padding, a sequence's first REAL token is not at index 0.
    # Calling the model without position_ids would treat pads as positions
    # 0, 1, 2… and corrupt RoPE. We count only non-pad tokens:
    #   cumsum(mask) - 1, then clamp so pads stay at position 0.
    # ------------------------------------------------------------------
    banner("3. position_ids from the mask")
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    show("position_ids  (pads → 0; real tokens count up from 0)", pos)
    print("\nSanity: for each row, max(pos) should equal (#real tokens - 1)")
    for i in range(B):
        n_real = int(mask[i].sum())
        print(f"  row {i}: n_real={n_real}  max(pos)={int(pos[i].max())}")

    # ------------------------------------------------------------------
    # 4. PREFILL — one forward pass over the full padded prompts.
    #    Builds the KV cache for every layer / every real token.
    # ------------------------------------------------------------------
    banner("4. PREFILL  (full prompt → logits + KV cache)")
    with torch.inference_mode():
        out = rt.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            use_cache=True,
        )

    # logits: [B, L, vocab] — we only care about the LAST column (next token).
    print(f"logits shape: {tuple(out.logits.shape)}  "
          f"(B, L, vocab={out.logits.shape[-1]})")

    cache = out.past_key_values
    # Greedy pick: argmax over vocab at the last position of each row.
    next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
    show("next_ids (greedy token after prefill)", next_ids)
    print("decoded:")
    for i in range(B):
        print(f"  [{i}] {tok.decode(next_ids[i])!r}")

    # Peek at layer-0 of the cache: keys/values grew to length L.
    k0 = cache.layers[0].keys
    v0 = cache.layers[0].values
    show("layer-0 cache keys shape   [B, n_heads, L, head_dim]", k0.shape)
    show("layer-0 cache values shape [B, n_heads, L, head_dim]", v0.shape)

    # ------------------------------------------------------------------
    # 5. Grow the mask + compute the next position for DECODE.
    #
    # After prefill, every sequence has produced one new token. That token
    # is REAL for every row (even short ones), so we append a column of 1s
    # to the attention mask. next_pos = how many real tokens so far
    # (= mask.sum before we append — which equals the new token's position).
    # ------------------------------------------------------------------
    banner("5. Grow mask + next position_ids for decode")
    next_pos = mask.sum(-1, keepdim=True)  # [B, 1]
    ones = torch.ones(B, 1, dtype=mask.dtype, device=rt.device)
    mask = torch.cat([mask, ones], dim=-1)  # [B, L+1]

    show("next_pos  (position index of the newly generated token)", next_pos)
    show("ones we append to the mask", ones)
    show("mask after append  shape [B, L+1]", mask)
    print(f"mask is now length {mask.shape[-1]}  (was L={L})")

    # ------------------------------------------------------------------
    # 6. DECODE step 1 — feed ONLY the new token; reuse the KV cache.
    #    Cache length grows from L → L+1.
    # ------------------------------------------------------------------
    banner("6. DECODE step 1  (1 new token + past_key_values)")
    with torch.inference_mode():
        out = rt.model(
            input_ids=next_ids,
            attention_mask=mask,
            position_ids=next_pos,
            past_key_values=cache,
            use_cache=True,
        )

    cache = out.past_key_values
    next_ids_2 = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    show("logits shape after decode (note L dim is now 1)",
         tuple(out.logits.shape))
    show("next_ids after decode step 1", next_ids_2)
    print("decoded:")
    for i in range(B):
        print(f"  [{i}] {tok.decode(next_ids_2[i])!r}")

    k0 = cache.layers[0].keys
    show("layer-0 cache keys shape after decode  (L → L+1)", k0.shape)

    # ------------------------------------------------------------------
    # 7. Show the two generated tokens glued onto each prompt.
    # ------------------------------------------------------------------
    banner("7. Prompt + 2 generated tokens")
    generated = torch.cat([next_ids, next_ids_2], dim=-1)  # [B, 2]
    for i, prompt in enumerate(PROMPTS):
        cont = tok.decode(generated[i], skip_special_tokens=True)
        print(f"[{i}] {prompt!r}")
        print(f"     → {cont!r}")

    print("\nDone. Compare this walk-through to engine/batched.py "
          "(prefill block + first decode iteration).")


if __name__ == "__main__":
    main()
