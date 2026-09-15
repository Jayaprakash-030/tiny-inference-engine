"""Milestone 3 — static batching.

Milestone 2 removed redundant work. This one amortizes unavoidable work.

The per-step floor (kernel launches for every layer, plus one full read of the
weights) is paid per STEP, not per token — the same whether 1 sequence or 32
ride along. Batching splits that fixed cost across the batch.

Three things change from cached_generate:
  1. Everything gains a batch dimension — inputs, logits, and the cache.
  2. An attention_mask tells the model to ignore padding.
  3. position_ids must be passed explicitly. With left padding a sequence's
     first real token is not at index 0, and calling the model directly skips
     the helper that would normally work this out.
"""

import statistics
import time

import torch

from engine.cached import cached_generate
from engine.model import Runtime
from engine.prompts import batch_of


@torch.inference_mode()
def batched_generate(rt: Runtime, prompts: list[str], max_new_tokens: int = 200):
    # Left padding is mandatory: decode appends at the END of each sequence, so
    # every sequence's next-token slot must line up at the same index.
    enc = rt.tokenizer(prompts, return_tensors="pt", padding=True)
    ids = enc.input_ids.to(rt.device)
    mask = enc.attention_mask.to(rt.device)
    batch_size = ids.shape[0]

    step_ms = []
    generated = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=rt.device)
    tokens_kept = 0      # useful tokens
    tokens_computed = 0  # slots computed, useful or not

    # --- PREFILL ---
    # Real positions: count only non-pad tokens, so padding does not advance.
    pos = (mask.cumsum(-1) - 1).clamp(min=0)

    rt.sync()
    t0 = time.perf_counter()

    out = rt.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    cache = out.past_key_values
    next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    rt.sync()
    step_ms.append((time.perf_counter() - t0) * 1000)

    generated.append(next_ids.clone())
    finished |= next_ids.squeeze(-1) == rt.tokenizer.eos_token_id
    tokens_kept += batch_size
    tokens_computed += batch_size

    next_pos = mask.sum(-1, keepdim=True)

    # --- DECODE ---
    for _ in range(max_new_tokens - 1):
        if finished.all():
            break

        # The mask grows by one real position each step, for every sequence.
        ones = torch.ones(batch_size, 1, dtype=mask.dtype, device=rt.device)
        mask = torch.cat([mask, ones], dim=-1)

        rt.sync()
        t0 = time.perf_counter()

        out = rt.model(input_ids=next_ids, attention_mask=mask,
                       position_ids=next_pos, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        rt.sync()
        step_ms.append((time.perf_counter() - t0) * 1000)

        # STATIC batching: finished sequences keep occupying their slot and keep
        # being computed. Count the waste rather than fixing it — fixing it is
        # milestone 4.
        tokens_computed += batch_size
        tokens_kept += int((~finished).sum())

        generated.append(next_ids.clone())
        finished |= next_ids.squeeze(-1) == rt.tokenizer.eos_token_id
        next_pos = next_pos + 1

    return torch.cat(generated, dim=-1), step_ms, tokens_kept, tokens_computed


def generation_lengths(out: torch.Tensor, eos_id: int, max_new_tokens: int) -> dict:
    """Per-sequence how many tokens were produced, and why each stopped.

    CAP  = ran to steps_run without ever emitting EOS (cap is binding if
           steps_run == max_new_tokens).
    EOS  = emitted eos_id; gen_tokens counts up to and including that token.
    """
    steps_run = out.shape[1]
    gen_tokens = []
    stopped = []
    for row in out:
        ids = row.tolist()
        if eos_id in ids:
            gen_tokens.append(ids.index(eos_id) + 1)
            stopped.append("EOS")
        else:
            gen_tokens.append(steps_run)
            stopped.append("CAP")

    return {
        "steps_run": steps_run,
        "max_new_tokens": max_new_tokens,
        "hit_cap": steps_run >= max_new_tokens,
        "gen_tokens": gen_tokens,
        "stopped": stopped,
        "n_hit_eos": sum(s == "EOS" for s in stopped),
        "n_hit_cap": sum(s == "CAP" for s in stopped),
        "min_gen_tokens": min(gen_tokens),
        "max_gen_tokens": max(gen_tokens),
    }


def print_generations(rt: Runtime, prompts: list[str], out: torch.Tensor,
                      max_new_tokens: int, preview: int = 120) -> dict:
    """Print each prompt's gen length, stop reason, and a short decode preview."""
    stats = generation_lengths(out, rt.tokenizer.eos_token_id, max_new_tokens)
    print(f"steps_run={stats['steps_run']}/{max_new_tokens}  "
          f"hit_cap={stats['hit_cap']}  "
          f"eos={stats['n_hit_eos']}/{len(prompts)}  "
          f"cap={stats['n_hit_cap']}/{len(prompts)}")
    for i, p in enumerate(prompts):
        n = stats["gen_tokens"][i]
        reason = stats["stopped"][i]
        text = rt.tokenizer.decode(out[i, :n], skip_special_tokens=True)
        print(f"  [{i}] gen_tokens={n:>4}  stopped={reason}  {p[:40]!r}")
        print(f"       {text[:preview]!r}")
    return stats


def check_batch_matches_single(rt: Runtime, prompts: list[str], n: int = 32) -> bool:
    """A sequence inside a batch must produce exactly what it produces alone.

    If padding or position_ids are wrong, output degrades silently.
    """
    batch_out, _, _, _ = batched_generate(rt, prompts, max_new_tokens=n)

    all_ok = True
    for i, prompt in enumerate(prompts):
        single_ids, prompt_len, _ = cached_generate(rt, prompt, max_new_tokens=n)
        single_out = single_ids[0, prompt_len:]
        row = batch_out[i][: len(single_out)]

        ok = torch.equal(row, single_out)
        all_ok &= ok
        if not ok:
            print(f"  [{i}] MISMATCH: {prompt[:40]}")
            print("      single:", rt.tokenizer.decode(single_out)[:80])
            print("      batch :", rt.tokenizer.decode(row)[:80])

    print("batch matches single:", all_ok)
    return all_ok


def measure_batch(rt: Runtime, batch_size: int, max_new_tokens: int = 200) -> dict:
    rt.reset_mem()
    prompts = batch_of(batch_size)
    out, step_ms, kept, computed = batched_generate(rt, prompts, max_new_tokens)
    lengths = generation_lengths(out, rt.tokenizer.eos_token_id, max_new_tokens)

    decode_ms = step_ms[1:]
    total_s = sum(step_ms) / 1000
    n_steps = len(step_ms)

    return {
        "milestone": 3,
        "label": f"batch_{batch_size}",
        "batch_size": batch_size,
        "steps": n_steps,
        "max_new_tokens": max_new_tokens,
        "hit_cap": lengths["hit_cap"],
        "n_hit_eos": lengths["n_hit_eos"],
        "n_hit_cap": lengths["n_hit_cap"],
        "min_gen_tokens": lengths["min_gen_tokens"],
        "max_gen_tokens": lengths["max_gen_tokens"],
        "gen_tokens": lengths["gen_tokens"],
        "prefill_ms": round(step_ms[0], 2),
        "decode_median_ms": round(statistics.median(decode_ms), 2) if decode_ms else None,
        # Throughput: useful tokens only, per second of wall clock.
        "tokens_per_sec": round(kept / total_s, 2),
        # What each sequence experiences, regardless of batch size.
        "per_seq_tok_per_sec": round(n_steps / total_s, 2),
        "wasted_slot_pct": round(100 * (computed - kept) / computed, 1),
        "peak_mem_gb": rt.peak_mem_gb(),
    }


def sweep(rt: Runtime, sizes=(1, 2, 4, 8, 16, 32), max_new_tokens: int = 200) -> list[dict]:
    """Stop at the first OOM — hitting the memory ceiling is a result, not a failure."""
    results = []
    for bs in sizes:
        try:
            r = measure_batch(rt, bs, max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs}: OOM — memory ceiling reached")
            torch.cuda.empty_cache()
            break
        results.append(r)
        print(f"batch {bs:>2}: {r['tokens_per_sec']:>8} tok/s total, "
              f"{r['per_seq_tok_per_sec']:>6} per seq, "
              f"{r['wasted_slot_pct']:>5}% wasted, "
              f"steps {r['steps']}/{max_new_tokens}, "
              f"eos {r['n_hit_eos']}/{bs}, "
              f"gen {r['min_gen_tokens']}-{r['max_gen_tokens']}, "
              f"{r['peak_mem_gb']} GB")
    return results
