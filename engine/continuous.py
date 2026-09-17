"""Milestone 4 — continuous batching.

Block 1 (this file so far): the workload only — requests with staggered output
limits and random arrivals. Same list will later be served by static waves and
by iteration-level scheduling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from engine.batched import staggered_limits
from engine.prompts import batch_of


@dataclass
class Request:
    req_id: int
    prompt: str
    max_new_tokens: int
    arrival_ms: float  # simulated server time when the request shows up


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
        print(f"{r.req_id:>4}  {r.arrival_ms:>10.1f}  {r.max_new_tokens:>5}  {r.prompt[:48]!r}")


if __name__ == "__main__":
    print_workload(make_workload(8, max_new_tokens=200, arrival_rate_hz=8.0, seed=0))
