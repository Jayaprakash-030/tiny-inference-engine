"""Append-only benchmark results, shared across milestones."""

import json
import statistics
from pathlib import Path

DEFAULT_PATH = Path("benchmarks/results.json")


def save(results: list[dict], path: Path = DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(existing + results, indent=2))
    print(f"appended {len(results)} results to {path}")


def load(path: Path = DEFAULT_PATH) -> list[dict]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else []


def by_milestone(n: int, path: Path = DEFAULT_PATH) -> list[dict]:
    return [r for r in load(path) if r.get("milestone") == n]


def report(result: dict, curve_key: str = "curve") -> dict:
    """Print a result dict, keeping the long curve on its own line."""
    print(f"--- {result.get('label')} ---")
    for k, v in result.items():
        if k != curve_key:
            print(f"{k:>22}: {v}")
    if curve_key in result:
        print(f"curve (every 16th step): {result[curve_key]}")
    print()
    return result


def summarize(step_ms: list[float]) -> dict:
    return {
        "median_ms": round(statistics.median(step_ms), 2),
        "p95_ms": round(sorted(step_ms)[int(len(step_ms) * 0.95)], 2),
        "first_ms": round(step_ms[0], 2),
        "last_ms": round(step_ms[-1], 2),
    }
