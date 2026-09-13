"""Fixed prompt set, so every milestone benchmarks the same work."""

SHORT = "The capital of France is"
LONG = "Explain how a GPU executes a matrix multiplication. " * 25

BATCH = [
    "The capital of France is",
    "Explain how a GPU executes a matrix multiplication.",
    "In 2026, the most important trend in machine learning is",
    "def fibonacci(n):",
    "The three laws of thermodynamics are",
    "A transformer model processes text by",
    "The difference between a CPU and a GPU is",
    "Once upon a time in a small village",
]


def batch_of(n: int) -> list[str]:
    return [BATCH[i % len(BATCH)] for i in range(n)]
