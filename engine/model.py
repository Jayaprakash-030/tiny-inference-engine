"""Model loading and the shared runtime handle.

Loading happens in load(), not at import time, so importing this module is cheap
and has no side effects. Everything downstream takes a Runtime explicitly rather
than reaching for a global — that is what lets the same code run in a notebook
and in a headless script.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass
class Runtime:
    model: object
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    model_id: str

    def sync(self):
        """GPU work is queued asynchronously — without this, timers measure the
        queueing, not the compute."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def reset_mem(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_mem_gb(self):
        if self.device.type != "cuda":
            return None
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)

    def describe(self):
        print("model :", self.model_id)
        print("device:", self.device)
        print("dtype :", self.dtype)
        print("layers:", self.model.config.num_hidden_layers)
        if self.device.type == "cuda":
            print("gpu   :", torch.cuda.get_device_name(0))


def load(model_id: str = DEFAULT_MODEL, device: str | None = None) -> Runtime:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if dev.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(dev).eval()

    # Needed for batching (milestone 3). Harmless before then.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return Runtime(model=model, tokenizer=tokenizer, device=dev, dtype=dtype, model_id=model_id)
