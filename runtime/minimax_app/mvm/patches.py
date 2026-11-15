"""Runtime fixes for Windows PyTorch builds. Import before running any model.

Windows wheels ship without FlashAttention. transformers asks SDPA for native GQA (`enable_gqa=True`) whenever
there is no attention mask, and on CUDA only the flash kernel supports that, so SDPA silently falls back to the
math kernel: for the Qwen3-VL conditioner (64 q / 8 kv heads, ~5k tokens) that is ~13 GB extra VRAM and ~50x
slower per layer. Repeating K/V instead lets the memory-efficient kernel run (measured 497 ms -> 9 ms).
"""
from __future__ import annotations

import sys


def apply() -> None:
    if sys.platform != "win32":
        return
    try:
        from transformers.integrations import sdpa_attention
        sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False
    except Exception:  # pragma: no cover - older/newer transformers layouts
        pass


apply()
