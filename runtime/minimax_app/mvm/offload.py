"""Block-by-block GPU offload that works with torchao int8 tensors on Windows.

Why not diffusers' group offloading: its streamed mode page-locks a full copy of the weights (Windows refuses
30+ GB, and torchao Int8Tensor has no aten pin_memory), and its non-stream path cannot re-point torchao storage
across devices.

How this works: the large block lists (transformer blocks, decoder layers, vision blocks) stay in host RAM;
everything else goes to the GPU. A pre-forward hook copies a block's tensors to the GPU and *swaps* them into
the parameters (torch.utils.swap_tensors), keeping the original host tensors aside; the post-forward hook swaps
the host tensors back and drops the GPU copies. Host memory is therefore allocated exactly once, which matters
with a 32 GB model on a 64 GB machine.
"""
from __future__ import annotations

import torch
from torch import nn


def _block_lists(model: nn.Module, min_len: int = 8) -> list[nn.ModuleList]:
    lists = []
    for _, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) >= min_len:
            if any(any(m is sub for sub in big.modules()) for big in lists):
                continue
            lists.append(m)
    return lists


def _nbytes(module: nn.Module) -> int:
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        inner = getattr(t, "qdata", None)
        total += (inner.numel() * inner.element_size()) if inner is not None else t.numel() * t.element_size()
    return total


def _slots(block: nn.Module):
    """(module, kind, name) for every parameter and buffer owned directly by a submodule of `block`."""
    out = []
    for m in block.modules():
        out += [(m, "p", n) for n, _ in m.named_parameters(recurse=False)]
        out += [(m, "b", n) for n, b in m.named_buffers(recurse=False) if b is not None]
    return out


class _Streamer:
    def __init__(self, block: nn.Module, device: torch.device):
        self.device = device
        self.slots = _slots(block)
        self.host = None  # host tensors parked while the block is on the GPU

    def onload(self, mod, args):
        with torch.inference_mode(False), torch.no_grad():
            host = []
            for m, kind, name in self.slots:
                t = getattr(m, name)
                g = t.to(self.device)
                if kind == "p":
                    g = nn.Parameter(g, requires_grad=False)
                torch.utils.swap_tensors(t, g)  # t now lives on the GPU, g holds the host data
                host.append(g)
            self.host = host

    def offload(self, mod, args, out):
        with torch.inference_mode(False), torch.no_grad():
            for (m, kind, name), h in zip(self.slots, self.host):
                torch.utils.swap_tensors(getattr(m, name), h)  # host data back in place, h holds the GPU copy
            self.host = None  # dropping the GPU copies frees VRAM


def apply(model: nn.Module, device="cuda", resident_gb: float = 0.0, min_len: int = 8) -> dict:
    device = torch.device(device)
    torch.__future__.set_swap_module_params_on_conversion(True)  # Module.to() must swap tensor subclasses
    lists = _block_lists(model, min_len)
    blocks = [b for lst in lists for b in lst]
    block_ids = {id(m) for b in blocks for m in b.modules()}
    with torch.no_grad():
        for m in model.modules():  # everything outside the block lists lives on the GPU
            if id(m) not in block_ids:
                m._apply(lambda t: t.to(device), recurse=False)
    budget = resident_gb * 2**30
    streamed = 0
    for b in blocks:
        size = _nbytes(b)
        if budget >= size:
            b.to(device)
            budget -= size
            continue
        b.to("cpu")
        s = _Streamer(b, device)
        b.register_forward_pre_hook(s.onload)
        b.register_forward_hook(s.offload)
        streamed += 1
    return {"block_lists": [len(x) for x in lists], "blocks": len(blocks), "streamed": streamed}
