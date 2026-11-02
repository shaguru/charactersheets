"""Verify imports and execute real CUDA kernels before downloading large models."""
import sys
import torch
import torchvision
import diffusers
import transformers
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLModel
import accelerate
import peft
import einops
import scipy
import fastapi
import uvicorn
import multipart

if "--imports-only" not in sys.argv:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Update the NVIDIA driver and rerun START.bat.")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Qwen requires native BF16 support: use an NVIDIA Ampere or newer GPU (RTX 30/40/50 series).")
    with torch.inference_mode():
        sample = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
        result = sample @ sample
        torch.cuda.synchronize()
        if result[0, 0].item() != 32:
            raise RuntimeError("CUDA calculation returned an unexpected result.")
        del sample, result
    torch.cuda.empty_cache()
    print(f"PyTorch {torch.__version__}; {torch.cuda.get_device_name(0)}; "
          f"CUDA {torch.version.cuda}; Diffusers {diffusers.__version__}; CUDA calculation passed.")
