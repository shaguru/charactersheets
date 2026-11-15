"""MiniMax-H3-Image generation and reference editing.

Default: native GGUF, with CPU offload, no full-precision conversion scratch copy.
Optional legacy Diffusers backend: reuse the video's int8 encoder, quantize the
official image transformer once, then run text/denoising in separate passes.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

from mvm import models, patches, settings  # noqa: F401 (patches applies fixes)

TEXT_STAGE = ("before_encode", "text_encoder", "input", "setup", "resize")


def _split(blocks, first: bool):
    names = list(blocks.sub_blocks)
    cut = names.index("text_encoder") + 1
    for k in (names[cut:] if first else ["text_encoder"]):
        blocks.sub_blocks.pop(k)
    return blocks


def _offload(model, budget_key: str, default_gb: float):
    from mvm import offload
    return offload.apply(model, resident_gb=float(settings.load().get(budget_key, default_gb)))


def _transformer(img_repo: str):
    import torch
    from diffusers import MiniMaxH3ImageTransformer2DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    q_dir = settings.models_dir() / "MiniMax-H3-Image-transformer-int8"
    if (q_dir / "config.json").exists():
        return MiniMaxH3ImageTransformer2DModel.from_pretrained(str(q_dir), dtype=torch.bfloat16, use_safetensors=False)
    print("First run: quantizing the H3-Image transformer to int8 (one-off, several minutes)...")
    tr = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        img_repo, subfolder="transformer_diffusers", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2),
                                          modules_to_not_convert=["proj_in", "context_embedder", "time_embedder",
                                                                  "time_proj", "token_refiner", "norm_out", "proj_out"]))
    tr.save_pretrained(str(q_dir), safe_serialization=False)
    return tr


def generate(prompt: str, images: list[Path], out_dir: Path, size=(2720, 1536), count: int = 1,
             stem: str = "sheet", seed: int | None = None) -> list[Path]:
    if settings.load().get("image_backend", "gguf") == "gguf":
        from mvm import image_native
        return image_native.generate(prompt, images, out_dir, size, count, stem, seed)
    import torch
    from diffusers import ModularPipeline
    from diffusers.utils import load_image
    from transformers import Qwen3VLForConditionalGeneration

    img_dir = models.require("image")  # repo ids (loaded from the local HF cache)
    vid_dir = models.require("video")
    refs = [load_image(str(i)) for i in images]
    w, h = size

    # pass 1: text encoder
    enc = _split(ModularPipeline.from_pretrained(str(img_dir)).blocks, first=True).init_pipeline(str(img_dir))
    te = Qwen3VLForConditionalGeneration.from_pretrained(str(vid_dir), subfolder="text_encoder", dtype=torch.bfloat16)
    te.requires_grad_(False)
    enc.update_components(text_encoder=te)
    enc.load_components(dtype=torch.bfloat16)  # tokenizer / processor / image_processor
    _offload(enc.text_encoder, "te_resident_gb", 16)
    with torch.no_grad():
        st = enc(prompt=prompt, image=refs, output=["prompt_embeds", "text_token_tags"])
    embeds = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in st.items()}
    del enc, te
    gc.collect()
    torch.cuda.empty_cache()

    # pass 2: transformer + VAE
    pipe = _split(ModularPipeline.from_pretrained(str(img_dir)).blocks, first=False).init_pipeline(str(img_dir))
    tr = _transformer(img_dir)  # img_dir is the repo id
    tr.requires_grad_(False)
    pipe.update_components(transformer=tr)
    pipe.load_components(dtype=torch.bfloat16)
    _offload(pipe.transformer, "image_resident_gb", 10)
    pipe.vae.to("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    base = seed if seed is not None else int(time.time()) % 100000
    for i in range(count):
        with torch.no_grad():
            img = pipe(image=refs, **embeds, height=h, width=w, min_generation_area=1024,
                       generator=torch.Generator().manual_seed(base + i), output="images")[0]
        f = out_dir / f"{stem}_{base + i}.png"
        img.save(f)
        files.append(f)
        print(f"saved {f}")
    del pipe, tr
    gc.collect()
    torch.cuda.empty_cache()
    return files
