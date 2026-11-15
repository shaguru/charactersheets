"""H3-Image in stable-diffusion.cpp: GGUF weights, CUDA 12 and CPU offload.

The native runtime is separate from Diffusers so the existing video environment and
its int8 weights stay usable. No conversion or full-precision scratch copy is needed.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
import zipfile
from pathlib import Path

from mvm import settings

TAG = "master-982-0518b34-u27b9ddc"
ASSET = f"sd-{TAG}-bin-win-cuda12-x64.zip"
SHA256 = "389686b13879b40bd2cbec3bfae21017a5e329fb930c0e9c3b4399b21061c327"
RUNTIME = settings.ROOT / "runtime" / "sd-cpp" / TAG

# Pin every model revision; download exactly these files, never a whole quant zoo.
BUNDLE = {
    "transformer": ("unsloth/MiniMax-H3-Image-GGUF", "2eda8f761a1625818e2933e7e726e337d197bc27",
                    "minimax_h3_image_pruned-UD-Q4_K_XL.gguf", 13_607_428_832),
    "text_encoder": ("unsloth/MiniMax-H3-GGUF", "babc040eff0084465460ead58fdb8ec44296f8dd",
                     "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", 18_218_065_024),
    "vae": ("MiniMaxAI/MiniMax-H3-Image", "5f90473023975e6271b0ed1dd84c4e88e4bbc6d3",
            "vae/diffusion_pytorch_model.safetensors", 2_336_964_672),
}


def executable() -> Path | None:
    return next(RUNTIME.rglob("sd-cli.exe"), None) if RUNTIME.exists() else None


def model_paths() -> dict[str, Path]:
    from huggingface_hub import try_to_load_from_cache
    paths = {}
    for key, (repo, rev, file, size) in BUNDLE.items():
        cached = try_to_load_from_cache(repo, file, revision=rev)
        if isinstance(cached, str) and Path(cached).is_file() and Path(cached).stat().st_size == size:
            paths[key] = Path(cached)
    return paths


def ready() -> bool:
    return executable() is not None and len(model_paths()) == len(BUNDLE)


def download_progress() -> dict:
    """Use completed files and Hub's in-place partial files for install feedback."""
    from huggingface_hub import constants
    complete = model_paths()
    total = sum(item[3] for item in BUNDLE.values())
    done = sum(BUNDLE[key][3] for key in complete)
    for key, (repo, _, _, size) in BUNDLE.items():
        if key in complete:
            continue
        blob_dir = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "blobs"
        partials = list(blob_dir.glob("*.incomplete")) if blob_dir.exists() else []
        done += min(size, sum(f.stat().st_size for f in partials if f.exists()))
    return {"downloaded_bytes": done, "total_bytes": total,
            "downloaded_gib": round(done / 2**30, 1), "total_gib": round(total / 2**30, 1),
            "ready": ready()}


def install_runtime() -> Path:
    import requests
    exe = executable()
    if exe:
        return exe
    cache = settings.ROOT / "runtime" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / ASSET
    print("Downloading H3-Image CUDA 12 runtime...", flush=True)
    digest = hashlib.sha256()
    with requests.get(f"https://github.com/unslothai/stable-diffusion.cpp/releases/download/{TAG}/{ASSET}",
                      stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with archive.open("wb") as out:
            for chunk in response.iter_content(8 * 1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError("H3 image runtime checksum failed. Re-run download image.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (RUNTIME / member.filename).resolve().is_relative_to(RUNTIME.resolve()):
                raise RuntimeError("Unsafe path in native runtime archive")
        z.extractall(RUNTIME)
    archive.unlink()
    exe = executable()
    if not exe:
        raise RuntimeError("Native runtime archive has no sd-cli.exe")
    return exe


def environment() -> dict:
    env = os.environ.copy()
    # CUDA DLLs already ship with our local PyTorch. Do not require system CUDA.
    torch_lib = settings.ROOT / "runtime" / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    exe = executable()
    env["PATH"] = os.pathsep.join([str(exe.parent) if exe else str(RUNTIME), str(torch_lib), env.get("PATH", "")])
    return env


def generate(prompt: str, images: list[Path], out_dir: Path, size=(1344, 768), count=1,
             stem="image", seed: int | None = None) -> list[Path]:
    if not ready():
        raise SystemExit("H3-Image is not installed. Run mvm.bat download image.")
    if not prompt.strip():
        raise ValueError("Enter an image description.")
    w, h = size
    if w % 32 or h % 32 or not 256 <= min(w, h) <= max(w, h) <= 4096:
        raise ValueError("Image dimensions must be multiples of 32, between 256 and 4096.")
    paths = model_paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = seed if seed is not None else int(time.time_ns() % 2**31)
    files = []
    for i in range(count):
        out = out_dir / f"{stem}_{base + i}.png"
        args = [str(executable()), "--diffusion-model", str(paths["transformer"]),
                "--vae", str(paths["vae"]), "--llm", str(paths["text_encoder"]),
                "--prompt", prompt, "--width", str(w), "--height", str(h),
                "--cfg-scale", "1.0", "--steps", "6", "--sampling-method", "euler",
                "--sigmas", "minimax-h3-image", "--rng", "cpu", "--seed", str(base + i),
                "--max-vram", str(settings.load().get("image_max_vram_gb", 20)),
                "--diffusion-fa", "--output", str(out)]
        for img in images:
            args.extend(["--ref-image", str(img)])
        print(f"[{time.strftime('%H:%M:%S')}] H3-Image {w}x{h}, 6 steps, seed {base+i}; loading GGUF weights", flush=True)
        subprocess.run(args, env=environment(), cwd=settings.ROOT, check=True)
        if not out.is_file():
            raise RuntimeError("Image renderer finished without an output image. See the job log.")
        files.append(out)
        print(f"saved {out}", flush=True)
    return files
