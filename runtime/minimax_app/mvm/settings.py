"""App-wide settings (settings.json at the app root). Sets cache env vars so nothing lands on C:."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"
FORGE_ROOT = next((p for p in (ROOT.parent.parent, ROOT.parent / "CharacterForge") if (p / "START.bat").is_file()),
                  ROOT.parent / "CharacterForge")

DEFAULTS = {
    # Where model weights live. Relative paths are relative to the app root.
    "models_dir": "models",
    # Diffusers-format int8 (torchao) build of MiniMax-H3: transformer_ref + text_encoder ~65 GB.
    "video_repo": "abhishekchohan/minimax-h3-int8",
    "video_quantized": True,
    # GGUF image model avoids downloading/converting the 66 GB BF16 transformer.
    "image_repo": "MiniMaxAI/MiniMax-H3-Image",
    "image_backend": "gguf",
    "image_max_vram_gb": 20,
    # Default render canvas. 960x544 is ~2.3x faster per step than the trained 1344x768.
    "width": 960,
    "height": 544,
    "steps": 20,
    "fps": 24,
    # Output compression: max | high | medium | draft (see mvm/media.py). A project can override with "quality".
    "quality": "high",
    # CharacterForge (optional character-sheet backend)
    "forge_dir": str(FORGE_ROOT),
    "forge_url": "http://127.0.0.1:8288",
    "ui_port": 8390,
    # GB of model blocks kept permanently on the GPU (the rest stream from RAM). Attention dominates step time,
    # so residency barely helps, and overfilling VRAM makes Windows spill into shared memory (100x slower).
    "te_resident_gb": 8,
    "dit_resident_gb": 0,
    # Reference images are encoded at this short edge (release default 2048 is too heavy for 24 GB).
    "ref_short_edge": 1024,
}


def load() -> dict:
    s = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    return s


def save(updates: dict) -> dict:
    s = load()
    s.update(updates)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s


def models_dir(s: dict | None = None) -> Path:
    s = s or load()
    p = Path(s["models_dir"])
    return p if p.is_absolute() else ROOT / p


def setup_env() -> None:
    """Keep caches inside this installation and its configured model folder."""
    md = models_dir()
    md.mkdir(parents=True, exist_ok=True)
    rt = ROOT / "runtime" / "cache"
    os.environ.setdefault("HF_HOME", str(md / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(md / "hf" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(rt / "torch"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(rt / "triton"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tmp = rt / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(tmp)
