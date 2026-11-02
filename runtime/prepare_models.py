"""Download only SDXL configs/tokenizers and arrange local adapter aliases."""
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "models"
CONFIG = MODELS / "sdxl-config"
REVISION = "462165984030d82259a11f4367a4eed129e94a7b"


def alias(source, dest):
    if dest.is_file():
        if dest.stat().st_size == source.stat().st_size:
            return
        raise RuntimeError(f"Wrong-sized model alias: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def prepare():
    ready = all((CONFIG / path).is_file() for path in (
        'model_index.json', 'scheduler/scheduler_config.json', 'unet/config.json',
        'vae/config.json', 'text_encoder/config.json', 'text_encoder_2/config.json',
        'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json', 'tokenizer/merges.txt',
        'tokenizer_2/tokenizer_config.json', 'tokenizer_2/vocab.json', 'tokenizer_2/merges.txt',
    ))
    if not ready:
        download_configs()
    encoder = MODELS / "clip_vision"
    encoder.mkdir(parents=True, exist_ok=True)
    if not (encoder / "config.json").is_file():
        source = hf_hub_download("h94/IP-Adapter", "models/image_encoder/config.json")
        shutil.copy2(source, encoder / "config.json")
    weights = encoder / "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    alias(weights, encoder / "model.safetensors")
    identity = MODELS / "ipadapter" / "image_encoder"
    alias(weights, identity / "model.safetensors")
    alias(encoder / "config.json", identity / "config.json")
    print("Local SDXL configuration and reference adapter ready.")


def download_configs():
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", revision=REVISION,
        local_dir=str(CONFIG), allow_patterns=[
            "model_index.json", "scheduler/scheduler_config.json",
            "text_encoder/config.json", "text_encoder_2/config.json",
            "tokenizer/*.json", "tokenizer/merges.txt",
            "tokenizer_2/*.json", "tokenizer_2/merges.txt",
            "unet/config.json", "vae/config.json", "LICENSE.md",
        ],
    )


if __name__ == "__main__":
    if "--if-installed" in sys.argv:
        import downloader
        if not any(m['family'] == 'sdxl' and downloader.model_status(str(MODELS), m)['installed']
                   for m in downloader.load_catalog()['models']):
            print('Qwen is self-contained; optional SDXL configuration is not needed.')
            sys.exit(0)
    prepare()
