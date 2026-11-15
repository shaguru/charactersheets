"""Model weights: where they live and how to fetch them.

Weights go into the Hugging Face cache under <models_dir>/hf/hub (HF_HUB_CACHE is set in settings.setup_env).
The renderer resolves the installed cache snapshot locally. No second local_dir copy is needed.
"""
from __future__ import annotations

from pathlib import Path

from mvm import settings

# Only what the ref2va workflow needs: skip the t2va/fl2va transformer (31.7 GB).
VIDEO_IGNORE = ["transformer/*", "*.mp4", "*.png"]
VIDEO_REVISIONS = {"abhishekchohan/minimax-h3-int8": "b06d1d16e2c113035df3d21b05a619a9a9d194bb"}
# The image repo carries its own bf16 copy of the 62 GB text encoder; we reuse the video build's int8 one.
IMAGE_IGNORE = ["text_encoder/*", "examples/*", "assets/*", "*.mp4", "transformer/*", "vae/*"]


def _marker(repo: str) -> Path:
    return settings.models_dir() / "hf" / f".complete_{repo.replace('/', '__')}"


def status(s: dict | None = None) -> dict:
    s = s or settings.load()
    from mvm import image_native
    image = image_native.ready() if s.get("image_backend", "gguf") == "gguf" else _marker(s["image_repo"]).exists()
    return {"video": _marker(s["video_repo"]).exists(), "image": image}


def snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=VIDEO_REVISIONS.get(repo), local_files_only=True,
                                  ignore_patterns=VIDEO_IGNORE if "h3-int8" in repo.lower() else IMAGE_IGNORE))


def require(which: str) -> str:
    """Return the repo id for `which`, after checking it is fully downloaded."""
    s = settings.load()
    repo = s["video_repo"] if which == "video" else s["image_repo"]
    if not _marker(repo).exists():
        raise SystemExit(f"The {which} model is not downloaded. Run: mvm download {which}   (into {settings.models_dir(s)})")
    return repo


def download(which: str) -> None:
    from huggingface_hub import snapshot_download
    s = settings.load()
    jobs = []
    if which in ("video", "all"):
        jobs.append((s["video_repo"], VIDEO_IGNORE))
    if which in ("image", "all"):
        if s.get("image_backend", "gguf") == "gguf":
            download_image_gguf()
        else:
            jobs.append((s["image_repo"], IMAGE_IGNORE))
    for repo, ignore in jobs:
        print(f"Downloading {repo} into {settings.models_dir(s)}  (resumable: re-run the same command if it stops)", flush=True)
        path = snapshot_download(repo, revision=VIDEO_REVISIONS.get(repo), ignore_patterns=ignore, max_workers=8)
        _marker(repo).write_text(path, encoding="utf-8")
        print(f"Done: {path}", flush=True)


def download_image_gguf():
    import shutil
    from huggingface_hub import hf_hub_download
    from mvm import image_native
    md = settings.models_dir()
    md.mkdir(parents=True, exist_ok=True)
    progress = image_native.download_progress()
    missing = max(0, progress["total_bytes"] - progress["downloaded_bytes"])
    # Include native runtime and headroom. Incomplete Hub downloads resume in place.
    needed = missing + 4 * 2**30
    free = shutil.disk_usage(md).free
    if free < needed:
        raise SystemExit(f"H3-Image needs {needed/2**30:.1f} GiB free including headroom; "
                         f"{md} has {free/2**30:.1f} GiB. Free space or set models_dir to another drive.")
    print(f"H3-Image GGUF: {missing/2**30:.1f} GiB remaining, {free/2**30:.1f} GiB free. "
          "Existing video weights stay in place.", flush=True)
    image_native.install_runtime()
    for key, (repo, rev, file, size) in image_native.BUNDLE.items():
        print(f"Downloading {key}: {file} ({size/2**30:.2f} GiB, resumable)", flush=True)
        hf_hub_download(repo, file, revision=rev)
        for doc in ("LICENSE", "NOTICE", "README.md"):
            try:
                hf_hub_download(repo, doc, revision=rev)
            except Exception:
                pass  # some official repositories have no NOTICE
    if not image_native.ready():
        raise SystemExit("H3-Image download did not complete; re-run mvm.bat download image.")
    print("H3-Image installed (UD-Q4_K_XL + Q4_K_M text encoder + official VAE).", flush=True)
