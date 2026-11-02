"""Standalone Character Forge image server.

This is deliberately independent of ComfyUI.  It loads the local SDXL
checkpoints with Diffusers, keeps one model on the CUDA device, and exposes
the same small /forge API used by the existing studio page.
"""
from __future__ import annotations

import gc
import glob
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, PngImagePlugin
import uvicorn
import downloader
from memory_policy import limit_dimensions, presets_for_profile, profile_for_vram

# Generation uses verified local files. Setup is the only code that fetches
# configs or weights, so a render cannot quietly wait for an online download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WEB = ROOT / "runtime" / "web"
PRESETS = ROOT / "runtime" / "presets.json"
CATALOG_PATH = ROOT / "runtime" / "models.json"
MODELS = DATA / "models"
INPUT = DATA / "input"
OUTPUT = DATA / "output"
SDXL_CONFIG = MODELS / "sdxl-config"
IP_DIR = MODELS / "ipadapter"
PORT = int(os.environ.get("FORGE_PORT", "8288"))
SETTINGS = DATA / "user" / "forge_settings.json"
for directory in (INPUT, OUTPUT, SETTINGS.parent, MODELS):
    directory.mkdir(parents=True, exist_ok=True)

with CATALOG_PATH.open("r", encoding="utf-8-sig") as f:
    CATALOG = json.load(f)
with PRESETS.open("r", encoding="utf-8") as f:
    PRESET_DATA = json.load(f)
MEMORY_PROFILE = profile_for_vram(
    torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
    os.environ.get("FORGE_LOW_VRAM", "auto"),
)

app = FastAPI(title="Character Forge direct engine")
app.mount("/forge/static", StaticFiles(directory=str(WEB)), name="forge-static")

STATE_LOCK = threading.RLock()
JOBS: list[dict[str, Any]] = []
QUEUE: list[dict[str, Any]] = []
CURRENT: dict[str, Any] | None = None
STOP = threading.Event()
WORKER_WAKE = threading.Event()
PIPE = None
PIPE_MODEL = None
PIPE_FAMILY = None
PIPE_LOCK = threading.RLock()
DL = {"queue": [], "current": None, "status": {}, "cancel": threading.Event()}


def download_worker():
    while True:
        with STATE_LOCK:
            if not DL["queue"]:
                DL["current"] = None
                return
            mid = DL["queue"].pop(0)
            DL["current"] = mid
            DL["cancel"].clear()
            DL["status"][mid] = {"state": "downloading", "done": 0, "total": 0}
        def progress(value):
            with STATE_LOCK:
                DL["status"][mid].update(value)
        try:
            downloader.download_model(str(MODELS), model(mid), read_settings().get("hf_token"), progress, DL["cancel"])
            if model(mid)["family"] == "sdxl":
                env = dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
                subprocess.run([sys.executable, str(ROOT / "runtime" / "prepare_models.py")],
                               env=env, check=True)
            DL["status"][mid] = {"state": "done"}
        except InterruptedError:
            DL["status"][mid] = {"state": "cancelled"}
        except Exception as exc:
            DL["status"][mid] = {"state": "error", "error": str(exc)}


@app.post("/forge/api/download")
def download(body: dict[str, Any]):
    mid = body.get("id")
    if not model(mid):
        raise HTTPException(400, "Unknown model")
    with STATE_LOCK:
        if mid == DL["current"] or mid in DL["queue"]:
            return {"ok": True}
        DL["queue"].append(mid)
        DL["status"][mid] = {"state": "queued"}
        if DL["current"] is None:
            DL["current"] = "starting"
            threading.Thread(target=download_worker, daemon=True).start()
    return {"ok": True}


@app.post("/forge/api/download/cancel")
def cancel_download():
    with STATE_LOCK:
        for mid in DL["queue"]:
            DL["status"][mid] = {"state": "cancelled"}
        DL["queue"].clear()
        DL["cancel"].set()
    return {"ok": True}


@app.exception_handler(HTTPException)
async def http_error(_request, exc):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)


def read_settings():
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def save_settings(body):
    with STATE_LOCK:
        value = read_settings()
        value.update(body)
        temporary = SETTINGS.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(SETTINGS)


class Cancelled(Exception):
    pass


def check_cancel():
    if STOP.is_set():
        raise Cancelled("Stopped by user")


def update(job, stage, **fields):
    with STATE_LOCK:
        job.update(stage=stage, updated=time.time(), **fields)


def safe_name(value: str | None, fallback="Project") -> str:
    value = re.sub(r"[^A-Za-z0-9 _\-]+", "", (value or "")).strip().replace(" ", "_")
    return value[:60] or fallback


def project_dir(project: str) -> Path:
    d = OUTPUT / safe_name(project)
    (d / "_forge").mkdir(parents=True, exist_ok=True)
    return d


def jobs_path(project: str) -> Path:
    return project_dir(project) / "_forge" / "jobs.jsonl"


def favs_path(project: str) -> Path:
    return project_dir(project) / "_forge" / "favorites.json"


def read_favs(project: str) -> set[str]:
    try:
        return set(json.loads(favs_path(project).read_text(encoding="utf-8")))
    except Exception:
        return set()


def write_favs(project: str, favs: set[str]) -> None:
    favs_path(project).write_text(json.dumps(sorted(favs)), encoding="utf-8")


def load_jobs(project: str) -> list[dict[str, Any]]:
    # The append-only journal contains initial records and terminal updates.
    # Last record wins, retaining the original insertion order in the gallery.
    with STATE_LOCK:
        p = jobs_path(project)
        if not p.exists():
            return []
        out = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                if value.get("deleted"):
                    out.pop(value.get("id"), None)
                else:
                    out[value["id"]] = value
            except (ValueError, KeyError, TypeError):
                continue
        return list(out.values())


def append_job(job: dict[str, Any]) -> None:
    with STATE_LOCK, jobs_path(job["project"]).open("a", encoding="utf-8") as f:
        f.write(json.dumps(job) + "\n")


def job_output_files(job: dict[str, Any]) -> list[Path]:
    """Return every generated file owned by a job, including legacy records."""
    if job.get("output_file"):
        return [Path(job["output_file"])]
    prefix = job.get("prefix", "")
    if not prefix:
        return []
    return [Path(p) for p in glob.glob(glob.escape(str(OUTPUT / prefix)) + "_*.png")]


def delete_job_files(job: dict[str, Any]) -> None:
    for path in job_output_files(job):
        try:
            path.resolve().relative_to(OUTPUT.resolve())
        except ValueError:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def model(mid: str) -> dict[str, Any] | None:
    return next((m for m in CATALOG["models"] if m["id"] == mid), None)


def file_status(f: dict[str, Any]) -> tuple[str, int]:
    p = MODELS / f["dir"] / f["name"]
    if p.is_file() and p.stat().st_size == f["size"]:
        return "ok", f["size"]
    return ("wrong-size", p.stat().st_size) if p.is_file() else ("missing", 0)


def model_status(m: dict[str, Any]) -> dict[str, Any]:
    total = done = needed = 0
    missing = 0
    for f in m.get("files", []):
        total += f["size"]
        st, n = file_status(f)
        done += f["size"] if st == "ok" else n
        if st != "ok":
            missing += 1
            needed += max(0, f["size"] - n)
    return {"installed": missing == 0, "bytes_done": done, "bytes_total": total, "bytes_needed": needed}


def confined_path(base: Path, name: str) -> Path:
    p = (base / name).resolve()
    if not p.is_relative_to(base.resolve()):
        raise HTTPException(400, "Invalid file path")
    return p


def ref_path(name: str) -> Path:
    output = name.endswith(" [output]")
    return confined_path(OUTPUT if output else INPUT, name.removesuffix(" [output]"))


def build_prompt(body: dict[str, Any]) -> str:
    """Turn the studio mode into ordinary SDXL prompt text.

    IP-Adapter carries the identity embedding, so model-specific Comfy image
    tokens are intentionally removed from the direct path.
    """
    mode = body.get("mode", "sheet")
    notes = (body.get("notes") or "").strip()
    if mode == "sheet":
        sheet = next((s for s in PRESET_DATA["sheets"] if s["id"] == body.get("sheet")), PRESET_DATA["sheets"][0])
        bg = PRESET_DATA["backgrounds"].get(body.get("bg", "white"), PRESET_DATA["backgrounds"]["white"])
        return sheet["prompt"].replace("{CHAR}", "the reference character").replace("{BG}", bg).replace("{NOTES}", notes)
    if mode == "scene":
        scene = (body.get("scene") or "").strip()
        if not scene:
            raise ValueError("Describe the scene first.")
        lock = PRESET_DATA["identity_lock"].replace("{CHAR}", "the reference character")
        return PRESET_DATA["scene_template"].replace("{SCENE}", scene).replace("{LOCK}", lock).replace("{CHAR}", "the reference character") + (" " + notes if notes else "")
    raw = (body.get("raw") or "").strip()
    if not raw:
        raise ValueError("Write a prompt first.")
    return raw.replace("{CHAR}", "the reference character")


_DYNAMIC_PROMPT = re.compile(r"\[([^\[\]]*\|[^\[\]]*)\]")


def expand_dynamic_prompt(text: str, seed: int) -> str:
    """Resolve [choice A | choice B] groups once for one generated image."""
    rng = random.Random(seed)

    def choose(match: re.Match[str]) -> str:
        choices = [part.strip() for part in match.group(1).split("|")]
        return rng.choice(choices) if all(choices) else match.group(0)

    return _DYNAMIC_PROMPT.sub(choose, text or "")


def model_prompt(prompt, m, refs):
    """Resolve extra image placeholders for native multi-image conditioning."""
    def replace(match):
        index = int(match.group(1))
        if m.get("family") != "qwen-edit" or not 2 <= index <= len(refs):
            raise ValueError(f"{match.group(0)} requires a corresponding Qwen reference image.")
        return f"image {index}"
    return re.sub(r"\{REF(\d+)\}", replace, prompt)


def image_for_job(job: dict[str, Any]) -> dict[str, str]:
    p = Path(job["output_file"])
    return {"filename": p.name, "subfolder": p.parent.relative_to(OUTPUT).as_posix(), "type": "output"}


@torch.inference_mode()
def prompt_embeddings(pipe, positive, negative):
    """Encode every 75-token CLIP window so sheet instructions aren't cut off."""
    encoders = [(pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)]
    tokens = [[tok(text, add_special_tokens=False, verbose=False).input_ids for tok, _ in encoders]
              for text in (positive, negative)]
    windows = max(1, max((len(ids) + 74) // 75 for pair in tokens for ids in pair))
    result = {}
    for key, pair in zip(("prompt_embeds", "negative_prompt_embeds"), tokens):
        hidden = []
        pooled = None
        for (tok, encoder), ids in zip(encoders, pair):
            pieces = []
            for n in range(windows):
                check_cancel()
                part = ids[n * 75:(n + 1) * 75]
                padded = [tok.bos_token_id, *part, tok.eos_token_id] + [tok.pad_token_id] * (75 - len(part))
                value = encoder(torch.tensor([padded], device=pipe._execution_device), output_hidden_states=True)
                pieces.append(value.hidden_states[-2])
                if n == 0:
                    pooled = value[0]
            hidden.append(torch.cat(pieces, dim=1))
        result[key] = torch.cat(hidden, dim=-1).to(dtype=pipe.unet.dtype)
        result[key.replace("prompt_embeds", "pooled_prompt_embeds")] = pooled.to(dtype=pipe.unet.dtype)
    return result


def load_pipeline(mid: str, job=None):
    global PIPE, PIPE_MODEL, PIPE_FAMILY
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    m = model(mid)
    if not m or m.get("family") not in {"sdxl", "qwen-edit"}:
        raise RuntimeError("The direct engine supports SDXL and Qwen Rapid AIO models.")
    ckpt = MODELS / next((f["dir"] + "/" + f["name"] for f in m["files"] if f["dir"] == "checkpoints"), "qwen-rapid-aio/Qwen-Rapid-AIO-NSFW-v19.safetensors")
    if not ckpt.is_file():
        raise RuntimeError(f"Checkpoint is missing: {ckpt.name}")
    if m.get("family") == "sdxl" and not SDXL_CONFIG.joinpath("model_index.json").is_file():
        raise RuntimeError("Local SDXL configuration is missing. Run START.bat once to finish setup.")
    if PIPE_MODEL == mid and PIPE is not None:
        return PIPE
    if PIPE is not None:
        del PIPE
        PIPE = None
        PIPE_MODEL = None
        PIPE_FAMILY = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if m.get("family") == "qwen-edit":
        from rapid_aio import load_rapid_aio
        checkpoint = MODELS / "qwen-rapid-aio" / "Qwen-Rapid-AIO-NSFW-v19.safetensors"
        config_dir = MODELS / "qwen-rapid-aio" / "config"
        if job:
            update(job, "assembling Qwen Rapid AIO")
        def loading(component, done, total):
            check_cancel()
            if job:
                update(job, f"loading Qwen {component}: {done}/{total} tensors")
        pipe = load_rapid_aio(checkpoint, config_dir, reference=True, progress=loading)
        check_cancel()
        update(job, "moving Qwen model with sequential CPU offload") if job else None
        # Qwen Rapid's transformer is too large for a 24 GB card as a single
        # offloaded module. Sequential hooks keep each block's weights on the
        # GPU only for its forward pass, trading time for a reliable fit.
        pipe.enable_sequential_cpu_offload(device="cuda")
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        PIPE, PIPE_MODEL, PIPE_FAMILY = pipe, mid, "qwen-edit"
        return pipe
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(ckpt), config=str(SDXL_CONFIG), dtype=dtype,
        local_files_only=True, low_cpu_mem_usage=True,
    )
    check_cancel()
    if job:
        update(job, "moving model to GPU")
    if not MEMORY_PROFILE["low_vram"]:
        pipe = pipe.to(device)
    check_cancel()
    if job:
        update(job, "loading reference adapter")
    # Diffusers expects the encoder under image_encoder/model.safetensors.
    # START.bat creates this hardlink without duplicating the 2.5 GB file.
    pipe.load_ip_adapter(
        str(IP_DIR), subfolder="", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        image_encoder_folder="image_encoder", local_files_only=True,
    )
    if MEMORY_PROFILE["low_vram"]:
        # Attach hooks after the adapter/encoder exist, so they are offloaded too.
        pipe.enable_sequential_cpu_offload(device="cuda")
    pipe.set_ip_adapter_scale(float(m.get("reference_weight", 0.8)))
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    pipe.vae.enable_tiling()
    PIPE, PIPE_MODEL, PIPE_FAMILY = pipe, mid, "sdxl"
    return pipe


def render(job: dict[str, Any]) -> None:
    global CURRENT
    with STATE_LOCK:
        CURRENT = job
        job.update(status="running", stage="loading model", progress=0, started=time.time(), updated=time.time(), step=0, steps=int(job["steps"]))
    try:
        with PIPE_LOCK:
            check_cancel()
            pipe = load_pipeline(job["model"], job)
            update(job, "encoding reference", progress=0.05)
            with Image.open(ref_path(job["refs"][0])) as source:
                reference = ImageOps.exif_transpose(source).convert("RGB")
            model_family = model(job["model"])["family"]
            if model_family == "qwen-edit":
                from rapid_aio_scheduler import QwenEulerAncestralScheduler
                qwen_refs = []
                for r in job["refs"][:3]:
                    with Image.open(ref_path(r)) as source:
                        qwen_refs.append(ImageOps.exif_transpose(source).convert("RGB"))
                steps = int(job["steps"])
                device = "cuda"
                gen = torch.Generator(device=device).manual_seed(int(job["seed"]))
                pipe.scheduler = QwenEulerAncestralScheduler(noise_sampler_seed=int(job["seed"]))
                update(job, "encoding Qwen references and prompt", progress=0.08, step=0, steps=steps)

                def qwen_cb(_pipe, step, _timestep, _kwargs):
                    update(job, "decoding image" if step + 1 == steps else "sampling", step=step + 1, steps=steps,
                           progress=0.10 + 0.82 * (step + 1) / steps)
                    check_cancel()
                    return _kwargs

                handles = []
                blocks = pipe.transformer.transformer_blocks
                for index, block in enumerate(blocks):
                    def block_progress(_module, _inputs, index=index):
                        check_cancel()
                        completed = job.get("step", 0)
                        update(job, f"sampling step {completed + 1}/{steps}, block {index + 1}/{len(blocks)}",
                               progress=0.10 + 0.82 * (completed + index / len(blocks)) / steps)
                    handles.append(block.register_forward_pre_hook(block_progress))
                try:
                    result = pipe(
                        image=qwen_refs, prompt=job["prompt"],
                        negative_prompt=(job.get("negative") or None) if job["cfg"] > 1 else None,
                        true_cfg_scale=float(job["cfg"]), width=int(job["width"]), height=int(job["height"]),
                        num_inference_steps=steps, max_sequence_length=512, generator=gen,
                        callback_on_step_end=qwen_cb,
                    ).images[0]
                finally:
                    for handle in handles:
                        handle.remove()
                update(job, "saving", progress=0.95)
                out = Path(job["output_file"])
                out.parent.mkdir(parents=True, exist_ok=True)
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("prompt", job["prompt"])
                metadata.add_text("model", job["model_name"])
                metadata.add_text("seed", str(job["seed"]))
                result.save(out, pnginfo=metadata)
                update(job, "done", status="done", progress=1.0, step=steps, finished=time.time())
                return

            # The face adapter was trained on head crops. Use the upper square
            # of the reference so a full-body input does not dominate identity
            # with its background and clothes.
            side = min(reference.width, reference.height)
            reference = reference.crop(((reference.width - side) // 2, 0, (reference.width + side) // 2, side))
            steps = int(job["steps"])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gen = torch.Generator(device=device).manual_seed(int(job["seed"]))
            pipe.set_ip_adapter_scale(job.get("reference_strength", 0.65))
            check_cancel()
            with torch.inference_mode():
                embeds = pipe.prepare_ip_adapter_image_embeds(reference, None, device, 1, job["cfg"] > 1)
                update(job, "encoding prompt", progress=0.08)
                texts = prompt_embeddings(pipe, job["prompt"], job.get("negative") or "")
            check_cancel()
            update(job, "sampling", progress=0.10, sampling_started=time.time())

            def cb(_pipe, step, _timestep, _kwargs):
                update(job, "decoding image" if step + 1 == steps else "sampling", step=step + 1, steps=steps,
                       progress=0.10 + 0.82 * (step + 1) / steps)
                check_cancel()
                return _kwargs

            result = pipe(
                **texts,
                width=int(job["width"]), height=int(job["height"]),
                num_inference_steps=steps, guidance_scale=float(job["cfg"]),
                generator=gen, ip_adapter_image_embeds=embeds,
                callback_on_step_end=cb,
            ).images[0]
            update(job, "saving", progress=0.95)
            out = Path(job["output_file"])
            out.parent.mkdir(parents=True, exist_ok=True)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("prompt", job["prompt"])
            metadata.add_text("model", job["model_name"])
            metadata.add_text("seed", str(job["seed"]))
            result.save(out, pnginfo=metadata)
            update(job, "done", status="done", progress=1.0, finished=time.time())
    except Cancelled as e:
        update(job, "cancelled", status="cancelled", error=str(e), finished=time.time())
    except Exception as e:
        update(job, "error", status="error", error=f"{type(e).__name__}: {e}", finished=time.time())
        traceback.print_exc()
    finally:
        try:
            append_job(job)
        except Exception:
            traceback.print_exc()
        with STATE_LOCK:
            CURRENT = None
            STOP.clear()


def worker() -> None:
    while True:
        WORKER_WAKE.wait(1)
        WORKER_WAKE.clear()
        while True:
            with STATE_LOCK:
                job = QUEUE.pop(0) if QUEUE else None
                if job is not None:
                    global CURRENT
                    CURRENT = job
                    STOP.clear()
            if job is None:
                break
            render(job)


threading.Thread(target=worker, daemon=True, name="forge-renderer").start()


@app.get("/forge")
def forge_root():
    return RedirectResponse("/forge/", status_code=307)


@app.get("/forge/")
def forge_index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/forge/api/config")
def config():
    projects = sorted(p.name for p in OUTPUT.iterdir() if p.is_dir() and (p / "_forge").is_dir()) if OUTPUT.exists() else []
    ms = []
    for m in CATALOG["models"]:
        mm = {k: v for k, v in m.items() if k != "files"}
        mm.update(model_status(m))
        mm["download"] = DL["status"].get(m["id"])
        ms.append(mm)
    return {"models": ms, "presets": presets_for_profile(PRESET_DATA, MEMORY_PROFILE),
            "memory_profile": MEMORY_PROFILE, "has_token": bool(read_settings().get("hf_token")),
            "free_gb": round(shutil.disk_usage(MODELS).free / 1e9, 1),
            "projects": projects, "last_project": read_settings().get("last_project") or "My_Project",
            "output_dir": str(OUTPUT), "models_dir": str(MODELS), "backend": "direct-diffusers",
            "cuda": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "runtime": runtime_status()}


def runtime_status() -> dict[str, Any]:
    with STATE_LOCK:
        current = dict(CURRENT) if CURRENT else None
        queued = len(QUEUE)
        loaded = PIPE is not None
        model_id = (current.get("model") if current and str(current.get("stage", "")).startswith(("loading", "assembling", "moving")) else PIPE_MODEL)
        family = PIPE_FAMILY
    loading = bool(current and str(current.get("stage", "")).startswith(("loading", "assembling", "moving")))
    state = "loading" if loading else ("loaded" if loaded else "unloaded")
    return {"state": state, "model_id": model_id, "model_name": model(model_id)["name"] if model_id and model(model_id) else None,
            "family": family, "queue": queued, "busy": bool(current)}


@app.post("/upload/image")
async def upload_image(image: UploadFile = File(...), subfolder: str = Form("forge_refs"), type: str = Form("input")):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image.filename or "image.png").name)
    if type != "input":
        raise HTTPException(400, "Uploads must be input references")
    target = INPUT / safe_name(subfolder, "forge_refs")
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{uuid.uuid4().hex[:8]}_{safe}"
    content = await image.read(30 * 1024 * 1024 + 1)
    if len(content) > 30 * 1024 * 1024:
        raise HTTPException(400, "Reference image must be under 30 MB")
    try:
        with Image.open(io.BytesIO(content)) as decoded:
            decoded.verify()
    except Exception:
        raise HTTPException(400, "Choose a valid image file")
    path.write_bytes(content)
    return {"name": path.name, "subfolder": target.relative_to(INPUT).as_posix() if type == "input" else target.relative_to(OUTPUT).as_posix()}


@app.get("/view")
def view(filename: str, subfolder: str = "", type: str = "output"):
    base = OUTPUT if type == "output" else INPUT
    p = confined_path(base, str(Path(subfolder) / Path(filename).name))
    if not p.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(p)


@app.post("/forge/api/project")
def new_project(body: dict[str, Any]):
    name = safe_name(body.get("name"))
    project_dir(name)
    save_settings({"last_project": name})
    return {"ok": True, "name": name}


@app.post("/forge/api/queue")
def queue(body: dict[str, Any]):
    if not torch.cuda.is_available():
        raise HTTPException(503, "CUDA is unavailable. Check the NVIDIA driver and local runtime; generation requires the GPU.")
    refs = [r for r in (body.get("refs") or []) if r]
    if not refs:
        raise HTTPException(400, "Drop at least one character image first.")
    mids = body.get("models") or []
    if not mids:
        raise HTTPException(400, "Pick a downloaded model first.")
    if len(refs) > 3:
        raise HTTPException(400, "Use at most three reference images.")
    selected_families = {model(mid).get("family") for mid in mids if model(mid)}
    if "qwen-edit" not in selected_families and len(refs) != 1:
        raise HTTPException(400, "The SDXL models use one character reference. Qwen Rapid AIO supports up to three.")
    reference_path = ref_path(refs[0])
    if any(not ref_path(ref).is_file() for ref in refs):
        raise HTTPException(400, "Reference image is missing. Drop the image again.")
    mode = body.get("mode", "sheet")
    if mode not in ("sheet", "scene", "free"):
        raise HTTPException(400, "Unknown generation mode")
    try:
        prompt = build_prompt(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    project = safe_name(body.get("project"))
    size = PRESET_DATA["sizes"].get(body.get("size") or "match", PRESET_DATA["sizes"]["match"])
    if mode == "sheet" and not body.get("size"):
        sh = next((x for x in PRESET_DATA["sheets"] if x["id"] == body.get("sheet")), PRESET_DATA["sheets"][0])
        size = PRESET_DATA["sizes"][sh["size"]]
    try:
        with Image.open(reference_path) as ref:
            rw, rh = ImageOps.exif_transpose(ref).size
        ratio = (1024 * 1024 / (rw * rh)) ** 0.5
        w = int(body.get("width") or size["w"] or round(rw * ratio / 8) * 8)
        h = int(body.get("height") or size["h"] or round(rh * ratio / 8) * 8)
        count = int(body.get("count") or 1)
        seed0 = int(body["seed"]) if body.get("seed") is not None else random.randint(1, 2**31)
        strength = float(body.get("reference_strength", 0.65))
    except (ValueError, TypeError, OSError):
        raise HTTPException(400, "Invalid image or generation settings")
    if w < 256 or h < 256 or w > 2048 or h > 2048 or w % 8 or h % 8 or w * h > 2_097_152:
        raise HTTPException(400, "Use dimensions from 256 to 2048, multiples of 8, and at most 2 megapixels.")
    requested_size = (w, h)
    w, h = limit_dimensions(w, h, MEMORY_PROFILE["max_pixels"])
    notices = []
    if (w, h) != requested_size:
        notices.append(f"Low VRAM: render size reduced from {requested_size[0]}x{requested_size[1]} to {w}x{h} for CPU offload.")
    if not 1 <= count <= 200 or not 0 <= strength <= 1.5 or not 0 <= seed0 < 2**63 - 200:
        raise HTTPException(400, "Invalid count, reference strength or seed")
    batch = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    queued = 0
    errors = []
    for mid in dict.fromkeys(mids):
        m = model(mid)
        if not m or not model_status(m)["installed"]:
            errors.append(f"{mid}: model is not downloaded")
            continue
        try:
            resolved_prompt = model_prompt(prompt, m, refs)
        except ValueError as exc:
            errors.append(f"{m['name']}: {exc}")
            continue
        try:
            steps = int(body.get("steps") or m["defaults"]["steps"])
            cfg = float(body.get("cfg") or m["defaults"]["cfg"])
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))
        if not 1 <= steps <= 100 or not 1 <= cfg <= 20:
            raise HTTPException(400, "Use 1–100 steps and CFG from 1 to 20")
        for k in range(count):
            seed = seed0 + k if body.get("seed_mode") == "fixed" else random.randint(1, 2**48)
            jid = f"{batch}_{mid}_{k:03d}"
            out = OUTPUT / project / mode / f"{jid}_{seed}.png"
            dynamic_prompt = expand_dynamic_prompt(resolved_prompt, seed)
            negative = str(body.get("negative") or "low quality, blurry, deformed, watermark").strip()
            dynamic_negative = expand_dynamic_prompt(negative, seed + 1)
            job = {"id": jid, "project": project, "batch": batch, "mode": mode, "sheet": body.get("sheet"),
                   "model": mid, "model_name": m["name"], "refs": refs[:int(m.get("max_refs", 1))], "prompt": resolved_prompt,
                   "prompt_template": resolved_prompt, "negative": dynamic_negative, "negative_template": negative, "width": w, "height": h,
                   "seed": seed, "steps": steps, "cfg": cfg, "sampler": m.get("sampler", "default"),
                   "guidance_name": m.get("guidance_name", "CFG scale"), "output_file": str(out),
                   "reference_strength": strength, "fast": False,
                   "created": time.time(), "req": {k: v for k, v in body.items() if k not in ("models", "count", "seed", "seed_mode")},
                   "status": "queued", "stage": "queued", "progress": 0.0}
            job["prompt"] = dynamic_prompt
            with STATE_LOCK:
                JOBS.append(job)
                append_job(job)
                QUEUE.append(job)
            queued += 1
    WORKER_WAKE.set()
    save_settings({"last_project": project})
    return {"queued": queued, "errors": errors, "batch": batch, "notices": notices}


@app.get("/forge/api/jobs")
def jobs(project: str = "My_Project"):
    project = safe_name(project)
    stored = load_jobs(project)
    favs = read_favs(project)
    with STATE_LOCK:
        live = {j["id"]: dict(j) for j in JOBS}
        current = dict(CURRENT) if CURRENT else None
        running = 1 if current else 0
        pending = len(QUEUE)
        positions = {j["id"]: n + 1 for n, j in enumerate(QUEUE)}
    out = []
    for j in reversed(stored):
        x = dict(j)
        if "output_file" not in j:
            # Jobs created by the retired ComfyUI backend do not have a
            # direct output path. Keep them visible in the gallery without
            # letting one legacy record break the status endpoint.
            prefix = j.get("prefix", "")
            files = sorted(glob.glob(glob.escape(str(OUTPUT / prefix)) + "_*.png")) if prefix else []
            x["status"] = "done" if files else "interrupted"
            x["images"] = [{"filename": Path(f).name, "subfolder": Path(f).parent.relative_to(OUTPUT).as_posix(), "type": "output"} for f in files]
            x["fav"] = j.get("id") in favs
            out.append(x)
            continue
        l = live.get(j["id"]) or (current if current and current["id"] == j["id"] else None)
        if l:
            x.update({k: l.get(k) for k in ("status", "stage", "progress", "step", "steps", "error", "started") if k in l})
        else:
            x["status"] = "done" if Path(j["output_file"]).is_file() else j.get("status", "interrupted")
            if x["status"] in ("running", "queued"):
                x.update(status="interrupted", stage="interrupted")
        x["images"] = [image_for_job(j)] if Path(j["output_file"]).is_file() else []
        if x["status"] == "done" and not x["images"]:
            x["status"] = "interrupted"
        x["queue_position"] = positions.get(j["id"])
        if l:
            x.update(l)
        x["elapsed"] = round(max(0, (x.get("finished") or time.time()) - x.get("started", time.time())), 1)
        x["fav"] = j.get("id") in favs
        out.append(x)
    return {"jobs": out, "queue_running": running, "queue_pending": pending, "runtime": runtime_status(),
            "current": {**{k: current.get(k) for k in ("id", "stage", "progress", "step", "steps", "model_name", "updated")},
                        "elapsed": round(time.time() - current.get("started", time.time()), 1)} if current else None}


@app.post("/forge/api/stop")
def stop():
    with STATE_LOCK:
        for job in QUEUE:
            update(job, "cancelled", status="cancelled", error="Stopped before rendering", finished=time.time())
            append_job(job)
        QUEUE.clear()
        if CURRENT:
            STOP.set()
    return {"ok": True}


@app.post("/forge/api/fav")
def fav(body: dict[str, Any]):
    project = safe_name(body.get("project"))
    favs = read_favs(project)
    jid = body.get("id")
    if body.get("fav"):
        favs.add(jid)
    else:
        favs.discard(jid)
    write_favs(project, favs)
    return {"ok": True}


@app.post("/forge/api/delete")
def delete_images(body: dict[str, Any]):
    """Delete generated images from one project without touching references/models."""
    project = safe_name(body.get("project"))
    requested = {str(value) for value in (body.get("ids") or []) if value}
    delete_all = bool(body.get("all"))
    if not delete_all and not requested:
        raise HTTPException(400, "Select at least one generated image.")
    stored = load_jobs(project)
    with STATE_LOCK:
        active_ids = {job["id"] for job in QUEUE}
        if CURRENT:
            active_ids.add(CURRENT["id"])
    targets = [job for job in stored if delete_all or job.get("id") in requested]
    deleted = skipped = 0
    for job in targets:
        if job.get("id") in active_ids:
            skipped += 1
            continue
        delete_job_files(job)
        tombstone = {"id": job["id"], "project": project, "deleted": True, "deleted_at": time.time()}
        append_job(tombstone)
        deleted += 1
    if delete_all:
        write_favs(project, set())
    else:
        favs = read_favs(project)
        write_favs(project, favs - {job.get("id") for job in targets})
    return {"ok": True, "deleted": deleted, "skipped": skipped}


@app.post("/forge/api/unload")
def unload_model():
    """Release the cached Diffusers pipeline and CUDA allocations."""
    global PIPE, PIPE_MODEL, PIPE_FAMILY
    with PIPE_LOCK:
        with STATE_LOCK:
            if CURRENT or QUEUE:
                raise HTTPException(409, "Stop the active render and clear the queue before unloading the model.")
        if PIPE is None:
            return {"ok": True, "state": "unloaded", "freed_cuda_mb": 0}
        del PIPE
        PIPE = None
        PIPE_MODEL = None
        PIPE_FAMILY = None
        gc.collect()
        freed = 0
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            freed = max(0, before - torch.cuda.memory_allocated()) // (1024 * 1024)
        return {"ok": True, "state": "unloaded", "freed_cuda_mb": int(freed)}


@app.post("/forge/api/settings")
def settings(body: dict[str, Any]):
    updates = {}
    if "last_project" in body:
        updates["last_project"] = safe_name(body["last_project"])
    if "hf_token" in body:
        updates["hf_token"] = str(body["hf_token"] or "").strip()
    save_settings(updates)
    return {"ok": True, "has_token": bool(read_settings().get("hf_token"))}


@app.post("/forge/api/open_folder")
def open_folder(body: dict[str, Any]):
    p = MODELS if body.get("which") == "models" else project_dir(body.get("project"))
    if os.name == "nt":
        os.startfile(str(p))
    return {"ok": True, "path": str(p)}


if __name__ == "__main__":
    if os.environ.get("FORGE_OPEN_BROWSER") == "1":
        def open_when_ready():
            import socket
            import webbrowser
            for _ in range(60):
                try:
                    with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                        webbrowser.open(f"http://127.0.0.1:{PORT}/forge/")
                        return
                except OSError:
                    time.sleep(1)
        threading.Thread(target=open_when_ready, daemon=True).start()
    print(f"[Character Forge] Direct Diffusers engine at http://127.0.0.1:{PORT}/forge/")
    print(f"[Character Forge] CUDA: {torch.cuda.is_available()} " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
    print(f"[Character Forge] Memory profile: {MEMORY_PROFILE['label']}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info", access_log=False)
