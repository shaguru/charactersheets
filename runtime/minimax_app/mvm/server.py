"""Optional local web page. Listens on 127.0.0.1 only (no login: single user, not reachable from the network).
Long jobs run as `mvm` CLI subprocesses, so the UI and agents use exactly the same code path."""
from __future__ import annotations

import shutil
import os
import subprocess
import sys
import time
import threading
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from mvm import gpulock, media, models, project, settings, studio

app = FastAPI(title="MiniMax Video Maker")
WEB = Path(__file__).parent / "web"
JOBS: dict[str, dict] = {}  # job id -> {proc, log, cmd}
GPU_ACTIONS = {"render", "sheet", "image", "video"}
SUBMIT_LOCK = threading.Lock()
CANVAS = {"fast": (768, 416),  # sides must be multiples of 32
           "standard": (960, 544), "native": (1344, 768)}  # 16:9; native = what H3 was trained on


def _running(gpu_only=False):
    return [k for k, v in JOBS.items() if v["proc"].poll() is None and (not gpu_only or v["gpu"])]


def _clean_log(text: str) -> str:
    # tqdm redraws with \r; keep only the last redraw of each line
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    lines = []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.split("\r")[-1]
        if re.match(r"\s*\|[# ]*\|", line) and lines and re.match(r"\s*\|[# ]*\|", lines[-1]):
            lines[-1] = line
        else:
            lines.append(line)
    return "\n".join(lines)


def _run(args: list[str], log: Path, gpu: bool = False) -> str:
    jid = time.strftime("%H%M%S") + f"_{len(JOBS)}"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w", encoding="utf-8")
    try:
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen([sys.executable, "-u", "-m", "mvm", *args], stdout=fh, stderr=subprocess.STDOUT,
                                cwd=settings.ROOT, env=env)
    finally:
        fh.close()
    JOBS[jid] = {"proc": proc, "log": log, "cmd": " ".join(args), "gpu": gpu}
    return jid


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/api/state")
def state():
    projs = sorted(d.name for d in project.PROJECTS.glob("*") if (d / "project.json").exists())
    jobs = [{"id": k, "cmd": v["cmd"], "running": v["proc"].poll() is None, "code": v["proc"].poll()} for k, v in JOBS.items()]
    h = gpulock.holder()
    from mvm import image_native
    return {"projects": projs, "models": models.status(), "jobs": jobs[-10:], "running": _running(),
            "image_install": image_native.download_progress(),
            "disk_free_gib": round(shutil.disk_usage(settings.models_dir()).free / 2**30, 1),
            "gpu_busy": h["what"] if h else None, "quality_presets": list(media.QUALITY), "canvas_presets": CANVAS}


@app.post("/api/project")
def new_project(name: str = Form(...)):
    p = project.create(name)
    return {"path": str(p), "name": p.name}


@app.get("/api/project/{name}")
def get_project(name: str):
    p = project.path(name)
    files = lambda sub: sorted(f.relative_to(p).as_posix() for f in (p / sub).glob("*") if f.is_file())  # noqa: E731
    return {"project": project.load_json(p / "project.json"), "direction": (p / "direction.md").read_text(encoding="utf-8"),
            "shots": project.load_json(p / "shots.json"), "status": project.status(p),
            "characters": files("assets/characters"), "sheets": files("sheets"), "final": files("final"),
            "images": [f for f in files("images") if Path(f).suffix.lower() in project.IMG_EXT]}


def _require_idle():
    busy, h = _running(gpu_only=True), gpulock.holder()
    if busy or h:
        raise HTTPException(409, f"A GPU job is already running ({h['what'] if h else JOBS[busy[0]]['cmd']}). "
                                 "Wait for it to finish or press Stop.")


@app.post("/api/project/{name}/image")
def create_image(name: str, body: dict):
    with SUBMIT_LOCK:
        _require_idle()
        if not models.status()["image"]:
            raise HTTPException(409, "H3-Image is not installed yet. Run mvm.bat download image.")
        try:
            req = studio.image_request(name, body)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        p = project.path(name)
        return {"job": _run(["image", name, "--request", str(req)], p / "logs" / f"{req.stem}.log", gpu=True)}


@app.post("/api/project/{name}/video")
def create_video(name: str, body: dict):
    with SUBMIT_LOCK:
        _require_idle()
        if not models.status()["video"]:
            raise HTTPException(409, "H3 video model is not downloaded.")
        try:
            sid = studio.video_shot(name, body)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"job": _run(["render", name, "--shots", sid], project.path(name) / "logs" / f"{sid}.log", gpu=True),
                "shot": sid}


@app.post("/api/project/{name}/upload")
async def upload(name: str, file: UploadFile = File(...)):
    p = project.path(name)
    ext = Path(file.filename or "").suffix.lower()
    if ext in project.IMG_EXT:
        dst = p / "assets/characters" / Path(file.filename).name
    elif ext in project.AUDIO_EXT:
        for old in (p / "assets/song").glob("*"):
            old.unlink()
        dst = p / "assets/song" / Path(file.filename).name
    else:
        raise HTTPException(400, "Images (png/jpg/webp) or a song (mp3/wav/flac/m4a)")
    with open(dst, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    project.create(name)  # refresh character map / song in project.json
    return {"saved": dst.relative_to(p).as_posix()}


@app.post("/api/project/{name}/direction")
def save_direction(name: str, text: str = Form(...)):
    (project.path(name) / "direction.md").write_text(text, encoding="utf-8")
    return {"ok": True}


@app.post("/api/project/{name}/shots")
async def save_shots(name: str, body: dict):
    project.save_json(project.path(name) / "shots.json", body)
    return {"ok": True}


def _run_action(name: str, action: str, shots: str, extra: str):
    p = project.path(name)
    if action in GPU_ACTIONS:
        busy = _running(gpu_only=True)
        h = gpulock.holder()
        if busy or h:
            raise HTTPException(409, f"A GPU job is already running ({h['what'] if h else JOBS[busy[0]]['cmd']}). "
                                     "Wait for it to finish or press Stop.")
    if action == "plan":
        args = ["plan", name] + (["--shot-len", extra] if extra else [])
    elif action == "render":
        args = ["render", name] + (["--shots", *shots.split()] if shots.strip() else []) + (["--force"] if extra == "force" else [])
    elif action == "assemble":
        args = ["assemble", name]
    elif action == "sheet":
        try:
            img = studio.image_file(p, shots)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if extra not in ("", "auto", "h3image", "forge"):
            raise HTTPException(400, "Unknown sheet backend")
        args = ["sheet", str(img), "--project", name] + (["--backend", extra] if extra else [])
    else:
        raise HTTPException(400, "unknown action")
    return {"job": _run(args, p / "logs" / f"{action}_{time.strftime('%Y%m%d-%H%M%S')}.log", gpu=action in GPU_ACTIONS)}


@app.post("/api/project/{name}/run/{action}")
def run_action(name: str, action: str, shots: str = Form(""), extra: str = Form("")):
    with SUBMIT_LOCK:
        return _run_action(name, action, shots, extra)


@app.post("/api/project/{name}/settings")
def save_settings(name: str, canvas: str = Form(""), steps: int = Form(0), quality: str = Form("")):
    p = project.path(name)
    proj = project.load_json(p / "project.json")
    if canvas in CANVAS:
        proj["width"], proj["height"] = CANVAS[canvas]
    if 4 <= steps <= 50:
        proj["steps"] = steps
    if quality in media.QUALITY:
        proj["quality"] = quality
    project.save_json(p / "project.json", proj)
    return {"project": proj}


@app.post("/api/job/{jid}/stop")
def stop_job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    if j["proc"].poll() is None:  # kill the whole tree (the venv launcher spawns the real python)
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(j["proc"].pid)], capture_output=True)
    return {"stopped": True}


@app.get("/api/job/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404)
    text = j["log"].read_text(encoding="utf-8", errors="replace") if j["log"].exists() else ""
    return {"running": j["proc"].poll() is None, "code": j["proc"].poll(), "cmd": j["cmd"],
            "log": _clean_log(text[-20000:])[-6000:]}


@app.get("/file/{name}/{rel:path}")
def file(name: str, rel: str):
    p = project.path(name)
    f = (p / rel).resolve()
    if p.resolve() not in f.parents or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


def run_server(port: int | None = None):
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port or settings.load()["ui_port"], log_level="warning")


run = run_server  # cli calls server.run
