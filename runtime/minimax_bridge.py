"""MiniMax runs in its existing local environment, sharing the Forge UI and queue."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_cache = {}
_status_lock = threading.Lock()

IMAGE_MODEL = {
    "id": "minimax-h3-image", "name": "MiniMax H3-Image", "maker": "MiniMax / Unsloth GGUF",
    "family": "minimax-image", "released": "2026-11", "params": "33B · UD-Q4_K_XL",
    "license": "MiniMax H3-Image Community", "commercial": "Check the MiniMax community license",
    "commercial_ok": False, "gated": False, "max_refs": 3, "recommended": False,
    "speed": "6 steps · CUDA + CPU offload", "defaults": {"steps": 6, "cfg": 1.0},
    "sampler": "Euler · H3-Image schedule", "guidance_name": "distilled guidance",
    "total_gb": 34.2, "external": True, "blurb": "Use the same character-sheet, scene and free-prompt presets with H3-Image. Shares the existing MiniMax runtime and model cache.",
    "files": [],
}
VIDEO_MODEL = {**IMAGE_MODEL, "id": "minimax-h3-video", "name": "MiniMax H3 Video",
               "maker": "MiniMax / abhishekchohan int8", "license": "MiniMax H3 Community",
               "family": "minimax-video", "params": "H3 Ref2VA · int8", "max_refs": 1,
               "defaults": {"steps": 12, "cfg": 1.0}, "speed": "Two-pass CPU offload",
               "sampler": "H3 video schedule", "total_gb": 80.6,
               "blurb": "Animate a selected image or uploaded reference. Uses the existing MiniMax video weights in place."}


def directory() -> Path:
    try:
        cfg = json.loads((ROOT / "data/user/forge_settings.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        cfg = {}
    configured = os.environ.get("MINIMAX_DIR") or cfg.get("minimax_dir")
    if configured:
        return Path(configured)
    sibling = ROOT.parent / "MINIMAX_VIDEO_Maker"
    return sibling if (sibling / "runtime/.venv/Scripts/python.exe").is_file() else ROOT / "runtime/minimax_app"


def environment():
    env = os.environ.copy()
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "PYTHONPATH", "VIRTUAL_ENV",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(key, None)
    env.update(MVM_FORGE_BRIDGE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def status() -> dict:
    root = directory()
    py = root / "runtime/.venv/Scripts/python.exe"
    with _status_lock:
        if _cache.get("root") == str(root) and time.monotonic() - _cache.get("time", 0) < 10:
            return dict(_cache["value"])
        value = {"models": {"image": False, "video": False}, "directory": str(root)}
        if py.is_file() and (root / "mvm/forge_bridge.py").is_file():
            try:
                result = subprocess.run([str(py), "-m", "mvm", "forge-bridge"], cwd=root, env=environment(),
                                        capture_output=True, text=True, encoding="utf-8", timeout=15)
                if result.returncode == 0:
                    value.update(json.loads(result.stdout.strip().splitlines()[-1]))
                else:
                    value["error"] = result.stderr[-400:]
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                value["error"] = str(exc)
        _cache.update(root=str(root), time=time.monotonic(), value=value)
        return value


def model_status(kind="image") -> dict:
    ready = bool(status()["models"].get(kind))
    size = 34_200_000_000 if kind == "image" else 80_600_000_000
    return {"installed": ready, "bytes_done": size if ready else 0,
            "bytes_total": size, "bytes_needed": 0 if ready else size}


@contextmanager
def gpu_slot():
    """Use the same OS lock as standalone MiniMax jobs for Forge image engines."""
    root = directory()
    if not (root / "mvm/gpulock.py").exists():
        yield
        return
    guard_path = root / "runtime/gpu.guard"
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    guard = guard_path.open("a+b")
    guard.seek(0, os.SEEK_END)
    if guard.tell() == 0:
        guard.write(b"0")
        guard.flush()
    guard.seek(0)
    try:
        import msvcrt
        msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        guard.close()
        raise RuntimeError("MiniMax is using the GPU. Wait for that job to finish before rendering with Qwen/SDXL.") from exc
    try:
        yield
    finally:
        guard.close()


def progress(text: str):
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    events = []
    native = list(re.finditer(r"\|\s*(\d+)/(\d+)\s*-\s*([^\r\n]*)", clean))
    if native:
        match = native[-1]
        step, total = int(match[1]), int(match[2])
        loading = total > 20 or bool(re.search(r"[KMG]?B/s", match[3]))
        stage = f"loading model weights {step}/{total}" if loading else f"sampling {step}/{total}"
        events.append((match.start(), stage, {"progress": (0.08 if loading else 0.15) + 0.75 * step / max(total, 1),
                                            "step": 0 if loading else step}))
    bars = list(re.finditer(r"([^\r\n]*?)(\d+)%\|[^\r\n]*?(\d+)/(\d+)", clean))
    if bars:
        m = bars[-1]
        if "loading" in m[1].lower() or int(m[4]) > 100:
            events.append((m.start(), f"loading model weights {m[3]}/{m[4]}",
                           {"progress": .05 + .1 * int(m[2])/100, "step": 0}))
        else:
            events.append((m.start(), f"sampling {m[3]}/{m[4]}",
                           {"progress": .1 + .8 * int(m[2])/100, "step": int(m[3])}))
    lines = list(re.finditer(r"^\[\d\d:\d\d:\d\d\]\s*([^\r\n]*)", clean, re.MULTILINE))
    if lines:
        events.append((lines[-1].start(), lines[-1][1].strip(), {}))
    return max(events, key=lambda event: event[0])[1:] if events else ("loading MiniMax", {})


def run(job: dict, references: list[Path], update, cancelled):
    root = directory()
    py = root / "runtime/.venv/Scripts/python.exe"
    request_dir = ROOT / "data/bridge"
    request_dir.mkdir(parents=True, exist_ok=True)
    request = {"kind": "video" if job["mode"] == "video" else "image", "id": job["id"],
               "prompt": job["prompt"], "refs": [str(p) for p in references], "output": job["output_file"],
               "width": job["width"], "height": job["height"], "seed": job["seed"] % 2**31,
               "steps": job["steps"], "seconds": job.get("seconds", 5)}
    req = request_dir / f"{job['id']}.json"
    req.write_text(json.dumps(request), encoding="utf-8")
    log = req.with_suffix(".log")
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen([str(py), "-u", "-m", "mvm", "forge-bridge", "--request", str(req)],
                                cwd=root, env=environment(), stdout=fh, stderr=subprocess.STDOUT)
    try:
        while proc.poll() is None:
            cancelled()
            stage, fields = progress(log.read_text(encoding="utf-8", errors="replace")[-16000:])
            update(stage, **fields)
            time.sleep(1)
        if proc.returncode:
            tail = log.read_text(encoding="utf-8", errors="replace")[-2500:]
            errors = [line for line in tail.splitlines() if re.search(r"Error|Exception|failed|Another GPU|CharacterForge", line)]
            raise RuntimeError((errors[-1] if errors else tail[-400:]) + f" (log: {log})")
        if not Path(job["output_file"]).is_file():
            raise RuntimeError(f"MiniMax did not save an output. See {log}")
    finally:
        if proc.poll() is None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
            proc.wait(timeout=30)
