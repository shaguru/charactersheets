"""Optional character-sheet backend: Tom's CharacterForge (Qwen Rapid AIO) over its local HTTP API.

CharacterForge is slow to load, so we only start it when asked and reuse it if it is already running.
API (runtime/direct_server.py): POST /upload/image, POST /forge/api/queue, GET /forge/api/jobs, GET /view.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import requests

from mvm import settings


def _url(s):
    return s["forge_url"].rstrip("/")


def running(s: dict | None = None) -> bool:
    s = s or settings.load()
    try:
        return requests.get(_url(s) + "/forge/api/config", timeout=2).ok
    except requests.RequestException:
        return False


def installed(s: dict | None = None) -> bool:
    s = s or settings.load()
    return (Path(s["forge_dir"]) / "START.bat").exists()


def describe(s: dict | None = None) -> str:
    s = s or settings.load()
    if running(s):
        return f"running at {_url(s)}"
    return f"installed at {s['forge_dir']} (not running)" if installed(s) else "not installed"


def ensure_running(s: dict, timeout: float = 600) -> None:
    if running(s):
        return
    if not installed(s):
        raise SystemExit(f"CharacterForge not found at {s['forge_dir']}")
    print("Starting CharacterForge (this can take a few minutes)...")
    subprocess.Popen(["cmd", "/c", "start", "CharacterForge", str(Path(s["forge_dir"]) / "START.bat")],
                     cwd=s["forge_dir"])
    t0 = time.time()
    while time.time() - t0 < timeout:
        if running(s):
            return
        time.sleep(5)
    raise SystemExit("CharacterForge did not come up in time.")


def make_sheet(image: Path, out_dir: Path, layout: str = "turnaround5", notes: str = "", count: int = 1,
               project: str = "MVM") -> list[Path]:
    s = settings.load()
    ensure_running(s)
    base = _url(s)
    with open(image, "rb") as fh:
        up = requests.post(base + "/upload/image", files={"image": (image.name, fh)},
                           data={"subfolder": "forge_refs", "type": "input"}, timeout=60)
    up.raise_for_status()
    ref = up.json()
    cfg = requests.get(base + "/forge/api/config", timeout=10).json()
    models = [m["id"] for m in cfg.get("models", []) if m.get("installed") or m.get("status", {}).get("installed")]
    model_id = "qwen-rapid-aio" if "qwen-rapid-aio" in models or not models else models[0]
    body = {"refs": [f"{ref['subfolder']}/{ref['name']}"], "models": [model_id], "mode": "sheet", "sheet": layout,
            "notes": notes, "count": count, "project": project}
    q = requests.post(base + "/forge/api/queue", json=body, timeout=30)
    if not q.ok:
        raise SystemExit(f"CharacterForge refused the job: {q.text}")
    batch = q.json()["batch"]
    print(f"Queued {q.json()['queued']} sheet(s) in CharacterForge (batch {batch}); waiting...")
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        jobs = [j for j in requests.get(base + "/forge/api/jobs", params={"project": project}, timeout=30).json()["jobs"]
                if j.get("batch") == batch]
        if jobs and all(j["status"] in ("done", "error", "cancelled", "interrupted") for j in jobs):
            break
        time.sleep(5)
    saved = []
    for j in jobs:
        for img in j.get("images", []):
            r = requests.get(base + "/view", params=img, timeout=60)
            r.raise_for_status()
            dst = out_dir / f"{image.stem}_{layout}_forge_{img['filename']}"
            dst.write_bytes(r.content)
            saved.append(dst)
        if j["status"] != "done":
            print(f"  job {j['id']}: {j['status']} {j.get('error', '')}")
    return saved
