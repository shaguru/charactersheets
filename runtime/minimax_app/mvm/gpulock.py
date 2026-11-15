"""One GPU job at a time, across the CLI, the web page and agents. Two renders at once both load the 33 GB text
encoder into 64 GB of RAM and grind the machine to a halt."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

import psutil

from mvm import settings

LOCK = settings.ROOT / "runtime" / "gpu.lock"


def release_forge_model():
    """Release Forge's cached pipeline if its optional server was opened again."""
    if os.environ.get("MVM_FORGE_BRIDGE") == "1":
        return  # Forge unloaded its own pipeline before starting this worker.
    import requests
    url = settings.load().get("forge_url", "").rstrip("/")
    if not url:
        return
    try:
        response = requests.get(url + "/forge/api/config", timeout=3)
    except requests.ConnectionError:
        return
    except requests.Timeout as exc:
        raise SystemExit("CharacterForge is not responding. Close it before starting an H3 job.") from exc
    if not response.ok:
        return
    runtime = response.json().get("runtime", {})
    if runtime.get("busy") or runtime.get("queue"):
        raise SystemExit("CharacterForge is rendering. Wait for it or stop its queue before starting H3.")
    if runtime.get("state") in ("loaded", "loading"):
        print("Unloading CharacterForge to make GPU/RAM space for H3...", flush=True)
        response = requests.post(url + "/forge/api/unload", json={}, timeout=120)
        if not response.ok:
            raise SystemExit("CharacterForge could not unload its model. Close it before starting H3.")


def holder() -> dict | None:
    try:
        info = json.loads(LOCK.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if info.get("pid") and psutil.pid_exists(info["pid"]):
        try:
            if psutil.Process(info["pid"]).create_time() <= info.get("time", 0) + 5:
                return info
        except psutil.Error:
            pass
    return None


@contextmanager
def gpu_job(what: str):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    # OS lock closes the race between simultaneous CLI/UI submissions. The OS also
    # releases it when Stop kills a process, even if finally cannot run.
    guard = (LOCK.parent / "gpu.guard").open("a+b")
    # msvcrt.locking requires an existing byte, and reading a byte from a file
    # that another Windows process has locked raises PermissionError.
    guard.seek(0, os.SEEK_END)
    if guard.tell() == 0:
        guard.write(b"0")
        guard.flush()
    guard.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        guard.close()
        h = holder()
        raise SystemExit(f"Another GPU job is running ({h['what'] if h else 'starting'}). Wait for it or stop it first.")
    try:
        LOCK.write_text(json.dumps({"pid": os.getpid(), "what": what, "time": time.time()}), encoding="utf-8")
        release_forge_model()
        yield
    finally:
        try:
            if json.loads(LOCK.read_text(encoding="utf-8")).get("pid") == os.getpid():
                LOCK.unlink()
        except (OSError, ValueError):
            pass
        guard.close()
