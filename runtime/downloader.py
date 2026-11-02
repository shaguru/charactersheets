"""Resumable model downloader. Used by the web UI and by SETUP (command line).

    python downloader.py --models-dir ..\\data\\models --ids qwen-rapid-aio
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "models.json")
GiB = 1024 ** 3


def load_catalog():
    with open(CATALOG, "r", encoding="utf-8") as f:
        return json.load(f)


def file_path(models_dir, f):
    return os.path.join(models_dir, f["dir"], f["name"])


def file_state(models_dir, f):
    p = file_path(models_dir, f)
    if os.path.isfile(p) and os.path.getsize(p) == f["size"]:
        return "ok", f["size"]
    part = p + ".part"
    if os.path.isfile(part):
        return "partial", os.path.getsize(part)
    if os.path.isfile(p):
        return "wrong-size", os.path.getsize(p)
    return "missing", 0


def model_status(models_dir, model):
    done = total = needed = 0
    missing = 0
    for f in model["files"]:
        st, n = file_state(models_dir, f)
        total += f["size"]
        if st == "ok":
            done += f["size"]
            continue
        missing += 1
        have = n if st in ("partial", "wrong-size") else 0
        done += have
        needed += max(0, f["size"] - have)
    return {"installed": missing == 0, "bytes_done": done, "bytes_total": total, "bytes_needed": needed}


class GatedError(Exception):
    pass


def _curl_exe():
    c = shutil.which("curl.exe") or shutil.which("curl")
    return c


def _download_curl(curl, url, part, token, progress, cancel):
    cmd = [curl, "-L", "--fail", "-s", "-S", "--retry", "5", "--retry-delay", "3",
           "-w", "%{http_code} %{url_effective}", "-C", "-", "-o", part, url]
    if token:
        cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    while p.poll() is None:
        if cancel is not None and cancel.is_set():
            p.kill()
            raise InterruptedError("cancelled")
        if os.path.exists(part):
            progress(os.path.getsize(part))
        time.sleep(1.0)
    err = (p.stderr.read() or b"").decode("utf-8", "replace")
    info = (p.stdout.read() or b"").decode("utf-8", "replace").strip()
    if p.returncode == 22 and ("401" in err or "403" in err) and "://huggingface.co/" in info:
        raise GatedError(err.strip())
    if p.returncode == 33:  # range not satisfiable -> already complete or bad part
        return
    if p.returncode != 0:
        raise RuntimeError(f"curl failed ({p.returncode}): {err.strip()[:300]}")


def _download_urllib(url, part, token, progress, cancel):
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403) and "://huggingface.co/" in (e.url or url):
            raise GatedError(str(e))
        if e.code == 416:
            return
        raise
    mode = "ab" if have and r.status == 206 else "wb"
    if mode == "wb":
        have = 0
    with open(part, mode) as out:
        while True:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            have += len(chunk)
            progress(have)


def download_file(models_dir, f, token=None, progress=None, cancel=None):
    p = file_path(models_dir, f)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.isfile(p) and os.path.getsize(p) == f["size"]:
        return
    if os.path.isfile(p):
        os.replace(p, p + ".part")  # wrong size: try to resume it
    part = p + ".part"
    progress = progress or (lambda n: None)
    curl = _curl_exe()
    for attempt in range(3):
        if curl:
            _download_curl(curl, f["url"], part, token, progress, cancel)
        else:
            _download_urllib(f["url"], part, token, progress, cancel)
        size = os.path.getsize(part) if os.path.exists(part) else 0
        if size == f["size"]:
            expected = f.get("sha256")
            if expected:
                digest = hashlib.sha256()
                with open(part, "rb") as check:
                    for chunk in iter(lambda: check.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                actual = digest.hexdigest()
                if actual.lower() != expected.lower():
                    os.remove(part)
                    raise RuntimeError(f"{f['name']}: SHA-256 mismatch (expected {expected}, got {actual})")
            os.replace(part, p)
            return
        if size > f["size"]:
            os.remove(part)  # corrupt, start over
    raise RuntimeError(f"{f['name']}: size mismatch after download")


def free_bytes(path):
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free


def download_model(models_dir, model, token=None, on_progress=None, cancel=None):
    """on_progress(dict) gets {file, file_done, file_total, done, total}."""
    st = model_status(models_dir, model)
    need = st["bytes_needed"]
    if need > free_bytes(models_dir) - 1 * GiB:
        raise RuntimeError(f"Not enough disk space: need {need / 1e9:.1f} GB, "
                           f"have {free_bytes(models_dir) / 1e9:.1f} GB free")
    total = st["bytes_total"]
    for f in model["files"]:
        if file_state(models_dir, f)[0] == "ok":
            continue
        base_done = sum(x["size"] for x in model["files"] if file_state(models_dir, x)[0] == "ok")

        def prog(n, f=f, base_done=base_done):
            if on_progress:
                on_progress({"file": f["name"], "file_done": n, "file_total": f["size"],
                             "done": base_done + n, "total": total})
        try:
            download_file(models_dir, f, token if "huggingface.co" in f["url"] else None, prog, cancel)
        except GatedError:
            raise GatedError(
                f"{model['name']} is gated on Hugging Face. Open {model['gate_url']}, sign in, click "
                f"'Agree and access', then paste a Hugging Face READ token in the Models tab (or SETUP).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cat = load_catalog()
    if a.list:
        for m in cat["models"]:
            st = model_status(a.models_dir, m)
            print(f"{m['id']:18} {m['total_gb']:5.1f} GB  {'INSTALLED' if st['installed'] else ''}")
        return
    ids = [i.strip() for i in a.ids.split(",") if i.strip()]
    failed = []
    for mid in ids:
        m = next((x for x in cat["models"] if x["id"] == mid), None)
        if not m:
            print(f"Unknown model id {mid}")
            failed.append(mid)
            continue
        print(f"\n=== {m['name']}  ({m['total_gb']} GB)  license: {m['license']}")
        last = [0.0]

        def show(p):
            now = time.time()
            if now - last[0] > 1.0:
                last[0] = now
                sys.stdout.write(f"\r  {p['file'][:46]:46}  {p['done'] / 1e9:6.2f} / {p['total'] / 1e9:.2f} GB   ")
                sys.stdout.flush()
        try:
            download_model(a.models_dir, m, a.token or None, show)
            print(f"\r  {m['name']} ready." + " " * 60)
        except Exception as e:
            print(f"\n  FAILED: {e}")
            failed.append(mid)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
