"""Simple image -> selected reference -> video workflow, shared by CLI and web."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from mvm import project

IMAGE_SIZES = {"landscape": (2048, 1152), "square": (2048, 2048), "portrait": (1152, 2048),
               "landscape4mp": (2720, 1536), "square4mp": (2048, 2048)}


def image_file(p: Path, rel: str) -> Path:
    f = (p / rel).resolve()
    if not f.is_relative_to(p.resolve()) or f.suffix.lower() not in project.IMG_EXT or not f.is_file():
        raise ValueError("Choose an image from this project's gallery or upload one.")
    return f


def image_request(name: str, body: dict) -> Path:
    p = project.path(name)
    prompt = str(body.get("prompt", "")).strip()
    if not prompt or len(prompt) > 16000:
        raise ValueError("Enter an image description (maximum 16,000 characters).")
    size = body.get("size", "landscape")
    if size not in IMAGE_SIZES:
        raise ValueError("Choose an image size from the list.")
    refs = body.get("refs", [])
    if not isinstance(refs, list) or len(refs) > 8:
        raise ValueError("Choose up to 8 reference images.")
    refs = [str(image_file(p, str(rel)).relative_to(p.resolve()).as_posix()) for rel in refs]
    if refs and "<Picture 1>" not in prompt:
        prompt = "Edit <Picture 1>. " + prompt
    seed = body.get("seed")
    seed = int(seed) if seed is not None and seed != "" else int(time.time_ns() % 2**31)
    if not 0 <= seed < 2**31:
        raise ValueError("Seed must be between 0 and 2147483647.")
    request = p / "work" / "requests" / f"image_{uuid.uuid4().hex[:12]}.json"
    request.parent.mkdir(parents=True, exist_ok=True)
    project.save_json(request, {"prompt": prompt, "refs": refs, "size": size, "seed": seed})
    return request


def generate_image(name: str, request: str):
    from mvm import engine_image
    p = project.path(name)
    data = json.loads(Path(request).read_text(encoding="utf-8"))
    refs = [image_file(p, rel) for rel in data["refs"]]
    files = engine_image.generate(data["prompt"], refs, p / "images", IMAGE_SIZES[data["size"]],
                                  seed=data["seed"], stem=f"h3_{uuid.uuid4().hex[:6]}")
    for f in files:
        project.save_json(f.with_suffix(".json"), {**data, "model": "MiniMax-H3-Image", "file": f.name})
    return files


def video_shot(name: str, body: dict) -> str:
    p = project.path(name)
    image = image_file(p, str(body.get("image", "")))
    prompt = str(body.get("prompt", "")).strip()
    if not prompt or len(prompt) > 16000 or "TODO" in prompt:
        raise ValueError("Describe the subject's movement, camera and scene (maximum 16,000 characters).")
    seconds = float(body.get("seconds", 5))
    if not 5 <= seconds <= 15:
        raise ValueError("H3 video clips must last 5 to 15 seconds.")
    raw_seed = body.get("seed")
    seed = int(raw_seed) if raw_seed is not None and raw_seed != "" else int(time.time_ns() % 2**31)
    if not 0 <= seed < 2**31:
        raise ValueError("Seed must be between 0 and 2147483647.")
    proj = project.load_json(p / "project.json")
    key = f"studio_{uuid.uuid4().hex[:12]}"
    # Store an extra reference mapping; image files can stay in images/ or sheets/.
    proj.setdefault("references", {})[key] = image.relative_to(p.resolve()).as_posix()
    project.save_json(p / "project.json", proj)
    data = project.load_json(p / "shots.json") or {"song": proj.get("song"), "duration": 0, "shots": []}
    sid = f"v_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    if "<Picture 1>" not in prompt:
        prompt = "Use the subject and visual style of <Picture 1>. " + prompt
    data["shots"].append({"id": sid, "start": 0, "end": seconds, "prompt": prompt, "refs": [key],
                          "audio_ref": False, "seed": seed, "takes": [], "pick": None, "status": "ready",
                          "studio": True, "source_image": image.relative_to(p.resolve()).as_posix()})
    project.save_json(p / "shots.json", data)
    return sid
