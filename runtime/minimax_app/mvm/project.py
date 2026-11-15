"""Job/project folders. Each job is self-describing so any agent can open it and continue.

projects/<name>/
  project.json      settings for this job (canvas, steps, song path, character map)
  direction.md      the creative brief in plain words (written by the human or agent)
  shots.json        the shot list: timing, prompt, references, seed, status
  assets/characters character / prop / location reference images
  assets/song       the song (one audio file)
  sheets/           generated character sheets
  work/             song slices, intermediate files
  renders/          one mp4 per shot (+ take number)
  final/            assembled music video
  logs/             render logs
"""
from __future__ import annotations

import json
import uuid
import re
import shutil
import subprocess
from pathlib import Path

from mvm import settings

PROJECTS = settings.ROOT / "projects"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
SUBDIRS = ["assets/characters", "assets/song", "sheets", "images", "work", "renders", "final", "logs"]

DIRECTION_TEMPLATE = """# Direction

Describe the video here in plain words: story, mood, look, locations, who appears when.

## Look
-

## Characters
(one line per image in assets/characters: `name` - who they are, what they wear)

## Story beats
- 0:00 -
"""


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_") or "job"


def path(name: str) -> Path:
    p = PROJECTS / slug(name)
    if not (p / "project.json").exists():
        raise SystemExit(f"No project '{name}'. Create it with: mvm new {name}")
    return p


def load_json(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def save_json(p: Path, data) -> None:
    tmp = p.with_suffix(p.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def create(name: str, assets: str | None = None, song: str | None = None) -> Path:
    p = PROJECTS / slug(name)
    for d in SUBDIRS:
        (p / d).mkdir(parents=True, exist_ok=True)
    if assets:
        src = Path(assets)
        for f in sorted(src.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                shutil.copy2(f, p / "assets/characters" / f.name)
            elif f.suffix.lower() in AUDIO_EXT and not song:
                song = str(f)
    if song:
        shutil.copy2(song, p / "assets/song" / Path(song).name)
    if not (p / "direction.md").exists():
        (p / "direction.md").write_text(DIRECTION_TEMPLATE, encoding="utf-8")
    s = settings.load()
    proj = load_json(p / "project.json", {}) or {}
    proj.setdefault("name", p.name)
    proj.setdefault("width", s["width"])
    proj.setdefault("height", s["height"])
    proj.setdefault("steps", s["steps"])
    proj.setdefault("fps", s["fps"])
    proj["characters"] = characters(p)
    proj["song"] = song_file(p)
    save_json(p / "project.json", proj)
    return p


def characters(p: Path) -> dict:
    """Reference name -> relative path. Name = file stem, lower-case."""
    proj = load_json(p / "project.json", {}) or {}
    out = {key: rel for key, rel in proj.get("references", {}).items()
           if (p / rel).resolve().is_relative_to(p.resolve()) and (p / rel).is_file()}
    for f in sorted((p / "assets/characters").iterdir()):
        if f.suffix.lower() in IMG_EXT:
            out[slug(f.stem).lower()] = f.relative_to(p).as_posix()
    return out


def song_file(p: Path) -> str | None:
    for f in sorted((p / "assets/song").iterdir()):
        if f.suffix.lower() in AUDIO_EXT:
            return f.relative_to(p).as_posix()
    return None


def duration(media: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(media)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def plan(p: Path, shot_len: float = 7.0, overwrite: bool = False) -> dict:
    """Split the song into H3-sized windows (5-15 s) and write a skeleton shots.json.
    Prompts are left for the human/agent to write from direction.md."""
    shots_file = p / "shots.json"
    if shots_file.exists() and not overwrite:
        raise SystemExit("shots.json already exists. Edit it, or pass --overwrite to regenerate the skeleton.")
    proj = load_json(p / "project.json")
    proj["characters"] = characters(p)
    proj["song"] = song_file(p)
    save_json(p / "project.json", proj)
    if not proj["song"]:
        raise SystemExit("Put a song in assets/song first.")
    shot_len = min(15.0, max(5.0, shot_len))
    total = duration(p / proj["song"])
    shots, t, i = [], 0.0, 1
    while t < total - 0.05:
        end = min(total, t + shot_len)
        if total - end < 5.0 and end < total:  # don't leave a tail shorter than H3's 5 s minimum
            end = total if total - t <= 15.0 else t + (total - t) / 2
        shots.append({
            "id": f"s{i:03d}", "start": round(t, 3), "end": round(end, 3),
            "prompt": "TODO: describe this shot. Refer to references as <Picture 1>, <Picture 2> in refs order.",
            "refs": list(proj["characters"].keys())[:2],
            "audio_ref": True,      # feed this slice of the song as <Audio 1> (lip-sync / rhythm)
            "seed": 1000 + i,
            "takes": [],            # filled by render
            "pick": None,           # which take goes into the edit (default: latest)
            "status": "todo",       # todo | ready | rendered | approved
        })
        t, i = end, i + 1
    data = {"song": proj["song"], "duration": total, "shots": shots}
    save_json(shots_file, data)
    return data


def shots(p: Path) -> dict:
    data = load_json(p / "shots.json")
    if not data:
        raise SystemExit("No shots.json yet. Run: mvm plan <project>")
    return data


def status(p: Path) -> str:
    proj = load_json(p / "project.json")
    data = load_json(p / "shots.json")
    lines = [f"Project {proj['name']}  canvas {proj['width']}x{proj['height']}  steps {proj['steps']}",
             f"Song: {proj.get('song')}", f"Characters: {', '.join(proj.get('characters', {})) or '-'}"]
    if data:
        from collections import Counter
        c = Counter(s["status"] for s in data["shots"])
        lines.append(f"Shots: {len(data['shots'])}  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
        todo = [s["id"] for s in data["shots"] if "TODO" in s["prompt"]]
        if todo:
            lines.append(f"Prompts still TODO: {', '.join(todo[:12])}{' ...' if len(todo) > 12 else ''}")
    finals = sorted((p / "final").glob("*.mp4"))
    if finals:
        lines.append(f"Latest cut: {finals[-1]}")
    return "\n".join(lines)
