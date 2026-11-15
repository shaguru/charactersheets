"""Cut the picked take of every shot to its song window and lay the original song underneath."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from mvm import media, project, settings


def picked_take(shot: dict) -> str | None:
    if not shot.get("takes"):
        return None
    if shot.get("pick") is not None:
        return next((t["file"] for t in shot["takes"] if t["take"] == shot["pick"]), shot["takes"][-1]["file"])
    return shot["takes"][-1]["file"]


def assemble(name: str, out: str | None = None) -> Path:
    p = project.path(name)
    proj = project.load_json(p / "project.json")
    data = project.shots(p)
    fps, w, h = proj["fps"], proj["width"], proj["height"]
    _, crf, x264p = media.preset(settings.load(), proj)
    work = p / "work" / "cut"
    work.mkdir(parents=True, exist_ok=True)
    parts, missing = [], []
    for s in data["shots"]:
        dur = s["end"] - s["start"]
        part = work / f"{s['id']}.mp4"
        take = picked_take(s)
        if take:
            src = ["-i", str(p / take)]
            vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},tpad=stop_mode=clone:stop_duration=15"
        else:  # placeholder so the cut still lines up with the song
            missing.append(s["id"])
            src = ["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps}"]
            vf = f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{s['id']} not rendered':fontcolor=white:fontsize=28:x=(w-tw)/2:y=(h-th)/2"
        subprocess.run(["ffmpeg", "-y", "-v", "error", *src, "-t", f"{dur:.3f}", "-vf", vf, "-an",
                        "-c:v", "libx264", "-preset", x264p, "-crf", str(crf), "-pix_fmt", "yuv420p", str(part)], check=True)
        parts.append(part)
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{x.as_posix()}'\n" for x in parts), encoding="utf-8")
    out_file = p / "final" / (out or f"{p.name}_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-i", str(p / data["song"]), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out_file)], check=True)
    if missing:
        print(f"Placeholders for unrendered shots: {', '.join(missing)}")
    return out_file
