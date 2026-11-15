"""Run an image/video request from Character Forge in this app's isolated runtime."""
from __future__ import annotations

import json
from pathlib import Path

from mvm import gpulock, project, settings


def status():
    from mvm import models
    return {"models": models.status(), "models_dir": str(settings.models_dir())}


def run(request_file: str):
    import shutil
    from mvm import engine_image, render
    request = json.loads(Path(request_file).read_text(encoding="utf-8"))
    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with gpulock.gpu_job(f"CharacterForge {request['kind']}"):
        if request["kind"] == "image":
            files = engine_image.generate(request["prompt"], [Path(p) for p in request["refs"]],
                                          output.parent, (request["width"], request["height"]),
                                          stem=output.stem, seed=request["seed"])
            if files[0] != output:
                files[0].replace(output)
        elif request["kind"] == "video":
            p = project.create("CharacterForge_Bridge")
            proj = project.load_json(p / "project.json")
            proj.update(width=request["width"], height=request["height"], steps=request["steps"],
                        quality="high", characters={"selected": request["refs"][0]})
            shot = {"id": request["id"], "start": 0, "end": request["seconds"], "prompt": request["prompt"],
                    "refs": ["selected"], "audio_ref": False, "seed": request["seed"], "takes": [], "status": "ready"}
            data = {"song": None, "duration": request["seconds"], "shots": [shot]}
            project.save_json(p / "shots.json", data)
            renderer = render.Renderer(p, proj)
            renderer.run(data, [shot], 1)
            result = project.shots(p)["shots"][0].get("takes", [])
            if not result:
                raise RuntimeError("Video renderer did not produce a clip. See the job log.")
            source = p / result[-1]["file"]
            # Output is normally on the same drive: rename avoids a second media copy.
            shutil.move(str(source), str(output))
        else:
            raise ValueError("Unknown CharacterForge request type")
    if not output.is_file():
        raise RuntimeError("MiniMax completed without an output file")
    print(f"FORGE_OUTPUT {output}", flush=True)
