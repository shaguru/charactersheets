"""mvm - MiniMax Video Maker command line. Run `mvm <command> -h` for details."""
from __future__ import annotations

import argparse
import sys

from mvm import settings

settings.setup_env()


def cmd_doctor(a):
    import shutil
    import torch
    s = settings.load()
    print(f"python   {sys.version.split()[0]}")
    print(f"torch    {torch.__version__}  cuda {torch.version.cuda}")
    ok = torch.cuda.is_available()
    print(f"gpu      {torch.cuda.get_device_name(0) if ok else 'NOT AVAILABLE'}")
    if ok:
        free, total = torch.cuda.mem_get_info()
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        print(f"vram     {free/2**30:.1f} / {total/2**30:.1f} GB free  bf16 matmul ok={bool((x @ x).isfinite().all())}")
    try:
        import diffusers
        from diffusers import ModularPipeline  # noqa: F401
        print(f"diffusers {diffusers.__version__}")
    except Exception as e:  # pragma: no cover
        print(f"diffusers MISSING ({e})")
    try:
        import psutil
        print(f"ram      {psutil.virtual_memory().total/2**30:.0f} GB")
    except ImportError:
        pass
    md = settings.models_dir(s)
    print(f"models   {md}  ({shutil.disk_usage(md).free/2**30:.0f} GB free on that drive)")
    print(f"ffmpeg   {shutil.which('ffmpeg') or 'MISSING'}")
    from mvm import models
    for k, v in models.status(s).items():
        print(f"model    {k:6s} {'downloaded' if v else 'not downloaded'}")
    from mvm import forge
    print(f"forge    {forge.describe(s)}")


def cmd_new(a):
    from mvm import project
    p = project.create(a.name, a.assets, a.song)
    print(p)
    print(project.status(p))


def cmd_plan(a):
    from mvm import project
    p = project.path(a.project)
    data = project.plan(p, a.shot_len, a.overwrite)
    print(f"Wrote {p / 'shots.json'}: {len(data['shots'])} shots over {data['duration']:.1f}s. Now write the prompts.")


def cmd_status(a):
    from mvm import project
    print(project.status(project.path(a.project)))


def cmd_render(a):
    from mvm import gpulock, render
    if a.dry_run:
        return render.render_project(a.project, only=a.shots, takes=a.takes, dry_run=True, force=a.force)
    with gpulock.gpu_job(f"render {a.project}"):
        render.render_project(a.project, only=a.shots, takes=a.takes, dry_run=False, force=a.force)


def cmd_assemble(a):
    from mvm import assemble
    print(assemble.assemble(a.project, a.out))


def cmd_sheet(a):
    from mvm import gpulock, sheets
    with gpulock.gpu_job(f"sheet {a.image}"):
        files = sheets.make_sheet(a.image, project=a.project, backend=a.backend, layout=a.layout, notes=a.notes, count=a.count)
    for f in files:
        print(f)


def cmd_download(a):
    from mvm import models
    models.download(a.which)


def cmd_image(a):
    from mvm import gpulock, studio
    with gpulock.gpu_job(f"image {a.project}"):
        for f in studio.generate_image(a.project, a.request):
            print(f)


def cmd_forge(a):
    from mvm import forge_bridge
    if a.request:
        forge_bridge.run(a.request)
    else:
        import json
        print(json.dumps(forge_bridge.status()))


def cmd_config(a):
    if a.set:
        upd = {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            upd[k] = int(v) if v.isdigit() else v
        settings.save(upd)
    import json
    print(json.dumps(settings.load(), indent=2))


def cmd_ui(a):
    from mvm import server
    server.run(port=a.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mvm", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check GPU, packages, models, disk").set_defaults(f=cmd_doctor)

    p = sub.add_parser("new", help="create a project folder")
    p.add_argument("name")
    p.add_argument("--assets", help="folder of reference images (and a song) to copy in")
    p.add_argument("--song", help="song file to copy in")
    p.set_defaults(f=cmd_new)

    p = sub.add_parser("plan", help="split the song into shots and write a skeleton shots.json")
    p.add_argument("project")
    p.add_argument("--shot-len", type=float, default=7.0, help="seconds per shot, 5-15 (default 7)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(f=cmd_plan)

    p = sub.add_parser("status", help="show project progress")
    p.add_argument("project")
    p.set_defaults(f=cmd_status)

    p = sub.add_parser("render", help="render shots with MiniMax-H3 on the GPU")
    p.add_argument("project")
    p.add_argument("--shots", nargs="*", help="shot ids (default: every shot without a take)")
    p.add_argument("--takes", type=int, default=1, help="takes per shot (new seed each)")
    p.add_argument("--force", action="store_true", help="re-render shots that already have takes")
    p.add_argument("--dry-run", action="store_true", help="validate and print what would render")
    p.set_defaults(f=cmd_render)

    p = sub.add_parser("assemble", help="cut picked takes to the song")
    p.add_argument("project")
    p.add_argument("--out", help="output file name (in final/)")
    p.set_defaults(f=cmd_assemble)

    p = sub.add_parser("sheet", help="make a character sheet from one picture")
    p.add_argument("image")
    p.add_argument("--project", help="save into this project's sheets/ (and add as a reference)")
    p.add_argument("--backend", choices=["auto", "h3image", "forge"], default="auto")
    p.add_argument("--layout", default="turnaround5", help="turnaround5 | turnaround3 | expressions | poses")
    p.add_argument("--notes", default="")
    p.add_argument("--count", type=int, default=1)
    p.set_defaults(f=cmd_sheet)

    p = sub.add_parser("download", help="download model weights")
    p.add_argument("which", choices=["video", "image", "all"])
    p.set_defaults(f=cmd_download)

    p = sub.add_parser("image", help="generate an H3 image from a saved studio request")
    p.add_argument("project")
    p.add_argument("--request", required=True)
    p.set_defaults(f=cmd_image)

    p = sub.add_parser("forge-bridge", help="isolated image/video worker for the Character Forge UI")
    p.add_argument("--request")
    p.set_defaults(f=cmd_forge)

    p = sub.add_parser("config", help="show or change settings.json, e.g. --set models_dir=I:/MVM_models")
    p.add_argument("--set", nargs="*")
    p.set_defaults(f=cmd_config)

    p = sub.add_parser("ui", help="start the local web page (127.0.0.1 only)")
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(f=cmd_ui)

    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
