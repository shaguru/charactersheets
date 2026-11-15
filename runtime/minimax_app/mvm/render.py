"""Render shots with MiniMax-H3 ref2va in two passes so it fits 64 GB RAM / 24 GB VRAM.

Pass 1: before_encode + Qwen3-VL text encoder (int8) for every shot -> prompt_embeds kept on CPU. Unload.
Pass 2: before_encode + VAE encoder + transformer_ref (int8, block-streamed from RAM) + decoders.
"""
from __future__ import annotations

import gc
import json
import subprocess
import time
from pathlib import Path

from mvm import models, offload, patches, project, settings  # noqa: F401 (patches applies fixes)

H3_FPS = 24


def snap_frames(seconds: float) -> int:
    """H3 decodes 17n+5 frames; pick the smallest >= the shot length, within 5-15 s."""
    n = max(5 * H3_FPS, round(seconds * H3_FPS))
    k = -(-(n - 5) // 17)
    frames = 17 * k + 5
    while frames > 15 * H3_FPS:
        frames -= 17
    return frames


def slice_audio(song: Path, start: float, end: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(song),
                    "-ac", "2", "-ar", "48000", str(out)], check=True)
    return out


def validate(p: Path, proj: dict, shot: dict) -> list[str]:
    errs = []
    if int(proj["width"]) % 32 or int(proj["height"]) % 32:
        errs.append(f"canvas {proj['width']}x{proj['height']} must be multiples of 32")
    dur = shot["end"] - shot["start"]
    if not 4.9 <= dur <= 15.01:
        errs.append(f"duration {dur:.2f}s outside 5-15 s")
    if "TODO" in shot["prompt"]:
        errs.append("prompt still TODO")
    if not shot.get("refs"):
        errs.append("needs at least one image ref (ref2va)")
    if len(shot.get("refs", [])) > 8:
        errs.append("max 8 image refs")
    for r in shot.get("refs", []):
        if r not in proj["characters"]:
            errs.append(f"unknown ref '{r}' (have: {', '.join(proj['characters'])})")
    return errs


def render_project(name: str, only=None, takes: int = 1, dry_run: bool = False, force: bool = False) -> None:
    p = project.path(name)
    proj = project.load_json(p / "project.json")
    proj["characters"] = project.characters(p)
    data = project.shots(p)
    todo = [s for s in data["shots"] if (only and s["id"] in only) or (not only and (force or not s.get("takes")))]
    bad = {s["id"]: e for s in todo if (e := validate(p, proj, s))}
    for sid, e in bad.items():
        print(f"skip {sid}: {'; '.join(e)}")
    todo = [s for s in todo if s["id"] not in bad]
    w, h = proj["width"], proj["height"]
    for s in todo:
        print(f"{s['id']}  {s['start']:.1f}-{s['end']:.1f}s  {snap_frames(s['end'] - s['start'])} frames  "
              f"{w}x{h}  refs={s['refs']}{' +audio' if s.get('audio_ref') else ''}  x{takes}")
    if dry_run or not todo:
        print("Nothing to render." if not todo else f"Dry run: {len(todo) * takes} clip(s).")
        return
    Renderer(p, proj).run(data, todo, takes)


class Renderer:
    def __init__(self, p: Path, proj: dict):
        self.p, self.proj = p, proj
        self.s = settings.load()
        # Resolve the installed snapshot once. In particular, Diffusers' sharded
        # VAE loader otherwise asks the Hub for revision metadata even offline.
        self.repo = str(models.snapshot(models.require("video")))
        self.log = open(p / "logs" / f"render_{time.strftime('%Y%m%d-%H%M%S')}.log", "a", encoding="utf-8")

    def say(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def references(self, shot: dict):
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3ImageReference
        refs = [MiniMaxH3ImageReference.from_file(str(self.p / self.proj["characters"][r])) for r in shot["refs"]]
        if shot.get("audio_ref"):
            wav = slice_audio(self.p / (self.proj.get("audio_ref_file") or self.proj["song"]), shot["start"], shot["end"], self.p / "work" / "audio" / f"{shot['id']}.wav")
            refs.append(MiniMaxH3AudioReference.from_file(str(wav)))
        return refs

    def _shrink_refs(self, pipe):
        # The release encodes reference images at a 2048 px short edge, i.e. ~16k vision tokens per image in the
        # Qwen3-VL conditioner. That overflows 24 GB and crawls; 1024 keeps identity detail at a quarter of the cost.
        pipe.register_to_config(reference_image_short_edge=int(self.s.get("ref_short_edge", 1024)))

    # ---------- pass 1 ----------
    def _embed_file(self, shot) -> Path:
        import hashlib
        key = json.dumps([shot["prompt"], shot["refs"], bool(shot.get("audio_ref")), float(shot["start"]), float(shot["end"]),
                          int(self.proj["width"]), int(self.proj["height"]), int(self.s.get("ref_short_edge", 1024)),
                          self.proj.get("audio_ref_file")], ensure_ascii=False)
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return self.p / "work" / "embeds" / f"{shot['id']}_{h}.pt"

    def encode_all(self, jobs):
        import torch
        out, missing = {}, []
        for shot, refs in jobs:  # cached prompt embeddings skip the 5-minute text-encoder load
            f = self._embed_file(shot)
            if f.exists():
                out[shot["id"]] = torch.load(f, weights_only=True)
            else:
                missing.append((shot, refs))
        if not missing:
            self.say("pass 1: all prompt embeddings cached")
            return out
        jobs = missing
        from diffusers import ModularPipeline
        self.say("pass 1: loading text encoder")
        t0 = time.time()
        wf = ModularPipeline.from_pretrained(self.repo, workflow="ref2va").blocks
        drop = [k for k in wf.sub_blocks if k not in ("before_encode", "text_encoder")]
        for k in drop:
            wf.sub_blocks.pop(k)
        enc = wf.init_pipeline(self.repo)
        self._shrink_refs(enc)
        enc.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=self.repo, local_files_only=True)
        enc.text_encoder.requires_grad_(False)
        info = offload.apply(enc.text_encoder, resident_gb=float(self.s.get("te_resident_gb", 8)))
        self.say(f"text encoder offload: {info}")
        self.say(f"text encoder ready in {time.time() - t0:.0f}s")
        with torch.no_grad():
            for shot, refs in jobs:
                st = enc(prompt=shot["prompt"], references=refs, num_frames=snap_frames(shot["end"] - shot["start"]),
                         height=self.proj["height"], width=self.proj["width"],
                         output=["prompt_embeds", "text_token_tags"])
                out[shot["id"]] = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in st.items()}
                f = self._embed_file(shot)
                f.parent.mkdir(parents=True, exist_ok=True)
                torch.save(out[shot["id"]], f)
                self.say(f"encoded {shot['id']} ({time.time() - t0:.0f}s since load start)")
        del enc, wf
        gc.collect()
        torch.cuda.empty_cache()
        return out

    # ---------- pass 2 ----------
    def load_video(self):
        import torch
        from diffusers import ModularPipeline
        self.say("pass 2: loading video transformer (int8, streamed from RAM)")
        t0 = time.time()
        wf = ModularPipeline.from_pretrained(self.repo, workflow="ref2va").blocks
        wf.sub_blocks.pop("text_encoder")
        pipe = wf.init_pipeline(self.repo)
        self._shrink_refs(pipe)
        pipe.load_components(names=["transformer_ref"], dtype=torch.bfloat16,
                             pretrained_model_name_or_path=self.repo, local_files_only=True)
        # VAEs straight onto the GPU: going through host RAM costs ~16 GB of transient fp32 copies
        pipe.load_components(names=["vae", "audio_vae"], dtype=torch.bfloat16, device_map="cuda",
                             pretrained_model_name_or_path=self.repo, local_files_only=True)
        pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=self.repo,
                             local_files_only=True)  # remaining small parts (schedulers, processors)
        missing = [name for name in ("transformer_ref", "vae", "audio_vae") if getattr(pipe, name, None) is None]
        if missing:
            raise RuntimeError(f"Could not load installed video components: {', '.join(missing)}. See the component error above.")
        tr = pipe.transformer_ref
        tr.requires_grad_(False)
        info = offload.apply(tr, resident_gb=float(self.s.get("dit_resident_gb", 0)))
        self.say(f"transformer offload: {info}")
        pipe.vae.to("cuda")
        pipe.audio_vae.to("cuda")
        self.say(f"video model ready in {time.time() - t0:.0f}s")
        return pipe

    def run(self, data, todo, takes):
        import torch
        from mvm import media
        crf, _, x264p = media.preset(self.s, self.proj)
        jobs = [(s, self.references(s)) for s in todo]
        embeds = self.encode_all(jobs)
        pipe = self.load_video()
        w, h, steps = self.proj["width"], self.proj["height"], self.proj["steps"]
        shots_file = self.p / "shots.json"
        failures = []
        for shot, refs in jobs:
            for t in range(takes):
                take_no = len(shot.get("takes", [])) + 1
                seed = shot["seed"] + 7919 * (take_no - 1)
                frames = snap_frames(shot["end"] - shot["start"])
                self.say(f"render {shot['id']} take {take_no} seed {seed} ({frames} frames, {steps} steps)")
                t0 = time.time()
                try:
                    rel = self._render_one(pipe, shot, refs, embeds, frames, w, h, steps, seed, take_no, media, crf, x264p)
                except Exception as e:  # keep the batch going; the shot simply stays without a take
                    import traceback
                    self.say(f"  !! {shot['id']} failed: {e}\n{traceback.format_exc()}")
                    failures.append(shot["id"])
                    torch.cuda.empty_cache()
                    break
                secs = round(time.time() - t0, 1)
                self.say(f"  -> {rel} in {secs}s")
                self._record(shot, take_no, rel, seed, secs)
        if failures:
            raise RuntimeError(f"Video generation failed for {', '.join(failures)}. Finished clips were kept; see the log.")
        self.say("done")

    def _render_one(self, pipe, shot, refs, embeds, frames, w, h, steps, seed, take_no, media, crf, x264p):
        import torch
        if True:
                with torch.no_grad():
                    emb = dict(embeds[shot["id"]])  # embeddings on the GPU; token tags stay on the CPU like the pipeline makes them
                    emb["prompt_embeds"] = emb["prompt_embeds"].to("cuda")
                    res = pipe(references=refs, **emb, num_frames=frames, height=h, width=w,
                               num_inference_steps=steps, generator=torch.Generator().manual_seed(seed),
                               output=["videos", "audio", "sampling_rate"])
                rel = f"renders/{shot['id']}_t{take_no}.mp4"
                media.write_video(res["videos"][0], self.p / rel, H3_FPS, crf, x264p,
                                  audio=res["audio"][0], sample_rate=res["sampling_rate"])
                return rel

    def _record(self, shot, take_no, rel, seed, secs):
        shots_file = self.p / "shots.json"
        if True:
                # re-read so concurrent prompt edits by a human/agent are not lost
                cur = json.loads(shots_file.read_text(encoding="utf-8"))
                for s in cur["shots"]:
                    if s["id"] == shot["id"]:
                        s.setdefault("takes", []).append({"take": take_no, "file": rel, "seed": seed, "seconds": secs,
                                                          "prompt": shot["prompt"]})
                        s["status"] = "rendered"
                        shot["takes"] = s["takes"]
                project.save_json(shots_file, cur)
