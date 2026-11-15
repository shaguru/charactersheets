"""Video writing with controllable compression (diffusers' encode_video uses libx264 defaults, CRF 23)."""
from __future__ import annotations

import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np

# quality preset -> (render CRF, final-cut CRF, x264 preset). Lower CRF = less compression, bigger files.
QUALITY = {
    "max": (6, 10, "slow"),      # visually lossless, large files
    "high": (10, 14, "slow"),    # default
    "medium": (16, 18, "medium"),
    "draft": (22, 23, "veryfast"),
}


def preset(settings: dict, proj: dict | None = None) -> tuple[int, int, str]:
    q = (proj or {}).get("quality") or settings.get("quality", "high")
    return QUALITY.get(q, QUALITY["high"])


def _frames_to_uint8(frames) -> list[np.ndarray]:
    out = []
    for f in frames:
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
        out.append(a)
    return out


def write_video(frames, path: Path, fps: int, crf: int, x264_preset: str = "slow", audio=None,
                sample_rate: int | None = None) -> Path:
    """frames: list of PIL images or HxWx3 arrays. audio: (channels, samples) tensor/array or None."""
    import soundfile as sf
    arr = _frames_to_uint8(frames)
    h, w = arr[0].shape[:2]
    executable = shutil.which("ffmpeg")
    if not executable:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [executable, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-"]
    tmp_wav = None
    if audio is not None:
        a = audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio, dtype=np.float32)
        fd, name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # Windows: an open handle would block ffmpeg reading it and us deleting it
        tmp_wav = Path(name)
        sf.write(tmp_wav, a.T if a.ndim == 2 else a, sample_rate, subtype="PCM_16")
        cmd += ["-i", str(tmp_wav)]
    cmd += ["-map", "0:v"] + (["-map", "1:a", "-c:a", "aac", "-b:a", "256k", "-shortest"] if tmp_wav else [])
    cmd += ["-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for a in arr:
        proc.stdin.write(a.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed writing {path}")
    if tmp_wav:
        try:
            tmp_wav.unlink(missing_ok=True)
        except OSError:
            pass
    return path
