"""Hardware-based render sizes; CPU offload still requires a CUDA GPU."""
import math

LOW_VRAM_PIXELS = 768 * 512
MAX_PIXELS = 2_097_152


def profile_for_vram(vram_bytes, mode="auto"):
    mode = mode.strip().lower()
    if mode not in {"auto", "1", "0"}:
        raise ValueError("FORGE_LOW_VRAM must be auto, 1 (on), or 0 (off).")
    low = mode == "1" or (mode == "auto" and 0 < vram_bytes <= 12 * 1024**3)
    return {
        "low_vram": low,
        "vram_gb": round(vram_bytes / 1024**3, 1),
        "max_pixels": LOW_VRAM_PIXELS if low else MAX_PIXELS,
        "label": "Low VRAM: CPU offload" if low else "Standard",
    }


def limit_dimensions(width, height, max_pixels):
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / (width * height))
    w = max(256, int(width * scale / 16) * 16)
    h = max(256, int(height * scale / 16) * 16)
    # The minimum dimension can push very wide/tall images above the budget.
    while w * h > max_pixels:
        if w >= h and w > 256:
            w -= 16
        else:
            h -= 16
    return w, h


def presets_for_profile(presets, profile):
    result = {**presets, "sizes": {}}
    for name, size in presets["sizes"].items():
        w, h = size["w"], size["h"]
        if w and h:
            w, h = limit_dimensions(w, h, profile["max_pixels"])
        result["sizes"][name] = {**size, "w": w, "h": h}
    return result
