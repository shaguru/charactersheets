"""Picture in -> character sheet out. Backends: MiniMax-H3-Image (local, if downloaded) or CharacterForge."""
from __future__ import annotations

import shutil
from pathlib import Path

from mvm import forge, models, settings
from mvm import project as projects

# Layout prompts for H3-Image. <Picture 1> is the dropped-in image. Sizes ~4 MP (the model's minimum
# generation area) to keep a 24 GB card practical; MiniMax's own recommended sizes are 16 MP.
LAYOUTS = {
    "turnaround5": ("A professional character turnaround model sheet of the exact character in <Picture 1>. Keep the face, "
                    "hairstyle, body proportions, outfit, colours, accessories and art style identical to <Picture 1>. Five "
                    "full-body views side by side, evenly spaced, same scale, feet on one ground line, in this order from left "
                    "to right: front view, three-quarter front view, left profile, right profile, back view. Neutral standing "
                    "pose, plain light grey studio background, flat even lighting, no text, no labels, no borders.", (2720, 1536)),
    "turnaround3": ("A clean character model sheet of the exact character in <Picture 1>, identical face, outfit, colours and "
                    "style. Three full-body views in one row: front view, side profile, back view. Same scale, neutral pose, "
                    "plain light grey background, flat lighting, no text.", (2368, 1760)),
    "expressions": ("An expression sheet of the exact character in <Picture 1>, identical design and style. A neat 3 by 2 grid "
                    "of head-and-shoulders portraits facing the viewer at the same scale: happy, sad, angry, surprised, scared, "
                    "thoughtful. Plain light background, no text.", (2368, 1760)),
    "poses": ("A pose sheet of the exact character in <Picture 1>, identical design, outfit and style. Six full-body poses in one "
              "row: walking, running, jumping, waving, sitting, dancing. Same scale, plain light background, no text.", (2720, 1536)),
}


def make_sheet(image: str, project: str | None = None, backend: str = "auto", layout: str = "turnaround5",
               notes: str = "", count: int = 1) -> list[Path]:
    img = Path(image)
    if not img.is_file():
        raise SystemExit(f"No such image: {image}")
    out_dir = (projects.path(project) / "sheets") if project else settings.ROOT / "sheets"
    if backend == "auto":
        backend = "h3image" if models.status()["image"] else "forge"
    if backend == "h3image":
        from mvm import engine_image
        prompt, size = LAYOUTS.get(layout, LAYOUTS["turnaround5"])
        if notes:
            prompt += " " + notes
        files = engine_image.generate(prompt, [img], out_dir, size=size, count=count, stem=f"{img.stem}_{layout}_h3")
    else:
        files = forge.make_sheet(img, out_dir, layout=layout, notes=notes, count=count)
    if project and files:  # make the sheet available as a reference for shots
        dst = projects.path(project) / "assets/characters" / f"{img.stem}_sheet{files[0].suffix}"
        if not dst.exists():
            shutil.copy2(files[0], dst)
    return files
