"""Generate HoMM-style pixel-art unit portrait icons via Gemini (Nano Banana).

Usage:  GEMINI_API_KEY=... uv run --with google-genai --with pillow \
            python scripts/gen_unit_icons.py [unit ...]

Each icon is requested as a bust/portrait crop (not a full figure) on a flat
magenta background, which is then chroma-keyed to transparency, auto-cropped,
padded square and downscaled with nearest-neighbour to a crisp 64x64 PNG in
src/tactica/web/static/units/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

OUT_DIR = Path(__file__).parent.parent / "src" / "tactica" / "web" / "static" / "units"
RAW_DIR = Path(__file__).parent / "raw_icons"

STYLE = (
    "Retro 16-bit fantasy pixel art unit icon for a 1990s turn-based strategy "
    "game, in the spirit of Heroes of Might and Magic. Chunky visible square "
    "pixels, limited warm palette, strong 1px dark outline around the subject, "
    "dramatic side lighting. IMPORTANT: this is a PORTRAIT CROP, not a full "
    "figure — only the recognisable upper part of the unit fills the frame. "
    "The entire background must be one flat uniform pure magenta color "
    "rgb(255,0,255) with no gradient, no shadow cast on it, no vignette. "
    "Subject centered, filling about 80%% of the frame. No text, no letters, "
    "no border, no frame, no watermark."
)

UNITS = {
    "pikeman": (
        "Subject: a medieval pikeman — head and shoulders of a stern soldier "
        "in a simple kettle helmet and padded gambeson, both hands gripping "
        "the wooden shaft of his upright pike; only the upper part of the "
        "pike shaft and its steel spearhead are visible."),
    "archer": (
        "Subject: a fantasy archer — head and chest of a hooded marksman in "
        "leather armor, drawing a shortbow with a nocked arrow aimed slightly "
        "off-frame; bow limbs cropped by the frame edges."),
    "griffin": (
        "Subject: a griffin — fierce eagle head and feathered neck with "
        "piercing golden eyes, sharp curved beak open in a screech, hint of "
        "folded wing feathers at the bottom edge of the crop."),
    "swordsman": (
        "Subject: a swordsman — bust of a knight in polished steel plate "
        "armor and closed visor helmet, the crossguard and lower blade of a "
        "longsword held vertically beside his head."),
    "cavalry": (
        "Subject: heavy cavalry — armored knight from the waist up with a "
        "couched lance, riding; only the front part of his armored warhorse "
        "(head, neck and chest in barding) is visible below him, the rest "
        "cropped out."),
}

MAGENTA = (255, 0, 255)
KEY_DIST2 = 110 ** 2  # generous: pixel-art edges should still key cleanly


def postprocess(raw: Image.Image, out_path: Path, size: int = 64) -> None:
    img = raw.convert("RGBA")
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b, _ = px[x, y]
            d2 = (r - MAGENTA[0]) ** 2 + (g - MAGENTA[1]) ** 2 + (b - MAGENTA[2]) ** 2
            if d2 < KEY_DIST2:
                px[x, y] = (0, 0, 0, 0)
            elif r > g and b > g and min(r, b) - g > 50:
                # magenta spill on outline pixels: the unit palette is warm
                # browns/golds, so anything this magenta-tinted is fringe
                px[x, y] = (0, 0, 0, 0)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    # pad to square with a small margin, then crisp downscale
    side = int(max(img.size) * 1.06)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    canvas = canvas.resize((size, size), Image.NEAREST)
    canvas.save(out_path, format="PNG")


def main() -> int:
    wanted = sys.argv[1:] or list(UNITS)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        prompt = f"{STYLE}\n\n{UNITS[name]}"
        print(f"[{name}] generating ...", flush=True)
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1",
                                               image_size="1K"),
            ),
        )
        saved = False
        for part in response.parts:
            if part.inline_data:
                import io
                raw = Image.open(io.BytesIO(part.inline_data.data))
                raw.save(RAW_DIR / f"{name}.jpg")
                postprocess(raw, OUT_DIR / f"{name}.png")
                print(f"[{name}] saved -> {OUT_DIR / (name + '.png')}")
                saved = True
                break
        if not saved:
            text = " ".join(p.text or "" for p in response.parts)
            print(f"[{name}] NO IMAGE returned: {text[:300]}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
