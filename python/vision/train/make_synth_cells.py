# vision/train/make_synth_cells.py

# Synthetic cell generator
# Renders digits 1–9 (and empty “0”) on small gray tiles.
# Adds simple noise/affine jitter; optional font variety.
# Produces train/val splits with folder names 0..9.

from __future__ import annotations
import random, math, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Tuple

OUT = Path("vision/data/synth")
FONTS_DIR = Path("vision/cells/assets/fonts")  # drop any .ttf here
IMG_SIZE = 28  # 28x28 grayscale like MNIST
TRAIN_PER_CLASS = 3000
VAL_PER_CLASS = 300

def find_fonts() -> List[Path]:
    fonts = list(FONTS_DIR.glob("*.ttf"))
    if not fonts:
        # fallback: PIL default
        return []
    return fonts

def rand_font(size: int, fonts: List[Path]) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if fonts:
        return ImageFont.truetype(str(random.choice(fonts)), size=size)
    return ImageFont.load_default()

def render_digit(d: int, fonts: List[Path]) -> Image.Image:
    bg = Image.new("L", (IMG_SIZE, IMG_SIZE), color=230)  # light gray background
    draw = ImageDraw.Draw(bg)

    # size & placement jitter
    scale = random.uniform(0.65, 0.9)
    fsize = max(10, int(IMG_SIZE * scale))
    font = rand_font(fsize, fonts)

    txt = "" if d == 0 else str(d)
    w, h = draw.textbbox((0, 0), txt, font=font)[2:]
    x = (IMG_SIZE - w) // 2 + random.randint(-1, 1)
    y = (IMG_SIZE - h) // 2 + random.randint(-1, 1)
    if d != 0:
        draw.text((x, y), txt, fill=random.randint(0, 40), font=font)  # dark ink

    # light blur / noise / affine
    if random.random() < 0.4:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.6)))

    # tiny rotation
    angle = random.uniform(-3.0, 3.0)
    bg = bg.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=230)

    return bg

def write_split(split: str, per_class: int, fonts: List[Path]):
    out = OUT / split
    for c in range(10):
        ddir = out / str(c)
        ddir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            im = render_digit(c, fonts)
            im.save(ddir / f"{c}_{i:05d}.png", "PNG")

def main():
    fonts = find_fonts()
    print(f"Found {len(fonts)} fonts. (Optional)")
    write_split("train", TRAIN_PER_CLASS, fonts)
    write_split("val", VAL_PER_CLASS, fonts)
    print(f"Wrote synthetic dataset to {OUT}")

if __name__ == "__main__":
    main()