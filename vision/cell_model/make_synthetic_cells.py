
import os, json, random, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

# Types: 0=blank, 1=printed, 2=handwritten, 3=notes
TYPE_BLANK, TYPE_PRINTED, TYPE_HAND, TYPE_NOTES = 0, 1, 2, 3

def find_font():
    # Try a few common fonts; fall back to default
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, 48)
            except:
                pass
    return ImageFont.load_default()

def jitter(val, amt):
    return val + random.randint(-amt, amt)

def add_noise(img, sigma=6):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def adjust_contrast(img, factor=None):
    if factor is None:
        factor = random.uniform(0.8, 1.3)
    return ImageOps.autocontrast(ImageEnhance.Contrast(img).enhance(factor))

def draw_grid_ghost(draw, W, H):
    # faint grid lines as background noise
    color = 220
    for i in range(1,9):
        lw = 1 if i % 3 else 2
        draw.line((i*W/9, 0, i*W/9, H), fill=color, width=lw)
        draw.line((0, i*H/9, W, i*H/9), fill=color, width=lw)

def render_printed_cell(size=64, digit=5, font=None):
    if font is None:
        font = find_font()
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    # optional faint grid ghost
    if random.random() < 0.3:
        draw_grid_ghost(d, size, size)
    s = random.randint(34, 48)
    f = ImageFont.truetype(font.path, s) if hasattr(font, "path") else font
    text = str(digit)
    w, h = d.textsize(text, font=f)
    x = (size - w)//2 + random.randint(-2,2)
    y = (size - h)//2 + random.randint(-2,2)
    # Printed digits usually thin, dark gray
    d.text((x, y), text, fill=random.randint(20, 40), font=f)
    if random.random()<0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.6)))
    return img

def render_handwritten_cell(size=64, digit=5, font=None):
    # Simulate handwriting: slightly thicker, darker, rotated
    if font is None:
        font = find_font()
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    s = random.randint(34, 50)
    f = ImageFont.truetype(font.path, s) if hasattr(font, "path") else font
    text = str(digit)
    w, h = d.textsize(text, font=f)
    x = (size - w)//2 + random.randint(-3,3)
    y = (size - h)//2 + random.randint(-3,3)
    # draw with stroke to mimic pen thickness
    d.text((x, y), text, fill=random.randint(0, 30), font=f, stroke_width=random.randint(1,2), stroke_fill=0)
    # random rotation
    img = img.rotate(random.uniform(-10,10), resample=Image.BICUBIC, fillcolor=255)
    if random.random()<0.5:
        img = add_noise(img, sigma=random.uniform(2,6))
    return img

def render_notes_cell(size=64, digits=[1,4,8], font=None):
    # Render tiny candidates in a 3x3 mini-grid
    if font is None:
        font = find_font()
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    # Decide mini positions for 1..9
    # Positions laid out 3x3
    positions = []
    margin = 6
    step = (size - 2*margin) // 3
    for rr in range(3):
        for cc in range(3):
            cx = margin + cc*step + step//2
            cy = margin + rr*step + step//2
            positions.append((cx,cy))
    s = random.randint(10, 16)
    f = ImageFont.truetype(font.path, s) if hasattr(font, "path") else font
    for dgt in digits:
        cx, cy = positions[dgt-1]
        tx = cx + random.randint(-2,2)
        ty = cy + random.randint(-2,2)
        d.text((tx, ty), str(dgt), fill=random.randint(0, 50), font=f, anchor="mm")
    if random.random()<0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.1,0.5)))
    return img

def make_one_cell(out_dir, idx, typ=None):
    if typ is None:
        typ = random.choices([TYPE_BLANK, TYPE_PRINTED, TYPE_HAND, TYPE_NOTES], weights=[0.4,0.25,0.2,0.15])[0]

    size = 64
    digit = 0
    notes_vec = [0]*9
    font = find_font()

    if typ == TYPE_BLANK:
        img = Image.new("L", (size, size), 255)
        if random.random()<0.2:
            img = add_noise(img, sigma=random.uniform(1,4))

    elif typ == TYPE_PRINTED:
        digit = random.randint(1,9)
        img = render_printed_cell(size=size, digit=digit, font=font)

    elif typ == TYPE_HAND:
        digit = random.randint(1,9)
        img = render_handwritten_cell(size=size, digit=digit, font=font)

    elif typ == TYPE_NOTES:
        # choose 2-5 digits as candidates
        k = random.randint(2,5)
        ds = sorted(random.sample(range(1,10), k=k))
        for dgt in ds:
            notes_vec[dgt-1] = 1
        img = render_notes_cell(size=size, digits=ds, font=font)

    # finalize: slight brightness/contrast jitter and small crop jitter
    if random.random()<0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 0.4)))
    # pad/crop jitter
    pad = random.randint(0,2)
    if pad>0:
        img = ImageOps.expand(img, border=pad, fill=255).resize((size,size), resample=Image.BICUBIC)

    # Save
    out_path = Path(out_dir) / f"img_{idx:05d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    # Labels
    type_onehot = [0,0,0,0]; type_onehot[typ]=1
    digit_onehot = [0]*10; digit_onehot[digit]=1  # 0=blank
    rec = {
        "path": str(out_path),
        "type": typ,
        "type_onehot": type_onehot,
        "digit": digit,
        "digit_onehot": digit_onehot,
        "notes": notes_vec
    }
    return rec

def generate_dataset(root, n_train=360, n_val=120, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    root = Path(root)
    (root/"images"/"train").mkdir(parents=True, exist_ok=True)
    (root/"images"/"val").mkdir(parents=True, exist_ok=True)

    train_manifest = []
    val_manifest = []

    for i in range(1, n_train+1):
        rec = make_one_cell(root/"images"/"train", i)
        train_manifest.append(rec)

    for i in range(1, n_val+1):
        rec = make_one_cell(root/"images"/"val", i)
        val_manifest.append(rec)

    (root/"train_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in train_manifest), encoding="utf-8")
    (root/"val_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in val_manifest), encoding="utf-8")
    print(f"Done. Train: {len(train_manifest)}  Val: {len(val_manifest)}")
    print(f"Root: {root}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="synthetic_cells")
    ap.add_argument("--train", type=int, default=360)
    ap.add_argument("--val", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate_dataset(args.out, n_train=args.train, n_val=args.val, seed=args.seed)
