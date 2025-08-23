# make_synthetic_cells_plus.py
# Extended synthetic generator with perspective warp, paper texture, and richer "handwriting".

import os, json, random, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageChops, ImageEnhance
import numpy as np

TYPE_BLANK, TYPE_PRINTED, TYPE_HAND, TYPE_NOTES = 0, 1, 2, 3
IMG_SIZE = 64

def find_fonts():
    # Try a small pack of fonts if present; fall back to DejaVu
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    fallbacks = [f for f in candidates if Path(f).exists()]
    return fallbacks if fallbacks else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

FONT_PATHS = find_fonts()

def rand_font(size):
    fp = random.choice(FONT_PATHS)
    try:
        return ImageFont.truetype(fp, size)
    except:
        return ImageFont.load_default()

def add_paper_texture(img, strength=0.08):
    w,h = img.size
    noise = np.random.normal(0, 255*strength, (h,w)).astype(np.float32)
    base = np.array(img).astype(np.float32)
    out = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def perspective_warp(img, max_jitter=4):
    w,h = img.size
    def jitter_pt(x,y):
        return (x + random.randint(-max_jitter,max_jitter),
                y + random.randint(-max_jitter,max_jitter))
    src = [(0,0),(w,0),(w,h),(0,h)]
    dst = [jitter_pt(0,0), jitter_pt(w,0), jitter_pt(w,h), jitter_pt(0,h)]
    return img.transform((w,h), Image.PERSPECTIVE, 
                         Image.Transform.quad_mesh(src, dst), 
                         resample=Image.Resampling.BICUBIC)

def draw_grid_ghost(draw, W, H):
    color = 220
    for i in range(1,9):
        lw = 1 if i % 3 else 2
        draw.line((i*W/9, 0, i*W/9, H), fill=color, width=lw)
        draw.line((0, i*H/9, W, i*H/9), fill=color, width=lw)

def render_printed_cell(digit):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    d = ImageDraw.Draw(img)
    if random.random()<0.25:
        draw_grid_ghost(d, IMG_SIZE, IMG_SIZE)
    s = random.randint(34, 48)
    f = rand_font(s)
    txt = str(digit)
    w,h = d.textbbox((0,0), txt, font=f)[2:]
    x = (IMG_SIZE - w)//2 + random.randint(-2,2)
    y = (IMG_SIZE - h)//2 + random.randint(-2,2)
    d.text((x,y), txt, fill=random.randint(20,40), font=f)
    if random.random()<0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2,0.6)))
    return img

def render_handwritten_cell(digit):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    d = ImageDraw.Draw(img)
    s = random.randint(34, 52)
    f = rand_font(s)
    txt = str(digit)
    w,h = d.textbbox((0,0), txt, font=f)[2:]
    x = (IMG_SIZE - w)//2 + random.randint(-4,4)
    y = (IMG_SIZE - h)//2 + random.randint(-4,4)
    # thicker stroke to mimic pen
    d.text((x,y), txt, fill=random.randint(0,25), font=f, stroke_width=random.randint(1,2), stroke_fill=0)
    img = img.rotate(random.uniform(-12,12), resample=Image.Resampling.BICUBIC, fillcolor=255)
    if random.random()<0.5:
        img = add_paper_texture(img, strength=random.uniform(0.04,0.10))
    if random.random()<0.3:
        img = perspective_warp(img, max_jitter=3)
    return img

def render_notes_cell(digits):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
    d = ImageDraw.Draw(img)
    margin = 6
    step = (IMG_SIZE - 2*margin)//3
    s = random.randint(10, 16)
    f = rand_font(s)
    for dgt in digits:
        rr = (dgt-1)//3; cc=(dgt-1)%3
        cx = margin + cc*step + step//2 + random.randint(-2,2)
        cy = margin + rr*step + step//2 + random.randint(-2,2)
        d.text((cx,cy), str(dgt), fill=random.randint(0,50), font=f, anchor="mm")
    if random.random()<0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.1,0.5)))
    if random.random()<0.3:
        img = perspective_warp(img, max_jitter=3)
    return img

def make_one_cell(idx, typ=None):
    if typ is None:
        typ = random.choices([TYPE_BLANK, TYPE_PRINTED, TYPE_HAND, TYPE_NOTES], weights=[0.4,0.25,0.2,0.15])[0]
    digit = 0; notes_vec = [0]*9
    if typ==TYPE_BLANK:
        img = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
        if random.random()<0.25:
            img = add_paper_texture(img, strength=random.uniform(0.02,0.08))
    elif typ==TYPE_PRINTED:
        digit = random.randint(1,9)
        img = render_printed_cell(digit)
    elif typ==TYPE_HAND:
        digit = random.randint(1,9)
        img = render_handwritten_cell(digit)
    else: # notes
        k = random.randint(2,5)
        ds = sorted(random.sample(range(1,10), k=k))
        for dgt in ds: notes_vec[dgt-1]=1
        img = render_notes_cell(ds)

    # Final jitter
    if random.random()<0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 0.4)))
    pad = random.randint(0,2)
    if pad>0:
        img = ImageOps.expand(img, border=pad, fill=255).resize((IMG_SIZE,IMG_SIZE), resample=Image.Resampling.BICUBIC)

    type_onehot = [0,0,0,0]; type_onehot[typ]=1
    digit_onehot = [0]*10; digit_onehot[digit]=1

    return img, {
        "type": typ,
        "type_onehot": type_onehot,
        "digit": digit,
        "digit_onehot": digit_onehot,
        "notes": notes_vec
    }

def generate(out_root, n_train=1000, n_val=200, seed=123):
    random.seed(seed); np.random.seed(seed)
    out_root = Path(out_root)
    (out_root/"images"/"train").mkdir(parents=True, exist_ok=True)
    (out_root/"images"/"val").mkdir(parents=True, exist_ok=True)

    def dump_split(split, N):
        manifest = []
        for i in range(1, N+1):
            img, rec = make_one_cell(i)
            p = out_root/"images"/split/f"img_{i:05d}.png"
            img.save(p)
            rec["path"] = str(p)
            manifest.append(rec)
        (out_root/f"{split}_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in manifest), encoding="utf-8")

    dump_split("train", n_train)
    dump_split("val", n_val)
    print("Wrote", out_root)

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="synthetic_cells_plus")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    generate(args.out, args.train, args.val, args.seed)
