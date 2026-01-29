
"""
Sudoku Grid Synthetic Dataset Generator (YOLO format)

Usage (from your repo root, recommended):
  python sudoku_synth_generator.py --out datasets/sudoku_grids --num 50 --val_ratio 0.2 --seed 42

Options:
  --out <path>          Root of the dataset (will create images/{train,val} and labels/{train,val})
  --num N               Total images to generate (train+val)
  --val_ratio R         Fraction (0..1) to put in val (default 0.2)
  --min_grids A         Min grids per image (default 1)
  --max_grids B         Max grids per image (default 12)
  --width W             Canvas width in pixels (randomized around this)
  --height H            Canvas height in pixels (randomized around this)
  --digits              Render random digits inside cells (printed-style)
  --candidates          Render "handwritten" candidate notes in random cells
  --headers             Add noisy headers/text blocks above/around grids
  --scribbles           Add random scribbles/circles/lines
  --low_contrast        Apply slight blur/low-contrast effect sometimes
  --seed S              Random seed

This generator is intentionally simple; refine as you wish.
"""
import argparse, os, random, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

def ensure_dirs(root):
    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        Path(root, p).mkdir(parents=True, exist_ok=True)

def yolo_line(xc, yc, w, h, W, H, cls=0):
    return f"{cls} {xc/W:.6f} {yc/H:.6f} {w/W:.6f} {h/H:.6f}\n"

def draw_sudoku_grid(draw, bbox, inner=9, heavy_every=3,
                     color=(0,0,0), outer_th=6, thin_th=2, heavy_th=4):
    x1,y1,x2,y2 = bbox
    draw.rectangle(bbox, outline=color, width=outer_th)
    w = x2 - x1; h = y2 - y1
    for i in range(1, inner):
        x = x1 + i * w / inner
        th = heavy_th if (i % heavy_every == 0) else thin_th
        draw.line([(x, y1), (x, y2)], fill=color, width=th)
        y = y1 + i * h / inner
        th = heavy_th if (i % heavy_every == 0) else thin_th
        draw.line([(x1, y), (x2, y)], fill=color, width=th)

def paste_digits(img, bbox, inner=9, prob_fill=0.4):
    # Printed-style digits using system default font
    draw = ImageDraw.Draw(img)
    x1,y1,x2,y2 = bbox
    w = (x2-x1)/inner
    h = (y2-y1)/inner
    for r in range(inner):
        for c in range(inner):
            if random.random() < prob_fill:
                d = str(random.randint(1,9))
                size = int(min(w,h)*0.6)
                try:
                    font = ImageFont.truetype("arial.ttf", size)
                except:
                    font = ImageFont.load_default()
                tx = x1 + c*w + (w-size)/2
                ty = y1 + r*h + (h-size)/2
                draw.text((tx,ty), d, fill=(0,0,0), font=font)

def draw_candidates(img, bbox, inner=9, prob_cell=0.5):
    # "Handwritten" tiny candidates (multiple digits) in cell corners
    draw = ImageDraw.Draw(img)
    x1,y1,x2,y2 = bbox
    w = (x2-x1)/inner
    h = (y2-y1)/inner
    for r in range(inner):
        for c in range(inner):
            if random.random() < prob_cell:
                # choose 2-4 candidate digits
                cand = "".join(str(random.randint(1,9)) for _ in range(random.randint(2,4)))
                size = int(min(w,h)*0.25)
                try:
                    font = ImageFont.truetype("arial.ttf", size)
                except:
                    font = ImageFont.load_default()
                # place in a random corner of the cell
                dx = random.choice([0.1, 0.6]) * w
                dy = random.choice([0.1, 0.6]) * h
                tx = x1 + c*w + dx
                ty = y1 + r*h + dy
                col = random.randint(0,30)
                draw.text((tx,ty), cand, fill=(col,col,col), font=font)

def add_scribbles(draw, W, H):
    for _ in range(random.randint(10, 30)):
        x1,y1 = random.randint(0,W-1), random.randint(0,H-1)
        x2,y2 = min(W-1,max(0,x1+random.randint(-200,200))), min(H-1,max(0,y1+random.randint(-200,200)))
        shade = random.randint(0,60)
        draw.line([(x1,y1),(x2,y2)], fill=(shade,shade,shade), width=random.randint(1,3))
    for _ in range(random.randint(3, 8)):
        x,y = random.randint(0,W-50), random.randint(0,H-50)
        r = random.randint(10, 40); shade = random.randint(0,80)
        draw.ellipse([x,y,x+2*r,y+2*r], outline=(shade,shade,shade), width=2)

def add_headers(draw, W, H):
    hdr_h = random.randint(40, 110)
    shade = random.randint(200,240)
    draw.rectangle([0,0,W,hdr_h], fill=(shade,shade,shade))
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
    draw.text((12,10), "PUZZLES  Issue %d"%(random.randint(1,60)), fill=(50,50,50), font=font)

def make_image(W, H, grids_px, opts):
    # base
    bg = (245,245,245)
    if opts["low_contrast"] and random.random()<0.5:
        bg = (230,230,230)
    img = Image.new("RGB", (W,H), bg)
    draw = ImageDraw.Draw(img)
    labels = []

    if opts["headers"] and random.random()<0.8:
        add_headers(draw, W, H)

    for bbox in grids_px:
        draw_sudoku_grid(draw, bbox)
        if opts["digits"] and random.random()<0.8:
            paste_digits(img, bbox)
        if opts["candidates"] and random.random()<0.8:
            draw_candidates(img, bbox)

        x1,y1,x2,y2 = bbox
        w = x2-x1; h=y2-y1; xc = x1 + w/2; yc = y1 + h/2
        labels.append((xc,yc,w,h))

    if opts["scribbles"] and random.random()<0.8:
        add_scribbles(draw, W, H)

    if opts["low_contrast"] and random.random()<0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    return img, labels

def layout_grids(W, H, n_grids):
    # choose between a few layout modes
    boxes = []
    if n_grids == 1:
        g = random.randint(int(min(W,H)*0.45), int(min(W,H)*0.72))
        x1 = (W - g)//2 + random.randint(-40,40)
        y1 = (H - g)//2 + random.randint(-40,40)
        boxes.append((x1,y1,x1+g,y1+g))
        return boxes

    # grid layouts (3x4, 3x2, etc.)
    cols = random.choice([2,3])
    rows = max(1, math.ceil(n_grids/cols))
    # compute sizes with gaps
    hgap = random.randint(40, 140)
    vgap = random.randint(50, 160)
    gW = (W - (cols-1)*hgap) // cols
    gH = (H - (rows-1)*vgap) // rows
    g = int(min(gW, gH) * random.uniform(0.80, 0.95))
    left = (W - (cols*g + (cols-1)*hgap))//2
    top  = (H - (rows*g + (rows-1)*vgap))//2
    for r in range(rows):
        for c in range(cols):
            if len(boxes) >= n_grids: break
            x = left + c*(g + hgap) + random.randint(-10,10)
            y = top  + r*(g + vgap) + random.randint(-10,10)
            boxes.append((x,y,x+g,y+g))
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Dataset root (will create images/ and labels/ subfolders)")
    ap.add_argument("--num", type=int, default=40, help="Total images to generate")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--min_grids", type=int, default=1)
    ap.add_argument("--max_grids", type=int, default=12)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--digits", action="store_true")
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--headers", action="store_true")
    ap.add_argument("--scribbles", action="store_true")
    ap.add_argument("--low_contrast", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.out)
    ensure_dirs(root)

    # Decide splits
    n_val = int(round(args.num * args.val_ratio))
    n_train = args.num - n_val
    plan = [("train", n_train), ("val", n_val)]

    opts = dict(
        digits=args.digits, candidates=args.candidates,
        headers=args.headers, scribbles=args.scribbles,
        low_contrast=args.low_contrast
    )

    counter = 0
    for split, n in plan:
        for i in range(n):
            # Slightly vary canvas size per image
            W = int(args.width  * random.uniform(0.9, 1.1))
            H = int(args.height * random.uniform(0.9, 1.1))

            n_grids = random.randint(args.min_grids, args.max_grids)
            boxes = layout_grids(W, H, n_grids)

            img, labels = make_image(W, H, boxes, opts)
            base = f"syn_{counter:04d}"
            img_path = root / f"images/{split}/{base}.jpg"
            lab_path = root / f"labels/{split}/{base}.txt"

            img.save(img_path, quality=92)
            with open(lab_path, "w") as f:
                for (xc,yc,w,h) in labels:
                    f.write(yolo_line(xc,yc,w,h,W,H))

            counter += 1

    print(f"[ok] Generated {counter} images at {root}")

if __name__ == "__main__":
    main()
