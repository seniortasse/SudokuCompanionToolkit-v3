# python/vision/data_tools/gen_synth_intersections.py
# Mixed synthetic generator for Sudoku intersection heatmap training.
# Levels:
#   D0 — clean 9×9 grid only (no digits/decoys/shadows/warps)
#   D3 — 9×9 with light digits in cells
#   D4 — 9×9 with header band OR header/footer text + optional page frame
#   D5 — D4 + shadows (half-plane and/or soft blobs)
#   D6 — D5 + geometric warps (rotation, keystone, cylindrical curvature)
#   D7 — D6-style grid + a NEIGHBOR GRID DECOY (top/bottom/left/right, 0.5–1.5× cell gap)
#   D8 — content sampled from earlier levels (D0/D3–D7) + stronger photographic layer
#
# All levels use 9×9 grids. Saves <img>.png + <img>.json with {"points":[[y,x],...]}

import argparse, json, math, random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


# ---------------------------
# Utils
# ---------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def draw_line(img: np.ndarray, p0, p1, color: int, t: int, broken: float = 0.0, rng: random.Random = None):
    """Draw straight (optionally broken) line."""
    if not rng or broken <= 0:
        cv2.line(img, tuple(map(int,p0)), tuple(map(int,p1)), color, int(max(1,t)), cv2.LINE_AA)
        return
    keep = 1.0 - broken
    x0,y0 = p0; x1,y1 = p1
    L = math.hypot(x1-x0, y1-y0)
    n = max(8, int(L/6))
    for k in range(n):
        t0 = k/n; t1 = (k+1)/n
        s0 = (1-t0)*x0 + t0*x1, (1-t0)*y0 + t0*y1
        s1 = (1-t1)*x0 + t1*x1, (1-t1)*y0 + t1*y1
        if rng.random() < keep:
            cv2.line(img, (int(s0[0]),int(s0[1])), (int(s1[0]),int(s1[1])), color, int(max(1,t)), cv2.LINE_AA)

def gaussian_shadow_halfplane(H,W,side,penumbra,gate_frac,a,theta_deg,rng:random.Random):
    xs = np.arange(W, dtype=np.float32)[None,:].repeat(H,0)
    ys = np.arange(H, dtype=np.float32)[:,None].repeat(W,1)
    theta = np.deg2rad(theta_deg)
    n = np.array([math.cos(theta), math.sin(theta)], np.float32)
    # boundary location
    if side=="right": c = rng.uniform(0.55*W,0.90*W)
    elif side=="left": c = rng.uniform(0.10*W,0.45*W)
    elif side=="bottom": c = rng.uniform(0.55*H,0.90*H)
    else: c = rng.uniform(0.10*H,0.45*H)
    d = (xs*n[0] + ys*n[1]) - c
    attn = 1.0/(1.0+np.exp(d/max(1.0,penumbra)))
    # gate to one side
    gate = np.ones_like(attn, np.float32)
    if side in ("left","right"):
        wgate = int(round(gate_frac*W))
        if side=="left":
            gate[:, :wgate] = 1.0
            gate[:, wgate:] = 0.0
        else:
            gate[:, :W-wgate] = 0.0
            gate[:, W-wgate:] = 1.0
    else:
        hgate = int(round(gate_frac*H))
        if side=="top":
            gate[:hgate, :] = 1.0
            gate[hgate:, :] = 0.0
        else:
            gate[:H-hgate, :] = 0.0
            gate[H-hgate:, :] = 1.0
    gate = cv2.GaussianBlur(gate, (0,0), rng.uniform(10,36))
    return np.clip(attn*gate,0,1)*a

def local_blob_mask(H,W,n_blobs,area_frac_range=(0.05,0.25),sigma_range=(8.0,40.0),rng:random.Random=None):
    mask = np.zeros((H,W), np.float32)
    total = H*W* rng.uniform(*area_frac_range)
    A = total/max(1,n_blobs)
    for _ in range(n_blobs):
        aspect = rng.uniform(0.5,2.0)
        ry = math.sqrt(A/(math.pi*max(1e-6,aspect)))
        rx = aspect*ry
        cx = rng.uniform(-0.1*W,1.1*W); cy = rng.uniform(-0.1*H,1.1*H)
        ang = rng.uniform(0,180)
        tmp = np.zeros((H,W), np.uint8)
        cv2.ellipse(tmp,(int(cx),int(cy)),(int(rx),int(ry)),ang,0,360,255,-1,cv2.LINE_AA)
        feather = cv2.GaussianBlur(tmp.astype(np.float32)/255.0,(0,0), rng.uniform(*sigma_range))
        mask += feather
    return np.clip(mask,0,1)

def perspective_warp(img: np.ndarray, pts_yx: np.ndarray, rng: random.Random, max_dx=0.15, max_dy=0.10):
    """Projective keystone. pts_yx shape [N,2] (y,x). Returns (img_warp, pts_warp)."""
    H,W = img.shape[:2]
    src = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    dst = src.copy()
    # choose horizontal or vertical keystone
    horiz = rng.random()<0.5
    dx = rng.uniform(0.0, max_dx)*W
    dy = rng.uniform(0.0, max_dy)*H
    if horiz:
        if rng.random()<0.5:
            dst[1,0]-=dx; dst[2,0]-=dx; dst[1,1]+=dy; dst[2,1]-=dy
        else:
            dst[0,0]+=dx; dst[3,0]+=dx; dst[0,1]+=dy; dst[3,1]-=dy
    else:
        if rng.random()<0.5:
            dst[0,1]+=dy; dst[1,1]+=dy; dst[0,0]+=dx; dst[1,0]-=dx
        else:
            dst[2,1]-=dy; dst[3,1]-=dy; dst[2,0]-=dx; dst[3,0]+=dx
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # transform points
    p = np.stack([pts_yx[:,1], pts_yx[:,0], np.ones(len(pts_yx),np.float32)], axis=1)  # (x,y,1)
    t = (M @ p.T).T
    t_xy = t[:, :2]/np.maximum(1e-6, t[:,2:3])
    pts_yx_w = np.stack([t_xy[:,1], t_xy[:,0]], axis=1)
    return out, pts_yx_w

def cylindrical_warp(img: np.ndarray, pts_yx: np.ndarray, rng: random.Random, axis="vertical", k=0.02):
    """Simple quadratic bow. axis='vertical' bows verticals; 'horizontal' bows horizontals."""
    H,W = img.shape[:2]
    xs = np.arange(W, dtype=np.float32)[None,:].repeat(H,0)
    ys = np.arange(H, dtype=np.float32)[:,None].repeat(W,1)
    cx,cy = (W-1)/2.0,(H-1)/2.0
    if axis=="vertical":
        y_norm = (ys-cy)/max(1.0,H)
        sgn = 1.0 if rng.random()<0.5 else -1.0
        map_x = xs + (k*sgn)*(y_norm**2)*W
        map_y = ys.astype(np.float32)
        out = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # pts
        y = pts_yx[:,0]; x = pts_yx[:,1]
        yn = (y - cy)/max(1.0,H)
        x_new = x + (k*sgn)*(yn**2)*W
        return out, np.stack([y, x_new], axis=1).astype(np.float32)
    else:
        x_norm = (xs-cx)/max(1.0,W)
        sgn = 1.0 if rng.random()<0.5 else -1.0
        map_x = xs.astype(np.float32)
        map_y = ys + (k*sgn)*(x_norm**2)*H
        out = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        y = pts_yx[:,0]; x = pts_yx[:,1]
        xn = (x - cx)/max(1.0,W)
        y_new = y + (k*sgn)*(xn**2)*H
        return out, np.stack([y_new, x], axis=1).astype(np.float32)

def draw_digits_9x9(img: np.ndarray, rect: Tuple[int,int,int,int], rng: random.Random):
    """Place light digits in each cell (some empty)."""
    x,y,w,h = rect
    cw, ch = w/9.0, h/9.0
    font = rng.choice([
        cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL
    ])
    thick = rng.choice([1,1,2])
    for r in range(9):
        for c in range(9):
            if rng.random() < 0.33:  # leave empty ~1/3
                continue
            d = str(rng.randint(1,9))
            target_h = rng.uniform(0.55,0.82)*ch
            # find scale
            (tw, th), base = cv2.getTextSize(d, font, 1.0, thick)
            s = max(0.25, min((target_h/max(1,th))*0.95, 3.0))
            (tw, th), base = cv2.getTextSize(d, font, s, thick)
            cx = x + int(round((c+0.5)*cw))
            cy = y + int(round((r+0.5)*ch))
            tx = int(round(cx - tw/2))
            ty = int(round(cy + (th-base)/2))
            col = rng.randint(30, 120)  # light gray-ish
            cv2.putText(img, d, (tx,ty), font, s, (col,), thick, cv2.LINE_AA)



def add_cell_center_decoy(img: np.ndarray,
                          pts_yx: np.ndarray,
                          rng: random.Random,
                          strength=(185, 255),
                          radius=(1, 2),
                          jitter_frac=0.12,
                          blur_sigma=(0.6, 1.2)):
    """
    Paint bright, slightly blurred dots at 9x9 cell centers as a decoy.
    - pts_yx : 100 true intersections [y,x] (10x10 grid).
    - Keeps GT unchanged; only modifies pixels in 'img'.
    """
    H, W = img.shape[:2]
    if pts_yx.size == 0:
        return

    # unique row/col coordinates of the 10x10 intersections
    ys = np.unique(np.round(pts_yx[:, 0]).astype(int))
    xs = np.unique(np.round(pts_yx[:, 1]).astype(int))
    if len(ys) < 2 or len(xs) < 2:
        return

    # cell centers = mid points between consecutive intersection rows/cols
    cy = (ys[:-1] + ys[1:]) / 2.0
    cx = (xs[:-1] + xs[1:]) / 2.0
    # cell pitch (used to scale jitter)
    pitch_y = float(np.median(np.diff(ys)))
    pitch_x = float(np.median(np.diff(xs)))
    jy = jitter_frac * pitch_y
    jx = jitter_frac * pitch_x

    # draw dots on an overlay -> blur -> take max with image
    overlay = np.zeros_like(img)
    for yy in cy:
        for xx in cx:
            y = int(round(yy + rng.uniform(-jy, jy)))
            x = int(round(xx + rng.uniform(-jx, jx)))
            r  = rng.randint(radius[0], radius[1])
            v  = int(rng.randint(strength[0], strength[1]))
            cv2.circle(overlay, (x, y), r, v, -1, lineType=cv2.LINE_AA)

    # blur the overlay so these become “heat-like” spots
    if isinstance(blur_sigma, tuple):
        sig = rng.uniform(*blur_sigma)
    else:
        sig = float(blur_sigma)
    k = int(max(3, 2 * int(3 * sig) + 1))  # odd kernel, >=3

    overlay = cv2.GaussianBlur(overlay, (k, k), sig)
    #overlay = cv2.GaussianBlur(overlay, (k, k), sigX=sig, sigmaY=sig)

    # brighten the image at these locations
    np.maximum(img, overlay, out=img)




def header_band(img: np.ndarray, rect: Tuple[int,int,int,int], rng: random.Random):
    """Dark header band above grid + centered label."""
    H,W = img.shape[:2]
    x,y,w,h = rect
    ch = h/9.0
    band_h = int(np.clip(rng.uniform(0.8,1.8)*ch, 6, H))
    by0 = max(0, y-band_h); by1 = y
    bx0, bx1 = x, x+w
    col = 0 if rng.random()<0.75 else rng.randint(20,80)
    cv2.rectangle(img,(bx0,by0),(bx1,by1), int(col), -1)
    label = rng.choice(["SUDOKU","STEP 5", f"Puzzle #{rng.randint(1000,9999)}"])
    fg = 255
    scale = np.clip((band_h*0.85)/18.0, 0.35, 2.8)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    tx = int(np.clip(x + (w-tw)//2, 0, max(0,W-tw)))
    ty = by0 + int((band_h + th)//2)
    cv2.putText(img,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX, scale, (fg,), 1, cv2.LINE_AA)

def header_footer_text(img: np.ndarray, rect: Tuple[int,int,int,int], rng: random.Random):
    """Thin text lines above/below grid; visual clutter only."""
    H,W = img.shape[:2]
    x,y,w,h = rect
    ch = h/9.0
    def make_block(where: str, n_lines: int):
        pad = rng.randint(2,5)
        scale = np.clip((rng.uniform(0.6,1.2)*0.65*ch)/18.0, 0.30, 2.5)
        thick = 1
        def x_anchor(tw):
            mode = rng.choices(["left","center","right"], weights=[0.4,0.2,0.4])[0]
            if mode=="left":  return x + rng.randint(0, 6)
            if mode=="right": return x + w - tw - rng.randint(0, 6)
            return x + (w - tw)//2
        if where=="header":
            y_base = y - pad
            for _ in range(n_lines):
                s = rng.choice(["Daily Sudoku Challenge","Practice makes progress","Number puzzle"])
                (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                tx = int(np.clip(x_anchor(tw), 0, max(0,W-tw)))
                ty = max(th, y_base)
                col = rng.randint(80,180)
                cv2.putText(img, s, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (col,), thick, cv2.LINE_AA)
                y_base -= (th + rng.randint(2,5))
        else:
            y_base = y + h + pad
            for _ in range(n_lines):
                s = rng.choice(["Logic over luck","Sharpen your skills","Mindful break"])
                (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                tx = int(np.clip(x_anchor(tw), 0, max(0,W-tw)))
                ty = min(H-1-base, y_base)
                col = rng.randint(80,180)
                cv2.putText(img, s, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (col,), thick, cv2.LINE_AA)
                y_base += (th + rng.randint(2,6))
    r = rng.random()
    if r < 0.33:   make_block("header", rng.randint(1,2))
    elif r < 0.66: make_block("footer", rng.randint(1,3))
    else:
        make_block("header", rng.randint(1,2))
        make_block("footer", rng.randint(1,3))

def page_frame_decoy(img: np.ndarray, rng: random.Random):
    H,W = img.shape[:2]
    pad = rng.randint(1,3); col = rng.randint(180,210); th = rng.randint(1,2)
    cv2.rectangle(img,(pad,pad),(W-1-pad,H-1-pad), int(col), th)
    for _ in range(rng.randint(2,5)):  # tiny gaps
        if rng.random()<0.5:
            x0 = rng.randint(pad, W-1-pad); x1 = min(W-1-pad, x0 + rng.randint(5,20))
            y = rng.choice([pad, H-1-pad])
            cv2.line(img,(x0,y),(x1,y), int(col+rng.randint(-10,10)), th)
        else:
            y0 = rng.randint(pad, H-1-pad); y1 = min(H-1-pad, y0 + rng.randint(5,20))
            x = rng.choice([pad, W-1-pad])
            cv2.line(img,(x,y0),(x,y1), int(col+rng.randint(-10,10)), th)

def photographic_layer(img: np.ndarray, rng: random.Random, strength=0.5):
    g = img.astype(np.float32)
    # brightness/contrast
    alpha = 1.0 + rng.uniform(-0.1, 0.25)*strength
    beta  = rng.uniform(-8, 16)*strength
    g = np.clip(alpha*g + beta, 0, 255)
    # blur chance
    if rng.random()<0.6*strength:
        k = rng.choice([3,5,7])
        g = cv2.GaussianBlur(g, (k,k), rng.uniform(0.2,1.2))
    # noise
    if rng.random()<0.75*strength:
        noise = np.random.randn(*g.shape).astype(np.float32)*rng.uniform(0.5,8.0)*strength
        g = np.clip(g + noise, 0, 255)
    # light vignette (scale with strength)
    if rng.random()<0.55*strength:
        H,W = g.shape[:2]
        cx, cy = (W-1)/2, (H-1)/2
        xs = np.linspace(0, W-1, W, dtype=np.float32)
        ys = np.linspace(0, H-1, H, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        r = np.sqrt(((X-cx)/(W/2))**2 + ((Y-cy)/(H/2))**2)
        v = np.clip(0.08*strength, 0.05, 0.22)
        g *= (1.0 - v*np.clip(r,0,1))
    return g.astype(np.uint8)


def _mask_from_points(H: int, W: int, pts_yx: np.ndarray, sigma: float) -> np.ndarray:
    """Sum-of-Gaussians mask around [y,x] points, clipped to [0,1]."""
    if pts_yx.size == 0:
        return np.zeros((H, W), np.float32)
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    m = np.zeros((H, W), np.float32)
    s2 = 2.0 * (sigma ** 2)
    for (y0, x0) in pts_yx:
        m += np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / max(1e-6, s2)).astype(np.float32)
    return np.clip(m, 0, 1)

def _apply_local_corner_blur(img: np.ndarray, pts_yx: np.ndarray,
                             mask_sigma: float, blur_sigma: float, amount: float) -> np.ndarray:
    """
    Blur more where the intersections are.
    amount in [0..1]: 0=no change, 1=full blend with blurred image at corners.
    """
    if pts_yx.size == 0 or amount <= 0:
        return img
    H, W = img.shape[:2]
    blurred = cv2.GaussianBlur(img, (0, 0), max(0.1, float(blur_sigma)))
    mask = _mask_from_points(H, W, pts_yx, float(mask_sigma))
    mask = (mask * float(amount)).astype(np.float32)  # 0..~1
    out = (img.astype(np.float32) * (1.0 - mask) + blurred.astype(np.float32) * mask)
    return np.clip(out, 0, 255).astype(np.uint8)

def _draw_digits_dark_9x9(img: np.ndarray, rect: Tuple[int,int,int,int], rng: random.Random):
    """Like draw_digits_9x9, but digits are darker/clearer to dominate faint lines."""
    x, y, w, h = rect
    cw, ch = w / 9.0, h / 9.0
    font = rng.choice([
        cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL
    ])
    thick = rng.choice([1, 2, 2])
    for r in range(9):
        for c in range(9):
            if rng.random() < 0.25:
                continue
            d = str(rng.randint(1, 9))
            target_h = rng.uniform(0.60, 0.85) * ch
            (tw, th), _ = cv2.getTextSize(d, font, 1.0, thick)
            s = max(0.25, min((target_h / max(1, th)) * 0.95, 3.2))
            (tw, th), base = cv2.getTextSize(d, font, s, thick)
            cx = x + int(round((c + 0.5) * cw))
            cy = y + int(round((r + 0.5) * ch))
            tx = int(round(cx - tw / 2))
            ty = int(round(cy + (th - base) / 2))
            col = rng.randint(10, 70)  # darker than lines
            cv2.putText(img, d, (tx, ty), font, s, (int(col),), thick, cv2.LINE_AA)

# ---------------------------
# Grid rendering & GT points
# ---------------------------

def draw_grid(img: np.ndarray, rect: Tuple[int,int,int,int], rows: int, cols: int,
              col_border=0, col_div=50, col_std=120,
              t_border=2, t_div=2, t_std=1, broken_border=0.0, broken_std=0.0, rng: random.Random=None):
    """Rect grid with darker border, mid dividers, light std lines."""
    x,y,w,h = rect
    # border
    draw_line(img,(x,y),(x+w,y), col_border, t_border, broken_border, rng)
    draw_line(img,(x+w,y),(x+w,y+h), col_border, t_border, broken_border, rng)
    draw_line(img,(x+w,y+h),(x,y+h), col_border, t_border, broken_border, rng)
    draw_line(img,(x,y+h),(x,y), col_border, t_border, broken_border, rng)
    # internals
    cw, ch = w/cols, h/rows
    V_divs = {3,6} if cols==9 else set()
    H_divs = {3,6} if rows==9 else set()
    for r in range(1, rows):
        yy = int(round(y + r*ch))
        is_div = r in H_divs
        draw_line(img,(x,yy),(x+w,yy), col_div if is_div else col_std,
                  t_div if is_div else t_std, broken_std, rng)
    for c in range(1, cols):
        xx = int(round(x + c*cw))
        is_div = c in V_divs
        draw_line(img,(xx,y),(xx,y+h), col_div if is_div else col_std,
                  t_div if is_div else t_std, broken_std, rng)

def intersections_from_rect(rect: Tuple[int,int,int,int], rows:int, cols:int) -> np.ndarray:
    """Return (rows+1)*(cols+1) intersections as [y,x] float32 (axis-aligned grid)."""
    x,y,w,h = rect
    xs = np.linspace(x, x+w, cols+1, dtype=np.float32)
    ys = np.linspace(y, y+h, rows+1, dtype=np.float32)
    pts = np.stack(np.meshgrid(ys, xs, indexing="ij"), axis=-1).reshape(-1,2)  # [N,2] [y,x]
    return pts

def clip_points_to_image(pts_yx: np.ndarray, H:int, W:int) -> np.ndarray:
    pts = pts_yx.copy()
    pts[:,0] = np.clip(pts[:,0], 0, H-1)
    pts[:,1] = np.clip(pts[:,1], 0, W-1)
    return pts


# ---------------------------
# Neighbor grid (D7)
# ---------------------------

def add_neighbor_grid(img: np.ndarray,
                      rng: random.Random,
                      main_rect: Tuple[int,int,int,int],
                      rows: int, cols: int,
                      *,
                      col_border: int, col_div: int, col_std: int,
                      t_border: int, t_div: int, t_std: int,
                      broken_border: float, broken_std: float,
                      with_digits_prob: float = 0.35):
    """Draw a second grid adjacent to the main one at 0.5–1.5× cell gap."""
    x,y,w,h = main_rect
    cw = w / max(1, cols)
    ch = h / max(1, rows)
    side = rng.choice(["top","bottom","left","right"])
    gap = rng.uniform(0.5, 1.5) * (cw if side in ("left","right") else ch)

    if side == "top":
        dec_rect = (x, int(round(y - gap - h)), w, h)
    elif side == "bottom":
        dec_rect = (x, int(round(y + h + gap)), w, h)
    elif side == "left":
        dec_rect = (int(round(x - gap - w)), y, w, h)
    else:  # right
        dec_rect = (int(round(x + w + gap)), y, w, h)

    draw_grid(img, dec_rect, rows, cols,
              col_border, col_div, col_std,
              t_border, t_div, t_std,
              broken_border, broken_std, rng)

    if rng.random() < with_digits_prob:
        draw_digits_9x9(img, dec_rect, rng)


# ---------------------------
# Sample builder per level
# ---------------------------

def _style_for_level(level: str, rng: random.Random):
    """Produce colors/thickness and breakage consistent per level."""
    # dark border / lighter inners by default
    col_border = rng.randint(0, 25)
    col_div    = min(200, col_border + rng.randint(20,55))
    col_std    = min(235, col_div + rng.randint(15,55))

    # weak-outer reversal: teach not to snap to page edge (D4+)
    if level in ("D4","D5","D6","D7") and rng.random()<0.5:
        col_border = rng.randint(110, 190)
        col_div    = rng.randint(0, 55)
        col_std    = rng.randint(0, 70)

    t_border = rng.choice([1,2,2,3])
    t_div    = max(2, int(round(t_border*0.8)))
    t_std    = rng.choice([1,1,1,2])

    broken_border = 0.0
    broken_std = 0.0
    if level in ("D6","D7"):  # allow brokenness from D6 onward
        broken_border = rng.uniform(0.05, 0.25)
        broken_std    = rng.uniform(0.00, 0.12)

    return (col_border, col_div, col_std, t_border, t_div, t_std, broken_border, broken_std)


def sample_once(rng: random.Random, img_size:int, level:str):
    # D8: pick a base level first, then apply stronger photo layer
    if level == "D8":
        base = rng.choice(["D0","D3","D4","D5","D6","D7"])
        img, pts = sample_once(rng, img_size, base)
        img = photographic_layer(img, rng, strength=0.95)
        return img, pts
    

    if level == "D9":
        H = W = img_size
        # paper base
        base_val = rng.randint(220, 255)
        img = np.ones((H, W), np.uint8) * base_val

        # faint main grid footprint
        side = rng.uniform(0.65, 0.88)
        w = h = int(round(side * img_size))
        margin = max(2, int(0.5 * (img_size - w)))
        x = rng.randint(margin - 2, margin + 2)
        y = rng.randint(margin - 2, margin + 2)
        rect = (x, y, w, h)
        rows = cols = 9

        # faint, thin lines for main grid (digits will be darker)
        col_border = rng.randint(160, 220)       # light
        col_div    = min(235, col_border + rng.randint(12, 40))
        col_std    = min(245, col_div + rng.randint(10, 35))
        t_border   = 1
        t_div      = 1
        t_std      = 1
        broken_border = 0.0
        broken_std    = rng.uniform(0.00, 0.05)  # maybe slightly broken

        draw_grid(img, rect, rows, cols,
                  col_border, col_div, col_std,
                  t_border, t_div, t_std,
                  broken_border, broken_std, rng)

        # strong digits on the main grid (so digits overshadow faint junctions)
        _draw_digits_dark_9x9(img, rect, rng)

        # main GT intersections (axis-aligned here)
        pts = intersections_from_rect(rect, rows, cols)

        # add a NEIGHBOR-GRID DECOY that is darker/clearer than the main grid
        add_neighbor_grid(
            img, rng, rect, rows, cols,
            col_border=rng.randint(0, 25),              # dark
            col_div=rng.randint(25, 70),
            col_std=rng.randint(60, 120),
            t_border=2, t_div=2, t_std=2,               # thicker than main
            broken_border=rng.uniform(0.00, 0.12),
            broken_std=rng.uniform(0.00, 0.10),
            with_digits_prob=0.50
        )

        # OPTIONAL: tiny global rotation (kept mild)
        if rng.random() < 0.5:
            ang = rng.uniform(-3.0, 3.0)
            M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ones = np.ones((len(pts), 1), np.float32)
            xy = np.stack([pts[:, 1], pts[:, 0]], axis=1)
            t = (M @ np.concatenate([xy, ones], axis=1).T).T
            pts = np.stack([t[:, 1], t[:, 0]], axis=1).astype(np.float32)

        # KEY: micro “corner blur” to make junctions faint/fuzzy vs digit centers
        img = _apply_local_corner_blur(
            img, pts,
            mask_sigma=rng.uniform(0.8, 1.8),   # how wide the corner mask is
            blur_sigma=rng.uniform(0.7, 1.6),   # actual blur amount
            amount=rng.uniform(0.6, 1.0)        # blend strength
        )

        # light photographic layer to keep variety
        if rng.random() < 0.5:
            img = photographic_layer(img, rng, strength=0.65)

        pts = clip_points_to_image(pts.astype(np.float32), H, W)
        return img, pts



    H=W=img_size
    # paper base
    base_val = rng.randint(215, 255)
    img = np.ones((H,W), np.uint8)*base_val

    # grid footprint (square) and placement
    side = rng.uniform(0.65, 0.9) if level!="D0" else rng.uniform(0.55, 0.92)
    w = h = int(round(side*img_size))
    margin = max(2, int(0.5*(img_size - w)))
    x = rng.randint(margin-2, margin+2)
    y = rng.randint(margin-2, margin+2)
    rect = (x,y,w,h)

    # 9×9 ONLY
    rows = cols = 9

    # style
    (col_border, col_div, col_std,
     t_border, t_div, t_std,
     broken_border, broken_std) = _style_for_level(level, rng)

    # main grid
    draw_grid(img, rect, rows, cols,
              col_border, col_div, col_std,
              t_border, t_div, t_std,
              broken_border, broken_std, rng)

    # D3 digits
    if level == "D3":
        draw_digits_9x9(img, rect, rng)

    # D4 decorations
    if level == "D4":
        if rng.random()<0.5:
            header_band(img, rect, rng)
        else:
            header_footer_text(img, rect, rng)
        if rng.random()<0.75:
            page_frame_decoy(img, rng)

    # D5 shadows
    if level == "D5":
        # half-plane
        if rng.random()<0.7:
            side = rng.choice(["left","right","top","bottom"])
            pen = rng.uniform(24, 120)
            gate = rng.uniform(0.35, 0.95)
            attn = rng.uniform(0.15, 0.60)
            m = gaussian_shadow_halfplane(H,W,side,pen,gate,attn,rng.uniform(-10,10), rng)
            img = np.clip(img.astype(np.float32)*(1.0 - m), 0, 255).astype(np.uint8)
        # blobs (maybe)
        if rng.random()<0.6:
            mask = local_blob_mask(H,W, rng.randint(1,3), (0.05,0.25), (8.0, 60.0), rng)
            a = rng.uniform(0.2, 0.75)
            img = np.clip(img.astype(np.float32)*(1.0 - a*mask), 0, 255).astype(np.uint8)

    # compute GT intersections (axis-aligned for now)
    pts = intersections_from_rect(rect, rows, cols)




    # --- D10: digit-dominant, ultrathin lines + warped page, LR shading, gutter, extra corner blur ---
    if level == "D10":
        H = W = img_size
        img = np.ones((H, W), np.uint8) * rng.randint(220, 255)

        # grid footprint
        side = rng.uniform(0.70, 0.88)
        w = h = int(round(side * img_size))
        margin = max(2, int(0.5 * (img_size - w)))
        x = rng.randint(margin - 2, margin + 2)
        y = rng.randint(margin - 2, margin + 2)
        rect = (x, y, w, h)
        rows = cols = 9

        # ultrathin, faint lines (thinner than default)
        col_border = rng.randint(160, 210)
        col_div    = min(230, col_border + rng.randint(8, 30))
        col_std    = min(240, col_div + rng.randint(8, 25))
        t_border   = 1
        t_div      = 1
        t_std      = 1
        draw_grid(img, rect, rows, cols,
                col_border, col_div, col_std,
                t_border, t_div, t_std,
                broken_border=rng.uniform(0.00, 0.06),
                broken_std=rng.uniform(0.00, 0.06),
                rng=rng)

        # darker, clearer digits to dominate faint lines
        _draw_digits_dark_9x9(img, rect, rng)

        # GT intersections
        pts = intersections_from_rect(rect, rows, cols)

        # anisotropic / wider local blur at junctions (extend the range a bit)
        # build a wider mask around corners then blend with a stronger blur
        def _corner_blur_aniso(im, pts_yx):
            H, W = im.shape
            base = _mask_from_points(H, W, pts_yx, sigma=rng.uniform(1.2, 2.0))
            m_x  = cv2.GaussianBlur(base, (0,0), sigmaX=rng.uniform(1.8, 2.6), sigmaY=rng.uniform(0.4, 0.8))
            m_y  = cv2.GaussianBlur(base, (0,0), sigmaX=rng.uniform(0.4, 0.8), sigmaY=rng.uniform(1.8, 2.6))
            mask = np.maximum(m_x, m_y)
            blur = cv2.GaussianBlur(im, (0,0), sigmaX=rng.uniform(0.9, 1.8), sigmaY=rng.uniform(0.9, 1.8))
            a = rng.uniform(0.4, 0.9)
            out = im.astype(np.float32) * (1.0 - a*mask) + blur.astype(np.float32) * (a*mask)
            return np.clip(out, 0, 255).astype(np.uint8)
        img = _corner_blur_aniso(img, pts)

        # mild warps (curved/keystone) so the whole page looks photographed
        if rng.random() < 0.5:
            img, pts = perspective_warp(img, pts, rng, max_dx=0.10, max_dy=0.08)
        if rng.random() < 0.5:
            img, pts = cylindrical_warp(img, pts, rng, axis=rng.choice(["vertical","horizontal"]), k=rng.uniform(0.01, 0.03))

        # soft left→right (or right→left) shading gradient
        if rng.random() < 0.9:
            H, W = img.shape
            xs = np.linspace(0, 1, W, dtype=np.float32)
            if rng.random() < 0.5: grad = xs
            else:                   grad = 1.0 - xs
            strength = rng.uniform(0.06, 0.18)
            shade = (1.0 - strength * grad)[None, :]
            img = np.clip(img.astype(np.float32) * shade, 0, 255).astype(np.uint8)

        # “book gutter” dark strip near one vertical border
        if rng.random() < 0.8:
            H, W = img.shape
            side = rng.choice(["left","right"])
            w_frac = rng.uniform(0.04, 0.10)
            gutter_w = max(3, int(round(w_frac * W)))
            mask = np.zeros((H, W), np.float32)
            if side == "left":
                mask[:, :gutter_w] = 1.0
            else:
                mask[:, W-gutter_w:] = 1.0
            mask = cv2.GaussianBlur(mask, (0,0), sigmaX=rng.uniform(6, 18), sigmaY=rng.uniform(6, 18))
            dark = rng.uniform(0.08, 0.25)
            img = np.clip(img.astype(np.float32) * (1.0 - dark * mask), 0, 255).astype(np.uint8)

        # optional camera layer
        if rng.random() < 0.5:
            img = photographic_layer(img, rng, strength=0.6)

        pts = clip_points_to_image(pts.astype(np.float32), H, W)
        return img, pts






    # D6 geometric warps (apply to image AND points)
    if level == "D6":
        # mild rotation first
        if rng.random()<0.5:
            ang = rng.uniform(-5,5)
            M = cv2.getRotationMatrix2D(((W-1)/2,(H-1)/2), ang, 1.0)
            img = cv2.warpAffine(img, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ones = np.ones((len(pts),1), np.float32)
            xy = np.stack([pts[:,1], pts[:,0]], axis=1)
            t = (M @ np.concatenate([xy, ones], axis=1).T).T
            pts = np.stack([t[:,1], t[:,0]], axis=1).astype(np.float32)

        # keystone (projective)
        if rng.random()<0.7:
            img, pts = perspective_warp(img, pts, rng, max_dx=0.15, max_dy=0.10)

        # cylindrical (book bow) i.e., curvature
        if rng.random()<0.5:
            axis = rng.choice(["vertical","horizontal"])
            img, pts = cylindrical_warp(img, pts, rng, axis=axis, k=rng.uniform(0.01,0.035))

    # D7 neighbor grid decoy (+ a bit of extra shadow & optional rotation)
    if level == "D7":
        # draw the decoy with the SAME style as main grid
        add_neighbor_grid(
            img, rng, rect, rows, cols,
            col_border=col_border, col_div=col_div, col_std=col_std,
            t_border=t_border, t_div=t_div, t_std=t_std,
            broken_border=broken_border, broken_std=broken_std,
            with_digits_prob=0.40
        )

        # slightly higher shadow chance
        if rng.random()<0.55:
            side = rng.choice(["left","right","top","bottom"])
            pen = rng.uniform(24, 120)
            gate = rng.uniform(0.35, 0.95)
            attn = rng.uniform(0.15, 0.60)
            m = gaussian_shadow_halfplane(H,W,side,pen,gate,attn,rng.uniform(-10,10), rng)
            img = np.clip(img.astype(np.float32)*(1.0 - m), 0, 255).astype(np.uint8)
        if rng.random()<0.55:
            mask = local_blob_mask(H,W, rng.randint(1,3), (0.05,0.25), (8.0, 60.0), rng)
            a = rng.uniform(0.2, 0.75)
            img = np.clip(img.astype(np.float32)*(1.0 - a*mask), 0, 255).astype(np.uint8)

        # mild rotation “still possible”
        if rng.random()<0.5:
            ang = rng.uniform(-5,5)
            M = cv2.getRotationMatrix2D(((W-1)/2,(H-1)/2), ang, 1.0)
            img = cv2.warpAffine(img, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ones = np.ones((len(pts),1), np.float32)
            xy = np.stack([pts[:,1], pts[:,0]], axis=1)
            t = (M @ np.concatenate([xy, ones], axis=1).T).T
            pts = np.stack([t[:,1], t[:,0]], axis=1).astype(np.float32)

    # mild “camera” look for D4–D7
    if level in ("D4","D5","D6","D7") and rng.random()<0.35:
        img = photographic_layer(img, rng, strength=0.5)

    # final clamp & return
    pts = clip_points_to_image(pts.astype(np.float32), H, W)
    return img, pts


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--n", type=int, default=60000, help="Number of images to generate")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--p",
        nargs="*",
        default=["D0=0.10","D3=0.15","D4=0.15","D5=0.15","D6=0.15","D7=0.15","D8=0.15"],
        help="Mixture like: D0=0.10 D3=0.15 D4=0.15 D5=0.15 D6=0.15 D7=0.15 D8=0.15 (must sum≈1)"
    )
    ap.add_argument("--show-every", type=int, default=0, help="Preview every K images (OpenCV window)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    ensure_dir(out)

    # parse mixture
    weights = {}
    for tok in args.p:
        k,v = tok.split("=")
        weights[k.strip()] = float(v)
    levels = list(weights.keys())
    probs  = np.array([weights[k] for k in levels], np.float32)
    probs  = probs / probs.sum()

    print(f"[gen] out={out} | n={args.n} | img={args.img_size} | mix=" +
          ", ".join([f"{k}:{weights[k]:.2f}" for k in levels]))

    for i in range(args.n):
        level = rng.choices(levels, weights=list(probs), k=1)[0]
        img, pts_yx = sample_once(rng, args.img_size, level)

        stem = f"syn_{i:06d}"
        cv2.imwrite(str(out / f"{stem}.png"), img)
        data = {"points": [[float(y), float(x)] for (y,x) in pts_yx.astype(np.float32)]}
        (out / f"{stem}.json").write_text(json.dumps(data))

        if (i+1) % 5000 == 0 or i == 0:
            print(f"  - generated {i+1}/{args.n} (last={level})")

        if args.show_every and ((i % args.show_every) == 0):
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for y,x in pts_yx:
                cv2.circle(vis, (int(round(x)),int(round(y))), 2, (0,0,255), -1, cv2.LINE_AA)
            cv2.imshow("preview", vis)
            key = cv2.waitKey(1)
            if key == 27:  # Esc
                break

    print("Done.")

if __name__ == "__main__":
    main()