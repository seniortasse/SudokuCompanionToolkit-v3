import cv2, numpy as np, sys
from pathlib import Path

def load_gray(p):
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(p)
    return im.astype(np.float32)/255.0

desk_dir = Path(sys.argv[1])   # e.g. .../desktop_tiles64
phone_dir = Path(sys.argv[2])  # e.g. .../phone_session/tiles64

rows = []
for p in sorted(phone_dir.glob("r*c*.png")):
    q = desk_dir/p.name
    if not q.exists(): 
        continue
    a, b = load_gray(p), load_gray(q)
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a, b = a[:h,:w], b[:h,:w]
    mae = float(np.mean(np.abs(a-b)))
    mse = float(np.mean((a-b)**2))
    rows.append((p.name, mae, mse))

rows.sort(key=lambda t: t[1], reverse=True)
print(f"Compared {len(rows)} tiles")
for n, mae, mse in rows[:10]:
    print(f"{n}: MAE={mae:.4f}  MSE={mse:.4f}")