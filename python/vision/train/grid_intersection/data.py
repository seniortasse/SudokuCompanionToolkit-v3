from dataclasses import dataclass
from typing import Tuple, List, Any, Dict, Optional
import os, json

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

# ---------------------------- Config ----------------------------

@dataclass
class DataCfg:
    train_manifest: str
    val_manifest: str
    image_size: int = 768
    grayscale: bool = True

# ------------------------- IO Utilities -------------------------

def _read_jsonl(p: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            recs.append(json.loads(ln))
    return recs

def _preflight_manifest(path: str, max_report: int = 10) -> None:
    """Quickly warn if any files referenced in the manifest are missing."""
    bad = []
    with open(path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if not ln.strip():
                continue
            rec = json.loads(ln)
            ip = rec.get("image_path")
            lp = rec.get("label_path")
            ok = True
            if not (ip and os.path.exists(ip)): ok = False
            if not (lp and os.path.exists(lp)): ok = False
            if not ok:
                bad.append((i, ip, lp))
                if len(bad) >= max_report:
                    break
    if bad:
        print(f"[data] WARNING: {len(bad)} missing/corrupt references in {path} (showing up to {max_report}):")
        for i, ip, lp in bad[:max_report]:
            print(f"  line#{i}: img_exists={os.path.exists(ip) if ip else False}  lbl_exists={os.path.exists(lp) if lp else False}")
        print("  (loader will drop broken samples at runtime)")

# ---------------------- Tensor Conversions ----------------------

def _imread_any(path: str, grayscale: bool) -> Optional[np.ndarray]:
    """Windows-safe image loader using np.fromfile -> cv2.imdecode."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imdecode(buf, flag)
        return img
    except Exception:
        return None

def _resize_mask(m: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)

def _resize_float(m: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)

def _to_torch_img(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # (1,H,W)
    else:
        # convert HWC BGR to CHW RGB and scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return t

# ---------------------------- Dataset ---------------------------

class GridDataset(Dataset):
    def __init__(self, manifest_path: str, cfg: DataCfg, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train
        self.recs = _read_jsonl(manifest_path)

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, idx: int):
        rec = self.recs[idx]
        ip: str = rec["image_path"]
        lp: str = rec["label_path"]
        size = self.cfg.image_size

        try:
            # --- image ---
            img = _imread_any(ip, grayscale=self.cfg.grayscale)
            if img is None:
                raise ValueError(f"cv2.imdecode failed for {ip}")
            if img.shape[0] != size or img.shape[1] != size:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            img_t = _to_torch_img(img)  # (1,H,W) or (3,H,W)

            # --- labels (A,H,V,J, O=(Ox,Oy)) ---
            with np.load(lp) as z:
                A = z["A"]; H = z["H"]; V = z["V"]; J = z["J"]; O = z["O"]
            # resize label maps to target size
            A = _resize_mask(A, size)
            H = _resize_mask(H, size)
            V = _resize_mask(V, size)
            J = _resize_float(J, size)  # J is a soft gaussian map -> keep float
            if O.ndim == 3 and O.shape[2] == 2:
                Ox = _resize_float(O[..., 0], size)
                Oy = _resize_float(O[..., 1], size)
            else:
                raise ValueError(f"Orientation field O has wrong shape: {O.shape}")

            # Normalize masks to [0,1] float; re-normalize orientation to unit vectors
            A = (A.astype(np.float32) / 255.0)
            H = (H.astype(np.float32) / 255.0)
            V = (V.astype(np.float32) / 255.0)
            # J can already be 0..255; put to [0,1]
            J = (J.astype(np.float32) / 255.0)
            mag = np.sqrt(Ox * Ox + Oy * Oy) + 1e-6
            Ox = (Ox / mag).astype(np.float32)
            Oy = (Oy / mag).astype(np.float32)

            y = np.stack([A, H, V, J, Ox, Oy], axis=0)  # (6,H,W)
            y_t = torch.from_numpy(y.astype(np.float32))

            rec_out = {
                "image_path": ip,
                "label_path": lp,
                "idx": int(idx),
            }
            return img_t, y_t, rec_out

        except Exception as e:
            # Compact log; training will drop this sample via collate.
            print(f"[dataset] DROP idx={idx} ip={ip} lp={lp} error={type(e).__name__}: {e}")
            return None

# ----------------------- Robust Collation -----------------------

def collate_drop_none(batch):
    """Drop None samples and collate (img, y, rec)."""
    keep = []
    for item in batch:
        if item is None:
            continue
        img, y, rec = item
        if img is None or y is None:
            continue
        if isinstance(rec, dict):
            # scrub None values to avoid default_collate errors
            rec = {k: (v if v is not None else "") for k, v in rec.items()}
        keep.append((img, y, rec))

    if not keep:
        raise RuntimeError("All samples in batch were None/broken. Check dataset/labels.")

    imgs = default_collate([k[0] for k in keep])
    ys   = default_collate([k[1] for k in keep])
    recs = [k[2] for k in keep]
    return imgs, ys, recs

# -------------------------- Dataloaders -------------------------

def make_loaders(cfg: DataCfg, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    # Preflight warnings
    _preflight_manifest(cfg.train_manifest)
    _preflight_manifest(cfg.val_manifest)

    train_ds = GridDataset(cfg.train_manifest, cfg, is_train=True)
    val_ds   = GridDataset(cfg.val_manifest,   cfg, is_train=False)

    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_drop_none,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_drop_none,
    )
    return train_ld, val_ld