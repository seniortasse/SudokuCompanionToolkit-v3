"""Create an MP4 animation from overlay frames using imageio-ffmpeg if available (or save raw frames and a how-to file as a fallback)."""


# animate_mp4.py
# Build an MP4 from per-move overlay images (and optional board frame).
# Requires: imageio (and FFmpeg available to imageio), or falls back to writing frames and printing an ffmpeg command.
#
# Usage:
#   python animate_mp4.py --dir demo_export --out demo_export/moves.mp4 \
#     --include_board 1 --size 900 --fps 1.0 --fade 1 --fade_frames 5
#
# To use the fallback:
#   ffmpeg -framerate 1 -i frames/frame_%05d.png -pix_fmt yuv420p demo_export/moves.mp4

import argparse, glob, sys
from pathlib import Path
from PIL import Image, ImageOps, ImageChops
import numpy as np

def gather_overlays(dir_path):
    return sorted(glob.glob(str(Path(dir_path)/"overlay_move_*.jpg")))

def load_img(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = ImageOps.fit(im, (size, size), method=Image.BICUBIC)
    return im

def frames_sequence(overlays, size=None, include_board=False, board_path=None, fade=False, fade_frames=4):
    frames = []
    # optional first frame
    if include_board and board_path and Path(board_path).exists():
        frames.append(load_img(board_path, size=size))
    prev = frames[-1] if frames else None
    for ov in overlays:
        img = load_img(ov, size=size)
        if fade and prev is not None:
            for k in range(1, fade_frames+1):
                alpha = k / (fade_frames+1)
                frames.append(Image.blend(prev, img, alpha))
        frames.append(img)
        prev = img
    return frames

def write_mp4(frames, out_path, fps=1.0):
    try:
        import imageio
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
        for f in frames:
            writer.append_data(np.asarray(f))
        writer.close()
        print(f"Wrote {out_path} ({len(frames)} frames)")
        return True
    except Exception as e:
        print("imageio/ffmpeg not available or failed:", e, file=sys.stderr)
        return False

def fallback_frames(frames, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(out_dir / f"frame_{i:05d}.png")
    cmd = f"ffmpeg -framerate 1 -i {out_dir}/frame_%05d.png -pix_fmt yuv420p output.mp4"
    (out_dir/"HOW_TO_FFMPEG.txt").write_text(cmd, encoding="utf-8")
    print(f"Saved {len(frames)} PNG frames to {out_dir}. To build MP4, run:\n{cmd}")

def main():
    """CLI entrypoint. Scans for overlay frames, writes an MP4 using FFmpeg via imageio; on failure, emits PNG frames and HOW_TO_FFMPEG.txt with an exact command to run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="demo_export")
    ap.add_argument("--out", type=str, default="demo_export/moves.mp4")
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--include_board", type=int, default=1)
    ap.add_argument("--fade", type=int, default=1)
    ap.add_argument("--fade_frames", type=int, default=5)
    args = ap.parse_args()

    overlays = gather_overlays(args.dir)
    if not overlays:
        print("No overlay_move_*.jpg files found.", file=sys.stderr)
        sys.exit(2)
    board = str(Path(args.dir)/"board_warped.png")
    frames = frames_sequence(overlays, size=args.size, include_board=bool(args.include_board), board_path=board, fade=bool(args.fade), fade_frames=args.fade_frames)
    ok = write_mp4(frames, args.out, fps=args.fps)
    if not ok:
        fallback_frames(frames, Path(args.out).with_suffix("").as_posix() + "_frames")

if __name__ == "__main__":
    main()