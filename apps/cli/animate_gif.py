"""Create an animated GIF from the sequence of overlay_move_XX.jpg frames."""

# animate_gif.py
# Build an animated GIF from per-move overlay images.
# Usage:
#   python animate_gif.py --dir demo_export --out demo_export/moves.gif \
#     --include_board 1 --size 900 --step_ms 900 --start_ms 1000 --end_ms 1500 \
#     --fade 1 --fade_frames 5 --fade_ms 40
#
# The script looks for files named overlay_move_*.jpg in --dir.
# Optionally includes board_warped.png as the first frame.

import argparse
import glob
from pathlib import Path

from PIL import Image, ImageOps


def gather_overlays(dir_path):
    return sorted(glob.glob(str(Path(dir_path) / "overlay_move_*.jpg")))


def load_img(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = ImageOps.fit(im, (size, size), method=Image.BICUBIC)
    return im


def animate(
    overlays,
    out_path,
    size=None,
    include_board=False,
    board_path=None,
    step_ms=900,
    start_ms=800,
    end_ms=1200,
    fade=False,
    fade_frames=4,
    fade_ms=30,
):
    frames = []
    durations = []

    # Optional first frame (board only)
    if include_board:
        if board_path and Path(board_path).exists():
            base = load_img(board_path, size=size)
        else:
            # fallback to first overlay without title band by lightly dimming
            base = (
                load_img(overlays[0], size=size)
                if overlays
                else Image.new("RGB", (size or 900, size or 900), "white")
            )
        frames.append(base)
        durations.append(start_ms)

    prev = frames[-1] if frames else None
    for ov in overlays:
        img = load_img(ov, size=size)
        if fade and prev is not None:
            # Crossfade from prev -> img
            for k in range(1, fade_frames + 1):
                alpha = k / (fade_frames + 1)
                mix = Image.blend(prev, img, alpha=alpha)
                frames.append(mix)
                durations.append(fade_ms)
        frames.append(img)
        durations.append(step_ms)
        prev = img

    if frames:
        durations[-1] = max(durations[-1], end_ms)  # ensure last frame holds

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Wrote {out_path} with {len(frames)} frames.")


def main():
    """CLI entrypoint. Scans a directory for overlay_move_*.jpg frames, encodes them into an animated GIF at a modest framerate, and saves to --out."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="demo_export")
    ap.add_argument("--out", type=str, default="demo_export/moves.gif")
    ap.add_argument("--size", type=int, default=900, help="final square size in px")
    ap.add_argument("--include_board", type=int, default=1)
    ap.add_argument("--step_ms", type=int, default=900)
    ap.add_argument("--start_ms", type=int, default=800)
    ap.add_argument("--end_ms", type=int, default=1500)
    ap.add_argument("--fade", type=int, default=1)
    ap.add_argument("--fade_frames", type=int, default=5)
    ap.add_argument("--fade_ms", type=int, default=40)
    args = ap.parse_args()

    overlays = gather_overlays(args.dir)
    board = Path(args.dir) / "board_warped.png"
    animate(
        overlays,
        args.out,
        size=args.size,
        include_board=bool(args.include_board),
        board_path=str(board),
        step_ms=args.step_ms,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        fade=bool(args.fade),
        fade_frames=args.fade_frames,
        fade_ms=args.fade_ms,
    )


if __name__ == "__main__":
    main()
