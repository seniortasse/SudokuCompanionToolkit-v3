# exports YOLO .pt -> TFLite int8 into mobile/assets
# (we'll fill this later with Ultralytics export code)
# tools/export_tflite.py
"""
Exports a trained YOLOv8 model to INT8 TFLite and drops it into:
python/mobile/android/app/src/main/assets/best-int8.tflite

Usage (from repo root):
  # 1) (Optional) Train:
  #    yolo detect train model=yolov8n.pt data=yolo/sudoku_grid.yaml imgsz=640 epochs=80 batch=16
  # 2) Export:
  #    python tools/export_tflite.py --pt yolo/runs/detect/train/weights/best.pt
"""
import argparse, os
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True, help="Path to trained .pt checkpoint")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--int8", action="store_true", help="Force int8; default on")
    #p.add_argument("--data", type=str, default="yolo/sudoku_grid.yaml")
    p.add_argument("--data", type=str, default="yolo/sudoku_grid.yaml",
                help="Dataset YAML for INT8 calibration")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit("Install Ultralytics first: pip install ultralytics\n" + str(e))

    model = YOLO(args.pt)
    print(f"[export] exporting {args.pt} -> TFLite int8 ...")
    tflite_path = model.export(format="tflite", imgsz=args.imgsz, int8=True,
                           data=args.data)

    # Move to android assets
    tflite_path = Path(tflite_path)
    dest = Path("python/mobile/android/app/src/main/assets") / "best-int8.tflite"
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tflite_path, dest)
    print(f"[export] wrote: {dest}")

    # Label map (single class)
    (dest.parent / "labelmap.txt").write_text("grid\n", encoding="utf-8")
    print(f"[export] wrote: {dest.parent / 'labelmap.txt'}")

if __name__ == "__main__":
    main()