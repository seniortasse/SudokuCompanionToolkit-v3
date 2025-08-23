# train_cell_model.py
# Example training loop using tf.data and the multitask model.
# Usage:
#   python train_cell_model.py --root /path/to/synthetic_cells --epochs 3 --bs 128 --qat 0
import argparse, os
import tensorflow as tf
from sudoku_cell_model import build_multitask_model, compile_model, enable_qat
from tfdata_loader import make_dataset

def main(args):
    train_json = os.path.join(args.root, "train_manifest.jsonl")
    val_json   = os.path.join(args.root, "val_manifest.jsonl")
    model = build_multitask_model(input_shape=(64,64,1))
    if args.qat:
        model = enable_qat(model)
    compile_model(model)
    train_ds = make_dataset(train_json, batch_size=args.bs, shuffle=True, augment=True, repeat=True)
    val_ds   = make_dataset(val_json, batch_size=args.bs, shuffle=False, augment=False, repeat=False)
    steps_per_epoch = max(1, args.train_steps or (100))
    val_steps = max(1, args.val_steps or (20))
    model.fit(train_ds, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
              validation_data=val_ds, validation_steps=val_steps)
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "sudoku_cell_model.keras"))
    print("Saved model to", os.path.join(args.out, "sudoku_cell_model.keras"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to synthetic_cells root")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out", type=str, default="./checkpoints")
    ap.add_argument("--qat", type=int, default=0)
    ap.add_argument("--train_steps", type=int, default=0)
    ap.add_argument("--val_steps", type=int, default=0)
    args = ap.parse_args()
    main(args)
