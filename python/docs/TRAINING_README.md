
# Training quickstart

1) Generate synthetic data (basic or enhanced):
```bash
# Basic
python make_synthetic_cells.py --out synthetic_cells --train 5000 --val 1000
# Enhanced
python make_synthetic_cells_plus.py --out synthetic_cells_plus --train 5000 --val 1000
```

2) Train the multi-task model:
```bash
python train_cell_model.py --root synthetic_cells --epochs 3 --bs 256 --train_steps 200 --val_steps 40
# With QAT (quantization-aware training)
python train_cell_model.py --root synthetic_cells --epochs 3 --bs 256 --train_steps 200 --val_steps 40 --qat 1
```

3) Convert to TFLite/CoreML (example TFLite):
```python
import tensorflow as tf
m = tf.keras.models.load_model("checkpoints/sudoku_cell_model.keras", compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(m)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("checkpoints/sudoku_cell_model_int8.tflite", "wb").write(tflite_model)
```

---

## Conversion & Mobile Inference

### TFLite (full INT8)
```bash
python convert_to_tflite.py --keras checkpoints/sudoku_cell_model.keras   --manifests synthetic_cells/train_manifest.jsonl synthetic_cells/val_manifest.jsonl   --out checkpoints/sudoku_cell_model_int8.tflite --max_samples 2000
```

**Android (Kotlin)**: see `android_tflite_stub.kt` for a minimal interpreter wrapper that expects **uint8** input `[1,64,64,1]`.

**iOS (Swift with TensorFlowLiteSwift)**: see `ios_tflite_stub.swift`.

### CoreML
```bash
python convert_to_coreml.py --keras checkpoints/sudoku_cell_model.keras   --out checkpoints/SudokuCell.mlmodel
```
**iOS (CoreML)**: see `ios_coreml_stub.swift`.

---

## New solver features
`solver_core.py` now includes **Locked Candidates (Pointing & Claiming)** eliminations.  
`sudo ku_tools.py` exposes these via `next_moves(..., max_difficulty="locked")` and `apply_action(...)` which can apply placements **or** eliminations.
