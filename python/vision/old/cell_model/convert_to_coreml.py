# convert_to_coreml.py
# Convert Keras model to CoreML (fp16). Requires coremltools.
# Usage:
#   python convert_to_coreml.py --keras checkpoints/sudoku_cell_model.keras --out checkpoints/SudokuCell.mlmodel
import argparse

import coremltools as ct
import tensorflow as tf


def main(args):
    model = tf.keras.models.load_model(args.keras, compile=False)
    # Keras -> CoreML
    mlmodel = ct.convert(
        model, inputs=[ct.ImageType(shape=(1, 64, 64, 1), scale=1 / 255.0, color_layout="G")]
    )
    mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=16)
    mlmodel.save(args.out)
    print("Saved", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)
