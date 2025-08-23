# sudoku_cell_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
try:
    import tensorflow_model_optimization as tfmot
except Exception:
    tfmot = None

NUM_TYPES = 4            # blank, printed, handwritten, notes
NUM_DIGITS = 10          # 0..9
NUM_NOTES = 9            # presence of 1..9

def build_backbone(input_shape=(64,64,1)):
    base = keras.applications.MobileNetV3Small(
        input_shape=(64,64,3),
        include_top=False,
        alpha=0.5,
        minimalistic=True,
        pooling='avg'
    )
    inp = keras.Input(shape=input_shape)
    x = layers.Concatenate()([inp, inp, inp])  # tile grayscale
    x = base(x, training=False)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    return inp, x

def build_multitask_model(input_shape=(64,64,1)):
    inp, feat = build_backbone(input_shape)
    # H1
    h1 = layers.Dense(128, activation="relu")(feat)
    out_type = layers.Dense(NUM_TYPES, activation="softmax", name="type")(h1)
    # H2
    h2 = layers.Dense(128, activation="relu")(feat)
    out_digit = layers.Dense(NUM_DIGITS, activation="softmax", name="digit")(h2)
    # H3
    h3 = layers.Dense(128, activation="relu")(feat)
    out_notes = layers.Dense(NUM_NOTES, activation="sigmoid", name="notes")(h3)
    return keras.Model(inp, [out_type, out_digit, out_notes], name="sudoku_cell_multitask")

def masked_losses(y_true, y_pred):
    t_type, t_digit, t_notes = y_true["type"], y_true["digit"], y_true["notes"]
    p_type, p_digit, p_notes = y_pred
    loss_type = keras.losses.categorical_crossentropy(t_type, p_type)
    type_idx = tf.argmax(t_type, axis=-1)
    mask_digit = tf.cast(tf.reduce_any(tf.one_hot([1,2], depth=4) == tf.expand_dims(type_idx,-1), axis=-1), tf.float32)
    mask_notes = tf.cast(tf.equal(type_idx, 3), tf.float32)
    loss_digit = keras.losses.categorical_crossentropy(t_digit, p_digit) * mask_digit
    loss_notes = keras.losses.binary_crossentropy(t_notes, p_notes) * tf.expand_dims(mask_notes, -1)
    loss_notes = tf.reduce_mean(loss_notes, axis=-1)
    loss = loss_type + 1.0*loss_digit + 0.5*loss_notes
    return loss

def compile_model(model):
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=masked_losses,
                  metrics={"type":"accuracy","digit":"accuracy"})

def enable_qat(model):
    if tfmot is None:
        print("QAT unavailable."); return model
    def apply_quant(layer):
        if isinstance(layer, layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer
    annotated = tf.keras.models.clone_model(model, clone_function=apply_quant)
    with tfmot.quantization.keras.quantize_scope():
        qat_model = tfmot.quantization.keras.quantize_apply(annotated)
    qat_model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=masked_losses)
    return qat_model

if __name__ == "__main__":
    m = build_multitask_model()
    compile_model(m)
    m.summary()
    if tfmot:
        qm = enable_qat(m); qm.summary()
