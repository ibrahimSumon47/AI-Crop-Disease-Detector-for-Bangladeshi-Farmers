"""
model.py — MobileNetV2 transfer-learning model for crop disease detection.

WHY MOBILENETV2
    * ~14 MB of weights -> the ONNX file stays well under the 50 MB repo
      rule, and loads fast inside the free-tier backend container.
    * ImageNet pretraining transfers well to leaf images (leaf texture,
      color blobs, lesion patterns live in the same low/mid layers).
    * It was designed for phones: small head => runs comfortably on
      free-tier CPU hosting (Hugging Face Spaces / Render).

ARCHITECTURE
    MobileNetV2 backbone (ImageNet weights, top removed)
      -> GlobalAveragePooling
      -> Dropout(0.3)  (regularization; leaf datasets are small)
      -> Dense(13, softmax)  (13 = len(CLASSES) from classes.py)

The model is built in two stages by train.py:
    stage 1: backbone frozen, only the new head learns
    stage 2: top ~40 backbone layers unfrozen at a tiny learning rate
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

# The canonical class count comes from classes.py — never hard-code 13 here,
# or labels.json and the model would disagree after a class list change.
from classes import N_CLASSES

# ---------------------------------------------------------------------------
# Hyperparameters (also imported by train.py to keep one source of truth)
# ---------------------------------------------------------------------------
INPUT_SHAPE = (224, 224, 3)      # MobileNetV2 native input; matches augmentations.py
DROPOUT_RATE = 0.3               # small dataset => meaningful dropout

# Stage-2 fine-tuning: how deep (from the top) we unfreeze the backbone.
FINE_TUNE_LAYERS = 40            # roughly the last 5 inverted-residual blocks
STAGE1_LR = 1e-3                 # head-only: bigger LR is safe (backbone frozen)
STAGE2_LR = 1e-5                 # fine-tuning: tiny LR so pretrained weights drift slowly


def build_model(n_classes: int = N_CLASSES) -> tf.keras.Model:
    """Build the MobileNetV2-based model (backbone frozen; training unfreezes it).

    Returns:
        A compiled Keras model ready for stage-1 (freeze) training.
    """
    # Load the pretrained backbone WITHOUT its ImageNet classification top;
    # global pooling is done ourselves so the head stays simple and ONNX-friendly.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # stage 1: freeze everything

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="leaf_image")
    # preprocess_input scales [-1, +1] — MUST match augmentations.normalize_image
    # and the backend's preprocess.py, otherwise serving-time inputs drift.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)  # keep frozen BN statistics in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(DROPOUT_RATE, name="head_dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="crop_disease_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LR),
        loss="sparse_categorical_crossentropy",   # integer labels from tf.data
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_finetune(model: tf.keras.Model) -> tf.keras.Model:
    """Unfreeze the top `FINE_TUNE_LAYERS` backbone layers and recompile at STAGE2_LR.

    Called by train.py between stage 1 and stage 2. BatchNorm layers stay
    frozen in the sense that BN statistics are not updated during forward
    pass (training=False at build time); fine-tuning updates only affine
    weights of the unfrozen layers.
    """
    model.get_layer("mobilenetv2_1.00_224").trainable = True

    # Everything except the top block stays frozen.
    base = model.get_layer("mobilenetv2_1.00_224")
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-FINE_TUNE_LAYERS:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick architecture self-test (runs without data): build + summarize.
    # NOTE: first run downloads ~9 MB of ImageNet weights into ~/.keras.
    m = build_model()
    m.summary()
    trainable = sum(w.get_value().sum() if hasattr(w, "get_value") and hasattr(w.get_value(), "sum") else 0 for w in m.trainable_weights)
    # A simpler printable check:
    print(f"Trainable variables: {len(m.trainable_weights)}")
    print(f"Output classes: {m.output_shape[-1]} (expected {N_CLASSES})")
