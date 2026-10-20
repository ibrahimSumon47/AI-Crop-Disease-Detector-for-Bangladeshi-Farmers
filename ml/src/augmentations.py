"""
augmentations.py — training-time image augmentation pipeline.

WHY AUGMENT
    Farm photos look very different from the lab datasets: different light,
    different backgrounds (soil, sky, measuring tape), camera shake, zoom.
    Augmentation teaches the network to ignore those art-style factors so
    it focuses on the leaf features. This is the single biggest defence
    against the lab-vs-field gap.

 WHAT IT PRODUCES
    A tf.data-ready function `build_train_dataset()` that resizes, then
    randomly applies, per image:
      * random rotation (±15°)       — leaves point any direction
      * random horizontal flip       — left/right symmetry of leaves
      * random vertical flip         — potted plants photographed upside down
      * brightness jitter (±20 %)    — early morning vs noon light
      * contrast jitter (±20 %)      — hazy vs sunny conditions
      * random zoom/crop (0.8-1.1×)  — arm's length vs closeup photos
      * slight blur (rare, 10 %)     — motion shake on cheap phones
      * background/color variation   — saturation & hue mild shifts
    Validation/test data get ONLY resize — never augmentation, so metrics
    stay trustworthy.

 NOTE
    We use Keras preprocessing layers exactly as they will behave during
    training; the same normalization constants are exported and reused by
    the backend, so serving-time preprocessing matches training exactly.
"""

from __future__ import annotations

import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants — single source of truth for serving-time preprocessing too.
# The backend (services/preprocess.py) must use the SAME values, which is
# why they are written again in backend/app/config.py and checked in tests.
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)          # MobileNetV2 native input
NORMALIZATION = "mobilenet_v2"   # [-1, +1] scaling, matches tf.keras.applications.mobilenet_v2.preprocess_input

# Augmentation strengths — deliberately moderate: too strong and the model
# learns noise, too weak and it overfits the lab look.
ROTATION_RANGE = 0.15            # radians  (~8.6°) applied via random_rotation factor
BRIGHTNESS_JITTER = 0.2          # ±20 %
CONTRAST_JITTER = 0.2            # ±20 %
ZOOM_RANGE = (0.8, 1.1)          # zoom-in factor range for random zoom
BLUR_PROBABILITY = 0.10          # 10 % of train images receive tiny blur


# ---------------------------------------------------------------------------
# Individual ops (tiny wrappers so the pipeline below reads at a glance)
# ---------------------------------------------------------------------------
def resize_image(image: tf.Tensor) -> tf.Tensor:
    """Resize to the fixed model input shape; keep RGB."""
    return tf.image.resize(tf.convert_to_tensor(image), IMAGE_SIZE)


def random_rotation(image: tf.Tensor) -> tf.Tensor:
    """Rotate ±~8° — enough for handheld photos, small enough to keep leaf veins readable."""
    return tf.keras.layers.RandomRotation(ROTATION_RANGE, fill_mode="reflect")(image)


def random_flips(image: tf.Tensor) -> tf.Tensor:
    """Random horizontal + vertical flips."""
    image = tf.image.random_flip_left_right(image)
    return tf.image.random_flip_up_down(image)


def random_brightness(image: tf.Tensor) -> tf.Tensor:
    """Brightness jitter; keep within valid range."""
    return tf.image.random_brightness(image, max_delta=BRIGHTNESS_JITTER)


def random_contrast(image: tf.Tensor) -> tf.Tensor:
    """Contrast jitter — simulates haze vs direct sun."""
    return tf.image.random_contrast(
        image, lower=1.0 - CONTRAST_JITTER, upper=1.0 + CONTRAST_JITTER)


def random_zoom(image: tf.Tensor) -> tf.Tensor:
    """Random crop/zoom: scales image content by a factor in ZOOM_RANGE."""
    return tf.keras.layers.RandomZoom(height_factor=(-(1.0 - ZOOM_RANGE[1]), (1.0 - ZOOM_RANGE[0])), fill_mode="reflect")(image)


def random_color_jitter(image: tf.Tensor) -> tf.Tensor:
    """Mild saturation / hue shifts — different phones render color differently."""
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return tf.image.random_hue(image, max_delta=0.05)


def maybe_blur(image: tf.Tensor) -> tf.Tensor:
    """With probability BLUR_PROBABILITY apply a tiny 3x3 blur (camera shake)."""
    def _blur() -> tf.Tensor:
        # A normalized 3x3 kernel adds mild softness.
        kernel = tf.constant(
            [[1.0, 2.0, 1.0],
             [2.0, 4.0, 2.0],
             [1.0, 2.0, 1.0]], dtype=tf.float32) / 16.0
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        image_f = tf.expand_dims(image, axis=0)
        blurred = tf.nn.depthwise_conv2d(image_f, kernel, strides=[1, 1, 1, 1], padding="SAME")
        return tf.squeeze(blurred, axis=0)

    should_blur = tf.random.uniform([], 0.0, 1.0) < BLUR_PROBABILITY
    # Use tf.cond to keep the graph pure (works with tf.data pipelines).
    return tf.cond(should_blur, _blur, lambda: image)


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Scale to the MobileNetV2 expected [-1, +1] range (matches preprocess_input)."""
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def train_transform(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Full augmentation for TRAIN data (resize -> augment -> normalize)."""
    image = resize_image(image)
    # maps expect uint8→float normalization order; augment in [0,1] then normalize.
    image = tf.cast(image, tf.float32) / 255.0
    image = random_flips(image)
    image = random_rotation(image)
    image = random_brightness(image)
    image = random_contrast(image)
    image = random_zoom(image)
    image = random_color_jitter(image)
    image = maybe_blur(image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = image * 255.0                       # back to 0..255 before [-1,1] scaling
    image = normalize_image(image)
    return image, label


def eval_transform(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """NO augmentation for val/test — only resize + normalize (deterministic metrics)."""
    image = resize_image(image)
    image = normalize_image(tf.cast(image, tf.float32))
    return image, label


def build_datasets(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[str, int]]:
    """Build tf.data pipelines directly from the split folders.

    Returns (train_ds, val_ds, class_names) where class_names maps to the
    folder order used by image_dataset_from_directory — the SAME order the
    backend's labels.json must use (checked by export_onnx.py).
    """
    common = dict(
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, seed=seed, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, seed=seed, shuffle=False, **common)

    # Save/restore the class list order (alphabetical folder names).
    class_names = train_ds.class_names

    # Apply augmentation only to training; prefetch both for speed.
    train_ds = (
        train_ds.unbatch()
        .map(train_transform, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds.unbatch()
        .map(eval_transform, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, class_names
