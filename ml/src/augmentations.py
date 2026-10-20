"""
augmentations.py — training-time image augmentation pipeline.

WHY AUGMENT
    Farm photos look very different from the lab datasets: different light,
    different backgrounds (soil, sky, a hand holding the leaf), camera shake,
    zoom level. Augmentation teaches the network to ignore those factors and
    focus on the leaf symptoms themselves. It is the single biggest defence
    against the lab-vs-field accuracy gap.

WHAT IT PRODUCES
    `build_datasets(train_dir, val_dir)` returns tf.data pipelines that resize
    and, for TRAINING ONLY, randomly apply per image:
      * random rotation (±8.6°)   — leaves point in any direction
      * random horizontal flip    — left/right symmetry of leaves
      * random vertical flip      — plants photographed from any angle
      * brightness jitter (±20 %) — early morning vs noon light
      * contrast jitter (±20 %)   — hazy vs sunny conditions
      * random zoom/crop (0.8–1.0× crop) — arm's length vs close-up photos
      * mild saturation/hue shift — different phone camera colour science
      * slight blur (10 % of images) — cheap-phone motion shake

    Validation/test data get ONLY resize + normalize, so reported metrics stay
    trustworthy (never augment the data you measure on).

IMPORTANT IMPLEMENTATION NOTE (why no tf.keras.layers here)
    Keras preprocessing layers such as RandomRotation/RandomZoom create an
    internal seed-generator variable the first time they are called. Calling
    them inside a `tf.data.Dataset.map()` function therefore raises
        ValueError: tf.function-decorated function tried to create variables
        on non-first call  ("singletTFVariable")
    as soon as tf.data re-traces the function for a second worker thread.
    That exact crash killed our first Colab run, so every transform below is
    written with STATELESS TensorFlow ops (tf.image.*, tf.raw_ops.*) instead.
    Same randomness, no variables, safe under num_parallel_calls=AUTOTUNE.

SERVING CONTRACT
    The constants here (input size, normalization scheme) are the contract
    with the backend: backend/app/services/preprocess.py must reproduce
    exactly this resize + [-1, +1] scaling, otherwise live predictions drift
    even though validation accuracy looks fine.
"""

from __future__ import annotations

import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants — single source of truth for training AND serving preprocessing.
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)  # MobileNetV2 native input resolution
NORMALIZATION = "mobilenet_v2"  # scale pixels to [-1, +1]

# Augmentation strengths: deliberately moderate. Too strong and the model
# learns noise; too weak and it simply memorises the lab look.
ROTATION_RANGE = 0.15  # radians, ~ ±8.6 degrees
BRIGHTNESS_JITTER = 0.2  # ±20 %
CONTRAST_JITTER = 0.2  # ±20 %
ZOOM_MIN_CROP = 0.80  # keep 80–100 % of each side, then resize back up
BLUR_PROBABILITY = 0.10  # 10 % of training images get a light blur
HUE_JITTER = 0.02  # keep hue shifts tiny: colour is a symptom signal!
SATURATION_MIN, SATURATION_MAX = 0.85, 1.15


# ---------------------------------------------------------------------------
# Individual ops (tiny readable wrappers, all stateless)
# ---------------------------------------------------------------------------
def resize_image(image: tf.Tensor) -> tf.Tensor:
    """Resize any input image to the fixed model input shape, keeping RGB."""
    return tf.image.resize(tf.convert_to_tensor(image), IMAGE_SIZE)


def random_rotation(image: tf.Tensor) -> tf.Tensor:
    """Rotate by a random angle in [-ROTATION_RANGE, +ROTATION_RANGE] radians.

    Implemented with `tf.raw_ops.ImageProjectiveTransformV3` (GPU-friendly,
    stateless). The 2x3 matrix maps output pixel coordinates back to input
    coordinates, so we build the INVERSE rotation matrix:

        [  cos  sin  0 ]
        [ -sin  cos  0 ]

    Out-of-frame pixels are filled by reflection so no black corners appear.
    """
    angle = tf.random.uniform([], -ROTATION_RANGE, ROTATION_RANGE)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    # transforms must be a [batch, 8] float32 tensor (the op reads the first
    # six entries of the 2x3 affine matrix; the last two are padding).
    transform = tf.stack(
        [cos_a, sin_a, 0.0, -sin_a, cos_a, 0.0, 0.0, 0.0], axis=0
    )
    batched = tf.expand_dims(tf.cast(image, tf.float32), axis=0)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=batched,
        transforms=tf.expand_dims(transform, axis=0),
        output_shape=tf.constant(IMAGE_SIZE, dtype=tf.int32),
        fill_value=tf.constant(0.0, dtype=tf.float32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )
    return tf.squeeze(rotated, axis=0)


def random_flips(image: tf.Tensor) -> tf.Tensor:
    """Random horizontal + vertical flip (leaves are symmetric enough for this)."""
    image = tf.image.random_flip_left_right(image)
    return tf.image.random_flip_up_down(image)


def random_brightness(image: tf.Tensor) -> tf.Tensor:
    """Brightness jitter — early-morning vs midday sunlight."""
    return tf.image.random_brightness(image, max_delta=BRIGHTNESS_JITTER)


def random_contrast(image: tf.Tensor) -> tf.Tensor:
    """Contrast jitter — hazy monsoon air vs hard direct sun."""
    return tf.image.random_contrast(
        image, lower=1.0 - CONTRAST_JITTER, upper=1.0 + CONTRAST_JITTER
    )


def random_zoom(image: tf.Tensor) -> tf.Tensor:
    """Zoom by cropping 80–100 % of the image and resizing back to 224x224.

    Pure tf.image ops (no Keras layer, no variables):
      1. pick a random crop fraction in [ZOOM_MIN_CROP, 1.0]
      2. crop that window at a random offset
      3. resize it back to the model input size -> a zoom-in effect
    """
    crop_fraction = tf.random.uniform([], ZOOM_MIN_CROP, 1.0)
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    new_h = tf.cast(tf.cast(height, tf.float32) * crop_fraction, tf.int32)
    new_w = tf.cast(tf.cast(width, tf.float32) * crop_fraction, tf.int32)
    # Always keep at least 1 pixel so crop_to_bounding_box never fails.
    new_h = tf.maximum(new_h, 1)
    new_w = tf.maximum(new_w, 1)

    # Random (not necessarily centred) offset => also acts as a random crop.
    max_top = tf.maximum(height - new_h, 0)
    max_left = tf.maximum(width - new_w, 0)
    top = tf.random.uniform([], 0, max_top + 1, dtype=tf.int32)
    left = tf.random.uniform([], 0, max_left + 1, dtype=tf.int32)

    cropped = tf.image.crop_to_bounding_box(image, top, left, new_h, new_w)
    return tf.image.resize(cropped, IMAGE_SIZE)


def random_color_jitter(image: tf.Tensor) -> tf.Tensor:
    """Mild saturation / hue shift — different phones render colour differently.

    Kept intentionally small: yellowing, browning and lesion colour are real
    diagnostic signals, so heavy colour jitter would destroy information.
    """
    image = tf.image.random_saturation(
        image, lower=SATURATION_MIN, upper=SATURATION_MAX
    )
    return tf.image.random_hue(image, max_delta=HUE_JITTER)


def maybe_blur(image: tf.Tensor) -> tf.Tensor:
    """With probability BLUR_PROBABILITY, apply a light 3x3 Gaussian-ish blur.

    Simulates cheap-phone motion shake / slightly out-of-focus shots.

    Kernel shape matters: `tf.nn.depthwise_conv2d` expects
    [kH, kW, in_channels, channel_multiplier]. For a 3-channel RGB image with
    one output channel per input channel that is [3, 3, 3, 1] — an earlier
    version passed [3, 3, 1, 1], which made the op demand a 1-channel input
    and crashed the whole Colab training run.
    """
    # 3x3 Gaussian-like kernel (sums to 1), applied independently per channel.
    kernel_2d = tf.constant(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=tf.float32
    ) / 16.0
    kernel = tf.tile(tf.reshape(kernel_2d, [3, 3, 1, 1]), [1, 1, 3, 1])

    def _blur() -> tf.Tensor:
        batched = tf.expand_dims(image, axis=0)
        blurred = tf.nn.depthwise_conv2d(
            batched, kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        return tf.squeeze(blurred, axis=0)

    should_blur = tf.random.uniform([], 0.0, 1.0) < BLUR_PROBABILITY
    # tf.cond keeps the graph pure and works inside tf.data pipelines.
    return tf.cond(should_blur, _blur, lambda: image)


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Scale pixel values from [0, 255] to [-1, +1] (MobileNetV2 convention)."""
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def train_transform(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Full augmentation for TRAIN data: resize -> augment (in [0,1]) -> [-1,1]."""
    image = resize_image(image)
    # Augment in the [0, 1] range: tf.image ops expect/return that range.
    image = tf.cast(image, tf.float32) / 255.0

    image = random_flips(image)
    image = random_rotation(image)
    image = random_brightness(image)
    image = random_contrast(image)
    image = random_zoom(image)
    image = random_color_jitter(image)
    image = maybe_blur(image)

    # Clamp before scaling up so brightness/contrast do not overshoot.
    image = tf.clip_by_value(image, 0.0, 1.0) * 255.0
    image = normalize_image(image)
    return image, label


def eval_transform(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """NO augmentation for val/test — only resize + normalize (deterministic metrics)."""
    image = resize_image(image)
    return normalize_image(image), label


def build_datasets(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Build tf.data pipelines straight from the split folders.

    Args:
        train_dir: folder of `train/<class_name>/*.jpg` images.
        val_dir:   folder of `val/<class_name>/*.jpg` images.
        batch_size: images per batch (32 fits a free Colab/Kaggle T4).
        seed: fixed seed so shuffling is reproducible across runs.

    Returns:
        (train_ds, val_ds, class_names) where `class_names` is the alphabetical
        folder order used by Keras. That SAME order must be written to
        labels.json by export_onnx.py, otherwise the index <-> disease mapping
        in the backend would silently be wrong.
    """
    common = dict(
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, seed=seed, **common
    )
    # shuffle=False for validation so metrics are stable between epochs.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, seed=seed, shuffle=False, **common
    )

    class_names = list(train_ds.class_names)

    # Augment only the training split, then batch and prefetch for GPU speed.
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
