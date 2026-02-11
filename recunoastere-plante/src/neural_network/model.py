# src/neural_network/model.py
from __future__ import annotations

from typing import Tuple
import tensorflow as tf


def build_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    CNN simplu și stabil pentru clasificare imagini (150x150 RGB).
    Include augmentare + normalizare în model (pipeline end-to-end).
    """

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    inputs = tf.keras.Input(shape=input_shape, name="image")

    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    # Bloc 1
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Bloc 2
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Bloc 3
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    model = tf.keras.Model(inputs, outputs, name="plant_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
