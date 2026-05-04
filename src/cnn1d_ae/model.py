# =========================
# FILE: src/cnn1d_ae/model.py
# =========================
from __future__ import annotations

import os
from typing import List

import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt


def setup_gpu() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Prioriza uma GPU cujo nome contenha "1650" (ou hint via env).
            hint = os.getenv("GPU_NAME_HINT", "1650").lower().strip()
            chosen = None
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                name = str(details.get("device_name", "")).lower()
                if hint and hint in name:
                    chosen = gpu
                    break

            if chosen is None:
                chosen = gpus[0]

            tf.config.set_visible_devices(chosen, "GPU")
            tf.config.experimental.set_memory_growth(chosen, True)
            details = tf.config.experimental.get_device_details(chosen)
            print(f"[GPU] Usando: {details.get('device_name', chosen.name)} | memory_growth habilitado.")
        else:
            print("[GPU] Nenhuma GPU detectada. Rodando em CPU.")
    except Exception as e:
        print(f"[GPU] Aviso: não foi possível configurar GPU: {e}")


def build_callbacks(patience: int) -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-6
        ),
    ]


def build_cnn1d_autoencoder(hp: kt.HyperParameters, time_steps: int, n_features: int) -> keras.Model:
    f1 = hp.Choice("filters_1", [8, 16, 32, 64])
    f2 = hp.Choice("filters_2", [8, 16, 32, 64])
    k1 = hp.Choice("kernel_1", [3, 5, 7, 9])
    k2 = hp.Choice("kernel_2", [3, 5, 7, 9])

    dropout = hp.Float("dropout", 0.0, 0.4, step=0.1)
    lr = hp.Choice("lr", [1e-4, 3e-4, 1e-3, 3e-3])

    s1 = hp.Choice("stride_1", [1, 2])
    s2 = hp.Choice("stride_2", [1, 2])

    inputs = keras.Input(shape=(time_steps, n_features))

    x = layers.Conv1D(filters=f1, kernel_size=k1, padding="same", strides=s1, activation="relu")(inputs)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters=f2, kernel_size=k2, padding="same", strides=s2, activation="relu")(x)

    x = layers.Conv1DTranspose(filters=f2, kernel_size=k2, padding="same", strides=s2, activation="relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv1DTranspose(filters=f1, kernel_size=k1, padding="same", strides=s1, activation="relu")(x)
    outputs = layers.Conv1DTranspose(filters=n_features, kernel_size=3, padding="same")(x)

    model = keras.Model(inputs, outputs, name="cnn1d_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
