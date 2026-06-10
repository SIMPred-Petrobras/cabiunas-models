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


class AnomalyValLoss(keras.callbacks.Callback):
    """Mede o MSE de reconstrucao num conjunto de sequencias anomalas (janelas de
    alarme) ao fim de cada epoca e registra em logs['val_loss_anomaly'].
    Para um AE de anomalia saudavel, essa curva deve ficar ACIMA da val_loss normal."""

    def __init__(self, x_anom, batch_size: int):
        super().__init__()
        self.x_anom = x_anom
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or self.x_anom is None or len(self.x_anom) == 0:
            return
        res = self.model.evaluate(self.x_anom, self.x_anom,
                                  batch_size=self.batch_size, verbose=0, return_dict=True)
        logs["val_loss_anomaly"] = float(res.get("loss", res.get("mse", 0.0)))


def build_callbacks(patience: int, x_anom=None, batch_size: int = 256) -> List[keras.callbacks.Callback]:
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-6
        ),
    ]
    if x_anom is not None and len(x_anom) > 0:
        cbs.append(AnomalyValLoss(x_anom, batch_size))
    return cbs


def build_gru_autoencoder(hp: kt.HyperParameters, time_steps: int, n_features: int) -> keras.Model:
    """GRU Seq2Seq autoencoder.

    Encoder: GRU(u1, return_sequences=True) → GRU(u2, return_sequences=False) → bottleneck
    Decoder: RepeatVector(T) → GRU(u2) → GRU(u1) → TimeDistributed(Dense)
    """
    u1 = hp.Choice("units_1", [16, 32, 64])
    u2 = hp.Choice("units_2", [8, 16, 32])
    dropout = hp.Float("dropout", 0.0, 0.4, step=0.1)
    lr = hp.Choice("lr", [1e-4, 3e-4, 1e-3, 3e-3])
    reg = keras.regularizers.l2(1e-4)

    latent_dim = u2
    input_dim  = time_steps * n_features
    ratio = latent_dim / max(input_dim, 1)
    print(f"[GRU-AE] bottleneck={latent_dim} vs entrada={input_dim} "
          f"(ratio={ratio:.3f}) | u1={u1} u2={u2}")

    inputs = keras.Input(shape=(time_steps, n_features))

    # Encoder
    x = layers.GRU(u1, return_sequences=True, kernel_regularizer=reg)(inputs)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    encoded = layers.GRU(u2, return_sequences=False, kernel_regularizer=reg)(x)

    # Decoder
    x = layers.RepeatVector(time_steps)(encoded)
    x = layers.GRU(u2, return_sequences=True, kernel_regularizer=reg)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.GRU(u1, return_sequences=True, kernel_regularizer=reg)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = keras.Model(inputs, outputs, name="gru_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_lstm_autoencoder(hp: kt.HyperParameters, time_steps: int, n_features: int) -> keras.Model:
    """LSTM Seq2Seq autoencoder.

    Encoder: LSTM(u1, return_sequences=True) → LSTM(u2, return_sequences=False) → bottleneck
    Decoder: RepeatVector(T) → LSTM(u2) → LSTM(u1) → TimeDistributed(Dense)
    """
    u1      = hp.Choice("units_1", [16, 32, 64])
    u2      = hp.Choice("units_2", [8, 16, 32])
    dropout = hp.Float("dropout", 0.0, 0.4, step=0.1)
    lr      = hp.Choice("lr", [1e-4, 3e-4, 1e-3, 3e-3])
    reg     = keras.regularizers.l2(1e-4)

    latent_dim = u2
    input_dim  = time_steps * n_features
    ratio      = latent_dim / max(input_dim, 1)
    print(f"[LSTM-AE] bottleneck={latent_dim} vs entrada={input_dim} "
          f"(ratio={ratio:.3f}) | u1={u1} u2={u2}")

    inputs = keras.Input(shape=(time_steps, n_features))

    x       = layers.LSTM(u1, return_sequences=True,  kernel_regularizer=reg)(inputs)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    encoded = layers.LSTM(u2, return_sequences=False, kernel_regularizer=reg)(x)

    x = layers.RepeatVector(time_steps)(encoded)
    x = layers.LSTM(u2, return_sequences=True,  kernel_regularizer=reg)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x       = layers.LSTM(u1, return_sequences=True,  kernel_regularizer=reg)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_cnn1d_autoencoder(hp: kt.HyperParameters, time_steps: int, n_features: int) -> keras.Model:
    f1 = hp.Choice("filters_1", [8, 16, 32, 64])
    f2 = hp.Choice("filters_2", [8, 16, 32, 64])
    k1 = hp.Choice("kernel_1", [3, 5, 7, 9])
    k2 = hp.Choice("kernel_2", [3, 5, 7, 9])

    dropout = hp.Float("dropout", 0.0, 0.4, step=0.1)
    lr = hp.Choice("lr", [1e-4, 3e-4, 1e-3, 3e-3])

    # Forca downsampling (s1>=1, s2>=2) para garantir um bottleneck real: sem
    # compressao temporal o AE fica overcomplete, memoriza e nao separa anomalia.
    s1 = hp.Choice("stride_1", [1, 2])
    s2 = hp.Choice("stride_2", [2, 4])

    # L2 weight decay leve para regularizar.
    reg = keras.regularizers.l2(1e-4)

    inputs = keras.Input(shape=(time_steps, n_features))

    x = layers.Conv1D(filters=f1, kernel_size=k1, padding="same", strides=s1,
                      activation="relu", kernel_regularizer=reg)(inputs)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters=f2, kernel_size=k2, padding="same", strides=s2,
                      activation="relu", kernel_regularizer=reg)(x)

    # Tamanho do espaco latente vs entrada (visibilidade do bottleneck).
    import math
    latent_dim = math.ceil(time_steps / (s1 * s2)) * f2
    input_dim = time_steps * n_features
    ratio = latent_dim / max(input_dim, 1)
    print(f"[AE] latente={latent_dim} vs entrada={input_dim} "
          f"(ratio={ratio:.2f} | {'OVERCOMPLETE!' if ratio >= 1 else 'comprimido'}) "
          f"| f1={f1} f2={f2} s1={s1} s2={s2}")

    x = layers.Conv1DTranspose(filters=f2, kernel_size=k2, padding="same", strides=s2,
                               activation="relu", kernel_regularizer=reg)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Conv1DTranspose(filters=f1, kernel_size=k1, padding="same", strides=s1,
                               activation="relu", kernel_regularizer=reg)(x)
    outputs_raw = layers.Conv1DTranspose(filters=n_features, kernel_size=3, padding="same")(x)

    # Garante que o decoder reconstrua exatamente time_steps passos.
    # Quando s1*s2 não divide time_steps (ex: s1=2,s2=4 com TS=60 → saída 64),
    # o crop remove o excesso sem afetar o gradiente.
    out_len = int(outputs_raw.shape[1])
    if out_len is not None and out_len != time_steps:
        excess = out_len - time_steps
        outputs = layers.Cropping1D((0, excess))(outputs_raw)
    else:
        outputs = outputs_raw

    model = keras.Model(inputs, outputs, name="cnn1d_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
