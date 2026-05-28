"""Backend per_sensor: 1 AE univariado por sensor, combinado por MAX.

Validado out-of-sample (scripts/ae_per_sensor_temporal.py):
- H=8h:  recall 0.58 / FA/dia 0.01  (vs multi 0.51 / 0.05)
- H=24h: recall 0.64 / FA/dia 0.00  (vs multi 0.57 / 0.03)
- H=72h: recall 0.83 / FA/dia 0.00  (vs multi 0.79 / 0.00)

Por que MAX é o aggregator natural:
- Cada AE univariado produz erro de reconstrução do seu próprio canal.
- O combinado MAX(.) é equivalente operacionalmente ao "OR de qualquer sensor
  acima do seu threshold quantilico" quando cada sensor tem escala MAE similar
  (o que ocorre porque os inputs sao normalizados antes do AE).
- 1 sensor degradando = combinado sobe imediatamente, sem diluicao pelos 16 normais.
"""
from __future__ import annotations
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from .config import PipelineConfig
from .scoring import reconstruction_mae_per_seq


def _build_univariate_ae(time_steps: int, f1: int, f2: int, s1: int, s2: int,
                         dropout: float = 0.1, l2: float = 1e-4) -> keras.Model:
    """AE pequeno e fixo para um único canal. Sem busca de hiperparâmetros."""
    reg = keras.regularizers.l2(l2) if l2 > 0 else None
    inp = keras.Input(shape=(time_steps, 1))
    x = layers.Conv1D(f1, 7, padding="same", strides=s1,
                      activation="relu", kernel_regularizer=reg)(inp)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(f2, 7, padding="same", strides=s2,
                      activation="relu", kernel_regularizer=reg)(x)
    x = layers.Conv1DTranspose(f2, 7, padding="same", strides=s2,
                               activation="relu", kernel_regularizer=reg)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv1DTranspose(f1, 7, padding="same", strides=s1,
                               activation="relu", kernel_regularizer=reg)(x)
    out = layers.Conv1DTranspose(1, 3, padding="same")(x)
    m = keras.Model(inp, out, name="cnn1d_ae_univariate")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
              metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return m


def train_per_sensor(
    cfg: PipelineConfig,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_train_full: np.ndarray,
    x_all: np.ndarray,
    sensors: List[str],
) -> Dict:
    """Treina 1 AE univariado por canal e produz o sinal combinado.

    Reutiliza as sequências já construídas no pipeline_multi (x_train, x_val,
    x_train_full, x_all têm shape [n_seq, TIME_STEPS, n_sensors]). Cada modelo
    é treinado no canal i, fatiado pra (n_seq, TIME_STEPS, 1).

    Retorna:
      models: dict[sensor → keras.Model]
      mae_per_sensor_seq: (n_seq, n_sensors) MAE por sequência por sensor (em x_all)
      combined_mae_seq: (n_seq,) MAX across sensors — substitui mae_seq_all
      train_mae_per_sensor: (n_train_full, n_sensors) MAE no train_full
      train_combined_mae_seq: (n_train_full,) MAX — substitui train_mae_seq
    """
    n_sensors = len(sensors)
    n_seq = x_all.shape[0]
    n_train_full = x_train_full.shape[0]

    mae_per_sensor = np.empty((n_seq, n_sensors), dtype=np.float32)
    train_mae_per_sensor = np.empty((n_train_full, n_sensors), dtype=np.float32)
    models: Dict[str, keras.Model] = {}

    es_patience = max(2, int(cfg.PATIENCE) // 2)
    print(f"[PER-SENSOR] treinando {n_sensors} AEs univariados "
          f"(f1={cfg.PER_SENSOR_F1} f2={cfg.PER_SENSOR_F2} "
          f"s={cfg.PER_SENSOR_S1}x{cfg.PER_SENSOR_S2} epochs={cfg.PER_SENSOR_EPOCHS})")

    for i, sensor in enumerate(sensors):
        x_train_i = x_train[:, :, i:i+1]
        x_val_i = x_val[:, :, i:i+1]
        x_train_full_i = x_train_full[:, :, i:i+1]
        x_all_i = x_all[:, :, i:i+1]

        model = _build_univariate_ae(
            time_steps=cfg.TIME_STEPS,
            f1=cfg.PER_SENSOR_F1, f2=cfg.PER_SENSOR_F2,
            s1=cfg.PER_SENSOR_S1, s2=cfg.PER_SENSOR_S2,
        )
        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=es_patience,
                restore_best_weights=True, mode="min",
            ),
        ]
        model.fit(
            x_train_i, x_train_i,
            validation_data=(x_val_i, x_val_i),
            epochs=int(cfg.PER_SENSOR_EPOCHS),
            batch_size=int(cfg.BATCH_SIZE),
            verbose=2,
            callbacks=cb,
        )
        mae_per_sensor[:, i] = reconstruction_mae_per_seq(model, x_all_i, cfg.BATCH_SIZE)
        train_mae_per_sensor[:, i] = reconstruction_mae_per_seq(model, x_train_full_i, cfg.BATCH_SIZE)
        models[sensor] = model
        print(f"[PER-SENSOR] [{i+1:2d}/{n_sensors}] {sensor} OK")
        keras.backend.clear_session()

    # Aggregator: MAX across sensors. Proxy operacional do OR-de-qualquer-sensor.
    combined_mae_seq = mae_per_sensor.max(axis=1).astype(np.float32)
    train_combined_mae_seq = train_mae_per_sensor.max(axis=1).astype(np.float32)
    return {
        "models": models,
        "mae_per_sensor_seq": mae_per_sensor,
        "combined_mae_seq": combined_mae_seq,
        "train_mae_per_sensor": train_mae_per_sensor,
        "train_combined_mae_seq": train_combined_mae_seq,
    }
