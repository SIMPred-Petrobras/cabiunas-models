# =========================
# FILE: src/cnn1d_ae/tuning.py
# =========================
from __future__ import annotations

from typing import Tuple, Dict

import pandas as pd
import keras_tuner as kt
from tensorflow import keras

from .config import PipelineConfig
from .model import (
    build_cnn1d_autoencoder,
    build_gru_autoencoder,
    build_lstm_autoencoder,
    build_dense_autoencoder,
    build_callbacks,
)

_BUILDERS = {
    "cnn1d": build_cnn1d_autoencoder,
    "gru":   build_gru_autoencoder,
    "lstm":  build_lstm_autoencoder,
    "dense": build_dense_autoencoder,
}


def run_tuner(
    cfg: PipelineConfig,
    out_dirs: Dict[str, str],
    x_train,
    x_val,
    n_features: int,
) -> Tuple[kt.HyperParameters, keras.Model, pd.DataFrame]:
    arch = getattr(cfg, "MODEL_ARCH", "cnn1d")
    builder = _BUILDERS.get(arch)
    if builder is None:
        raise ValueError(f"MODEL_ARCH '{arch}' não reconhecido. Opções: {list(_BUILDERS)}")
    print(f"[TUNER] Arquitetura: {arch}")

    def hypermodel(hp: kt.HyperParameters):
        return builder(hp, cfg.TIME_STEPS, n_features)

    tuner = kt.RandomSearch(
        hypermodel=hypermodel,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=cfg.MAX_TRIALS,
        executions_per_trial=cfg.EXECUTIONS_PER_TRIAL,
        directory=out_dirs["tuner"],
        project_name="cnn1d_ae_trials",
        overwrite=True,
    )

    callbacks = build_callbacks(cfg.PATIENCE)

    tuner.search(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    trials_rows = []
    for t in tuner.oracle.get_best_trials(num_trials=cfg.MAX_TRIALS):
        row = {"trial_id": t.trial_id, "score_val_loss": t.score}
        row.update(t.hyperparameters.values)
        trials_rows.append(row)

    df_trials = pd.DataFrame(trials_rows).sort_values("score_val_loss", ascending=True)
    return best_hp, best_model, df_trials


def refit_best_model(cfg: PipelineConfig, best_model: keras.Model, x_train, x_val, x_anom=None) -> keras.callbacks.History:
    callbacks = build_callbacks(cfg.PATIENCE, x_anom=x_anom, batch_size=cfg.BATCH_SIZE)
    history = best_model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    return history
