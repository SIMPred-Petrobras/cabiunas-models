# =========================
# FILE: src/cnn1d_ae/tuning.py
# =========================
from __future__ import annotations

from typing import Tuple, Dict

import pandas as pd
import keras_tuner as kt
from tensorflow import keras

from .config import PipelineConfig
from .model import build_cnn1d_autoencoder, build_callbacks


def run_tuner(
    cfg: PipelineConfig,
    out_dirs: Dict[str, str],
    x_train,
    x_val,
    n_features: int,
) -> Tuple[kt.HyperParameters, keras.Model, pd.DataFrame]:
    # Fixa a seed global (Python/NumPy/TF) ANTES da busca -- sem isso, a
    # amostragem de hiperparametros do RandomSearch e a inicializacao de
    # pesos de cada trial usam o estado aleatorio global do processo, que
    # nao e reproduzivel entre execucoes remotas separadas. Confirmado
    # via seed-sweep (docs/analise_cnn1dae_exp13.md) que o CNN1D-AE tinha
    # variancia real (85,0%-92,5% de hit_rate entre sementes) mesmo
    # fixando RANDOM_SEED no config -- o proprio best_hp vencedor podia
    # mudar entre runs porque a busca em si nao era determinada por essa
    # seed. `seed=` no RandomSearch fixa a amostragem do oracle; o
    # set_random_seed cobre a inicializacao de pesos de cada trial.
    keras.utils.set_random_seed(cfg.RANDOM_SEED)

    def hypermodel(hp: kt.HyperParameters):
        return build_cnn1d_autoencoder(hp, cfg.TIME_STEPS, n_features)

    tuner = kt.RandomSearch(
        hypermodel=hypermodel,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=cfg.MAX_TRIALS,
        executions_per_trial=cfg.EXECUTIONS_PER_TRIAL,
        directory=out_dirs["tuner"],
        project_name="cnn1d_ae_trials",
        overwrite=True,
        seed=cfg.RANDOM_SEED,
    )

    callbacks = build_callbacks(cfg.PATIENCE)

    tuner.search(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
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


def refit_best_model(cfg: PipelineConfig, best_model: keras.Model, x_train, x_val) -> keras.callbacks.History:
    # Reseed logo antes do fit final -- os pesos de best_model ja saem
    # deterministicos da busca (run_tuner fixa a seed antes do
    # tuner.search()), mas .fit() reembaralha o dataset por epoca por
    # padrao (shuffle=True); sem reseed aqui, essa reembaralhada consome
    # o estado aleatorio global residual deixado pelo numero variavel de
    # operacoes da busca (MAX_TRIALS trials), que nao e o mesmo entre
    # execucoes mesmo com a mesma RANDOM_SEED fixada no inicio.
    keras.utils.set_random_seed(cfg.RANDOM_SEED)
    callbacks = build_callbacks(cfg.PATIENCE)
    history = best_model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return history
