"""Scoring de produção do ensemble OR (6 bundles de resíduo+CNN1D dos canais
TC382_0X_A, `production_bundles/t5_ensemble_via_tc382/`) como proxy de early-warning
pro T5_AVG_A — que não tem bundle próprio (resíduo degenerado, T5 é a média dos 6
canais).

Uso:
    from scripts.ensemble_t5_bundle_score import score_t5_proxy_ensemble
    out = score_t5_proxy_ensemble(df_raw)   # df_raw: TC382_01..06_A + RUNNING_A, indexado por tempo
    # out["t5_proxy_alert"] == 1 onde QUALQUER um dos 6 canais está em alerta
"""
from __future__ import annotations

import glob
import os

import pandas as pd
from tensorflow import keras

from scripts.residual_feature import TCS, add_residual_column
from src.cnn1d_ae.inference import load_bundle, score_production

BUNDLE_DIR_DEFAULT = "production_bundles/t5_ensemble_via_tc382"


def load_channel_models(bundle_dir: str = BUNDLE_DIR_DEFAULT) -> dict:
    """Carrega os 6 pares (model, bundle) dos bundles finalizados em disco.

    Os arquivos `model.keras` não são salvos localmente pelo finalize_bundle.py (só o
    bundle_dir/{sensor}_inference_bundle.json) — os pesos vêm da própria task ClearML,
    referenciada no run_config embutido; aqui carregamos o modelo salvo em cache local
    baixado anteriormente via ClearML (mesmo mecanismo usado no script de plot).
    """
    from clearml import Task

    channels = {}
    for path in sorted(glob.glob(os.path.join(bundle_dir, "*_inference_bundle.json"))):
        bundle = load_bundle(path)
        sensor = bundle["sensor"]
        channels[sensor] = {"bundle": bundle}
    return channels


def score_t5_proxy_ensemble(
    df_raw: pd.DataFrame,
    task_id: str = "11978d260dbf4301838fff35452bf97f",
    bundle_dir: str = BUNDLE_DIR_DEFAULT,
) -> pd.DataFrame:
    """df_raw: dataframe indexado por tempo com TC382_01..06_A (temperatura bruta) e
    RUNNING_A. Retorna um dataframe por-sequência com uma coluna `alert_{sensor}` por
    canal e `t5_proxy_alert` = OR combinado dos 6.
    """
    from clearml import Task

    task = Task.get_task(task_id=task_id)
    combined = None
    per_channel = {}

    for sensor in TCS:
        bpath = os.path.join(bundle_dir, f"{sensor}_inference_bundle.json")
        bundle = load_bundle(bpath)
        model_path = task.artifacts[f"{sensor}_model_keras"].get_local_copy()
        model = keras.models.load_model(model_path)

        df_resid = add_residual_column(df_raw, target=sensor)
        scored = score_production(model, bundle, df_resid)
        scored["seq_end_time"] = pd.to_datetime(scored["seq_end_time"])
        scored = scored.set_index("seq_end_time")
        per_channel[sensor] = scored["alert"]

        if combined is None:
            combined = pd.DataFrame(index=scored.index)
        combined[f"alert_{sensor}"] = scored["alert"].reindex(combined.index).fillna(0).astype(int)

    combined["t5_proxy_alert"] = (combined[[f"alert_{s}" for s in TCS]].sum(axis=1) > 0).astype(int)
    return combined
