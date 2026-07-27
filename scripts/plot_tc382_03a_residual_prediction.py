"""Gera o plot diagnóstico (série + MAE + alarmes) do bundle deployável de
resíduo+CNN1D pra TC382_03_A, usando o caminho de inferência de produção real
(add_residual_column + score_production), não o artefato gerado em treino (que usa
threshold_q/unidade de resíduo, não o ponto de operação calibrado).

Uso:
    PYTHONPATH=. python scripts/plot_tc382_03a_residual_prediction.py
"""
from __future__ import annotations

import pandas as pd
from clearml import Task
from tensorflow import keras

from scripts.eval_per_sensor_level import ALARM_CSV_DEFAULT, load_alarms_gap
from scripts.residual_feature import add_residual_column
from src.cnn1d_ae.inference import load_bundle, score_production
from src.cnn1d_ae.plots import plot_series_with_mae_reconstruction

TASK_ID = "11978d260dbf4301838fff35452bf97f"
BUNDLE_PATH = "production_bundles/tc382_03a_residual_cnn1d/TC382_03_A_inference_bundle.json"
RAW_CSV = "../dados/sensores_2024h2_2025_2026_30s.csv"
TARGET = "TC382_03_A"
OUT_PATH = "eval_predictive_out/TC382_03_A_residual_cnn1d_series_prediction.png"


def main() -> None:
    print("[1/5] Carregando bundle + modelo...")
    bundle = load_bundle(BUNDLE_PATH)
    task = Task.get_task(task_id=TASK_ID)
    model_key = next(k for k in task.artifacts if k == f"{TARGET}_model_keras")
    model_path = task.artifacts[model_key].get_local_copy()
    model = keras.models.load_model(model_path)

    print("[2/5] Carregando série bruta...")
    cols = ["data_datetime", "RUNNING_A", "TC382_01_A", "TC382_02_A", "TC382_03_A",
            "TC382_04_A", "TC382_05_A", "TC382_06_A"]
    raw = pd.read_csv(RAW_CSV, low_memory=False, usecols=cols)
    t = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.loc[t.notna()].copy()
    raw["data_datetime"] = t.loc[t.notna()]  # preserva tz-aware (evita .values, que descarta tz)
    raw = raw.set_index("data_datetime").sort_index()
    raw_temp = pd.to_numeric(raw[TARGET], errors="coerce")

    print("[3/5] Computando resíduo (feature de entrada do modelo) e pontuando...")
    df_resid = add_residual_column(raw.reset_index(), target=TARGET).set_index("data_datetime")
    scored = score_production(model, bundle, df_resid)
    scored["seq_end_time"] = pd.to_datetime(scored["seq_end_time"])
    df_seq_scores = scored.set_index("seq_end_time")[["mae_seq"]]
    anomalous_times = pd.DatetimeIndex(scored.loc[scored["alert"] == 1, "seq_end_time"])
    operational_state = scored.set_index("seq_end_time")["operational_state"] if "operational_state" in scored.columns else None
    # reindexado pro índice da série bruta pra _shade_machine_states funcionar corretamente
    if operational_state is not None:
        operational_state = operational_state.reindex(raw_temp.index, method="ffill")

    print("[4/5] Carregando alarmes reais de TC382_03_A...")
    raw_alarms = load_alarms_gap(ALARM_CSV_DEFAULT, [])
    alarm_times = pd.Series(pd.to_datetime(raw_alarms.get(TARGET, [])))
    if alarm_times.dt.tz is None:
        alarm_times = alarm_times.dt.tz_localize("UTC")
    else:
        alarm_times = alarm_times.dt.tz_convert("UTC")

    print(f"[5/5] Plotando -> {OUT_PATH}")
    # Painel 1 (série): pontos vermelhos = alerta de produção real (EWMA vs
    # ewma_abs_threshold). Painel 2 (MAE por sequência): linha de referência é o
    # threshold BRUTO por sequência (bundle["threshold"], calibrado por target_rate) —
    # não o ewma_abs_threshold, que vive na escala do MAE já suavizado, não do MAE cru.
    plot_series_with_mae_reconstruction(
        series=raw_temp,
        df_seq_scores=df_seq_scores,
        threshold=float(bundle["threshold"]),
        anomalous_times=anomalous_times,
        alarm_times=alarm_times,
        out_path=OUT_PATH,
        title="TC382_03_A — resíduo+CNN1D (produção calibrada)",
        operational_state=operational_state,
    )
    print(f"OK: {len(anomalous_times)} pontos em alerta, {len(alarm_times)} alarmes reais")


if __name__ == "__main__":
    main()
