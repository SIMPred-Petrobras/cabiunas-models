#!/usr/bin/env python3
"""Simula suavizar o SENSOR DE REFERÊNCIA (rolling median) antes de classificar
o estado operacional, mantendo a lógica de máscara atual (endpoint) intacta —
hipótese: o flicker vem de ruído/jitter no sensor de status perto de falhas
(ex.: corrente oscilando 0↔100A a cada 1-2min), não de uma limitação da regra
de máscara em si. Suavizar a ENTRADA deveria estabilizar o estado sem afrouxar
o critério de aceitação (ainda exige state=="on" no fim da janela).

Compara, para os mesmos 6 casos de scan_mask_erased_precursors.py:
  - recovered_near: pontos de anomalia recuperados na janela perto da falha
  - rate_sim: taxa de anomalia/dia global (deve ficar estável ou cair, não
    explodir como no teste de "máscara majoritária" sem suavização)
em janelas de suavização candidatas (0=atual, 5, 15, 30 min).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.transpetro.io import _is_long_format, _pivot_long
from src.cnn1d_ae.scoring import (
    build_operational_state, mask_anomaly_seq_by_operational_state, map_seq_to_point_anomalies,
)
from src.cnn1d_ae.pipeline import parse_failure_dates

FEATHER_BASE = Path("/home/dvar/transpetro/PROJETO")
SMOOTH_CANDIDATES = [0, 5, 15, 30]

CASES = [
    ("uni", "B-3403C"), ("uni", "B-4064A"), ("mult", "B-4064A"),
    ("uni", "B-24001B"), ("uni", "B-402E"), ("mult", "B-8801C"),
]


def root_for(mode: str) -> Path:
    base = "Uni_sensor" if mode == "uni" else "Mult_sensor"
    return Path(f"resultados/experimento_2_supressao_transiente/{base}")


def load_raw_series(feather_path: str, sensor: str) -> pd.Series:
    df = pd.read_feather(FEATHER_BASE / feather_path)
    if _is_long_format(df):
        df = _pivot_long(df)
    if not isinstance(df.index, pd.DatetimeIndex):
        tcol = next(c for c in ("Timestamp", "Data Hora", "data_datetime") if c in df.columns)
        df = df.set_index(pd.to_datetime(df[tcol], errors="coerce"))
    df = df.sort_index()
    return pd.to_numeric(df[sensor], errors="coerce")


def run_case(mode: str, eq: str) -> dict:
    eq_dir = root_for(mode) / eq
    calib = json.loads((eq_dir / "csv" / "calibration_report.json").read_text(encoding="utf-8"))
    cfg = json.loads((eq_dir / "csv" / "run_config.json").read_text(encoding="utf-8"))
    thr = float(calib["threshold"])
    time_steps = int(cfg["TIME_STEPS"])
    point_rule, point_window, point_min_count = cfg["POINT_RULE"], int(cfg["POINT_WINDOW"]), int(cfg["POINT_MIN_COUNT"])
    fails = parse_failure_dates(cfg.get("FAILURE_DATE", ""))
    ref_sensor = cfg["OPERATIONAL_REF_SENSOR"]

    seq = pd.read_csv(eq_dir / "csv" / "sequence_scores_all.csv")
    seq["t"] = pd.to_datetime(seq["seq_start_time"], errors="coerce")
    seq = seq.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    anomaly_seq_raw = (seq["mae_seq"].values > thr)

    pt = pd.read_csv(eq_dir / "csv" / "point_anomalies_all.csv")
    pt["t"] = pd.to_datetime(pt.iloc[:, 0], errors="coerce")
    pt = pt.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    pt_idx = pd.DatetimeIndex(pt["t"])
    n_days = max(1e-9, (pt_idx.max() - pt_idx.min()).total_seconds() / 86400.0)

    near_mask = np.zeros(len(seq), dtype=bool)
    for f in fails:
        near_mask |= (seq["t"] >= f - pd.Timedelta(days=10)) & (seq["t"] <= f + pd.Timedelta(days=2))
    seq_end_pos = np.arange(time_steps - 1, time_steps - 1 + len(seq))
    valid = seq_end_pos < len(pt_idx)
    seq_end_times = pt_idx[seq_end_pos[valid]]
    near_seq_end_times = seq_end_times[np.isin(seq_end_pos[valid], np.where(near_mask)[0])]

    raw_ref = load_raw_series(cfg["FEATHER_PATH"], ref_sensor)

    row = {"mode": mode, "equip": eq, "rate_atual": round(float(calib["anomaly_rate_points_per_day"]), 2)}
    for sm in SMOOTH_CANDIDATES:
        ref = raw_ref.rolling(f"{sm}min", min_periods=1).median() if sm > 0 else raw_ref
        state = build_operational_state(
            index=pt_idx, sensor_series=ref,
            off_value_quantile=cfg["OFF_VALUE_QUANTILE"], off_abs_threshold=cfg.get("OFF_ABS_THRESHOLD"),
            off_long_min_hours=cfg["OFF_LONG_MIN_HOURS"], transient_padding_minutes=cfg["TRANSIENT_PADDING_MINUTES"],
            transient_diff_quantile=cfg["TRANSIENT_DIFF_QUANTILE"],
        )
        anomaly_seq_masked = mask_anomaly_seq_by_operational_state(anomaly_seq_raw, pt_idx, time_steps, state)
        df_point_sim = map_seq_to_point_anomalies(
            anomaly_seq_masked, pt_idx, time_steps, point_rule, point_window, point_min_count)
        recovered_near = int(df_point_sim.reindex(near_seq_end_times)["is_anom_point"].fillna(0).sum())
        rate_sim = float(df_point_sim["is_anom_point"].sum()) / n_days
        row[f"sm{sm}_recuperados_near"] = recovered_near
        row[f"sm{sm}_rate_sim"] = round(rate_sim, 2)
    return row


def main() -> None:
    rows = [run_case(mode, eq) for mode, eq in CASES]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = Path("analysis/SIMULACAO_MASCARA_SUAVIZADA.csv")
    df.to_csv(out, index=False)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
