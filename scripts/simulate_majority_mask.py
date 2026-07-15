#!/usr/bin/env python3
"""Simula a máscara operacional com "estado majoritário da janela" (fração de
pontos on+transiente dentro dos TIME_STEPS minutos que terminam em cada
sequência) em vez de "estado no minuto final", para vários percentuais de
corte candidatos — sem re-treinar nada, só reprocessando mae_seq já salvo.

Para cada candidato P, mede:
  - recovered: quantos pontos de anomalia aparecem na janela que hoje está
    apagada (a mesma identificada por scan_mask_erased_precursors.py)
  - rate_before / rate_after: taxa de anomalia/dia global (todo o período),
    pra ver se o candidato solta ruído demais em geral, não só perto da falha.

Não decide a regra sozinho — gera a tabela pra comparar antes de implementar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cnn1d_ae.scoring import map_seq_to_point_anomalies
from src.cnn1d_ae.pipeline import parse_failure_dates

CANDIDATES = [0.3, 0.5, 0.7]

CASES = [
    ("uni", "B-3403C"), ("uni", "B-4064A"), ("mult", "B-4064A"),
    ("uni", "B-24001B"), ("uni", "B-402E"), ("mult", "B-8801C"),
]


def root_for(mode: str) -> Path:
    base = "Uni_sensor" if mode == "uni" else "Mult_sensor"
    return Path(f"resultados/experimento_2_supressao_transiente/{base}")


def run_case(mode: str, eq: str) -> dict:
    eq_dir = root_for(mode) / eq
    seq = pd.read_csv(eq_dir / "csv" / "sequence_scores_all.csv")
    seq["t"] = pd.to_datetime(seq["seq_start_time"], errors="coerce")
    seq = seq.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    pt = pd.read_csv(eq_dir / "csv" / "point_anomalies_all.csv")
    pt["t"] = pd.to_datetime(pt.iloc[:, 0], errors="coerce")
    pt = pt.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    pt_idx = pd.DatetimeIndex(pt["t"])
    state = pd.Series(pt["operational_state"].values, index=pt_idx) if "operational_state" in pt.columns else None

    calib = json.loads((eq_dir / "csv" / "calibration_report.json").read_text(encoding="utf-8"))
    cfg = json.loads((eq_dir / "csv" / "run_config.json").read_text(encoding="utf-8"))
    thr = float(calib["threshold"])
    time_steps = int(cfg["TIME_STEPS"])
    point_rule, point_window, point_min_count = cfg["POINT_RULE"], int(cfg["POINT_WINDOW"]), int(cfg["POINT_MIN_COUNT"])
    fails = parse_failure_dates(cfg.get("FAILURE_DATE", ""))
    n_days = max(1e-9, (pt_idx.max() - pt_idx.min()).total_seconds() / 86400.0)

    anomaly_seq_raw = (seq["mae_seq"].values > thr)
    baseline_rate = float(calib.get("anomaly_rate_points_per_day"))

    # janela "apagada" perto da falha (mesma lógica do scan: -10d/+2d)
    near_mask = np.zeros(len(seq), dtype=bool)
    for f in fails:
        near_mask |= (seq["t"] >= f - pd.Timedelta(days=10)) & (seq["t"] <= f + pd.Timedelta(days=2))

    present = state.isin(["on", "transiente"]).astype(float) if state is not None else pd.Series(1.0, index=pt_idx)
    frac_present = present.rolling(window=time_steps, min_periods=time_steps).mean()

    seq_end_pos = np.arange(time_steps - 1, time_steps - 1 + len(seq))
    valid = seq_end_pos < len(pt_idx)
    seq_end_times = pt_idx[seq_end_pos[valid]]
    frac_at_seq_end = frac_present.reindex(seq_end_times).fillna(0.0).values

    row = {"mode": mode, "equip": eq, "rate_atual": round(baseline_rate, 2)}
    for p in CANDIDATES:
        allowed = np.zeros(len(seq), dtype=bool)
        allowed[valid] = frac_at_seq_end >= p
        anomaly_seq_sim = anomaly_seq_raw & allowed

        df_point_sim = map_seq_to_point_anomalies(
            anomaly_seq_sim, pt_idx, time_steps, point_rule, point_window, point_min_count)
        # pontos recuperados dentro da janela perto da falha (mesmos timestamps de fim de sequência)
        near_seq_end_times = seq_end_times[np.isin(seq_end_pos[valid], np.where(near_mask)[0])]
        recovered_near = int(df_point_sim.reindex(near_seq_end_times)["is_anom_point"].fillna(0).sum())

        rate_sim = float(df_point_sim["is_anom_point"].sum()) / n_days
        row[f"p{int(p*100)}_recuperados_near"] = recovered_near
        row[f"p{int(p*100)}_rate_sim"] = round(rate_sim, 2)
    return row


def main() -> None:
    rows = [run_case(mode, eq) for mode, eq in CASES]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = Path("analysis/SIMULACAO_MASCARA_MAJORITARIA.csv")
    df.to_csv(out, index=False)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
