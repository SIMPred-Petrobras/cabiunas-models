#!/usr/bin/env python3
"""Simula uma mudança cirúrgica na máscara: aceitar `transiente` como válido
(não só `on`) no endpoint da janela — mantém off_curto/off_longo excluídos
exatamente como hoje, e mantém a checagem em UM PONTO só (não janela
majoritária, que já provou explodir ruído). Hipótese, motivada pelo caso
B-3403C: o MAE alvo sobe justamente durante a RAMPA de desligamento (corrente
caindo de ~108A pra 0), que a máscara classifica como "transiente" e descarta
— mas isso pode ser o próprio evento de falha/parada, não ruído.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cnn1d_ae.scoring import map_seq_to_point_anomalies
from src.cnn1d_ae.pipeline import parse_failure_dates

CASES = [
    ("uni", "B-3403C"), ("uni", "B-4064A"), ("mult", "B-4064A"),
    ("uni", "B-24001B"), ("uni", "B-402E"), ("mult", "B-8801C"),
]


def root_for(mode: str) -> Path:
    base = "Uni_sensor" if mode == "uni" else "Mult_sensor"
    return Path(f"resultados/experimento_2_supressao_transiente/{base}")


def mask_allow_transiente(anomaly_seq, index, time_steps, state):
    seq_end_pos = np.arange(time_steps - 1, time_steps - 1 + len(anomaly_seq))
    valid = seq_end_pos < len(index)
    out = anomaly_seq.astype(bool).copy()
    if not valid.any():
        return out
    seq_end_idx = index[seq_end_pos[valid]]
    allowed = state.reindex(seq_end_idx).fillna("on").isin(["on", "transiente"]).values
    out_valid = out[valid] & allowed
    out[valid] = out_valid
    return out


def run_case(mode: str, eq: str) -> dict:
    eq_dir = root_for(mode) / eq
    calib = json.loads((eq_dir / "csv" / "calibration_report.json").read_text(encoding="utf-8"))
    cfg = json.loads((eq_dir / "csv" / "run_config.json").read_text(encoding="utf-8"))
    thr = float(calib["threshold"])
    time_steps = int(cfg["TIME_STEPS"])
    point_rule, point_window, point_min_count = cfg["POINT_RULE"], int(cfg["POINT_WINDOW"]), int(cfg["POINT_MIN_COUNT"])
    fails = parse_failure_dates(cfg.get("FAILURE_DATE", ""))

    seq = pd.read_csv(eq_dir / "csv" / "sequence_scores_all.csv")
    seq["t"] = pd.to_datetime(seq["seq_start_time"], errors="coerce")
    seq = seq.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    anomaly_seq_raw = (seq["mae_seq"].values > thr)

    pt = pd.read_csv(eq_dir / "csv" / "point_anomalies_all.csv")
    pt["t"] = pd.to_datetime(pt.iloc[:, 0], errors="coerce")
    pt = pt.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    pt_idx = pd.DatetimeIndex(pt["t"])
    state = pd.Series(pt["operational_state"].values, index=pt_idx)
    n_days = max(1e-9, (pt_idx.max() - pt_idx.min()).total_seconds() / 86400.0)

    # janelas de avaliação: near_48h (±2d, o que conta pra classe BOM) e near_10d (2-10d antes)
    near48_mask = np.zeros(len(seq), dtype=bool)
    near10_mask = np.zeros(len(seq), dtype=bool)
    for f in fails:
        near48_mask |= (seq["t"] >= f - pd.Timedelta(days=2)) & (seq["t"] <= f + pd.Timedelta(days=2))
        near10_mask |= (seq["t"] >= f - pd.Timedelta(days=10)) & (seq["t"] < f - pd.Timedelta(days=2))
    seq_end_pos = np.arange(time_steps - 1, time_steps - 1 + len(seq))
    valid = seq_end_pos < len(pt_idx)
    seq_end_times = pt_idx[seq_end_pos[valid]]
    t48 = seq_end_times[np.isin(seq_end_pos[valid], np.where(near48_mask)[0])]
    t10 = seq_end_times[np.isin(seq_end_pos[valid], np.where(near10_mask)[0])]

    def eval_point_df(df_point, allow_transiente: bool):
        # replica o SEGUNDO filtro (nível de ponto) que pipeline.py aplica após
        # map_seq_to_point_anomalies — precisa respeitar o mesmo allow_transiente,
        # senão a simulação fica mais otimista que a produção (bug já corrigido).
        st_pt = state.reindex(df_point.index).fillna("on")
        ok = ["on", "transiente"] if allow_transiente else ["on"]
        df_point = df_point.copy()
        df_point.loc[~st_pt.isin(ok), "is_anom_point"] = 0
        return (int(df_point.reindex(t48)["is_anom_point"].fillna(0).sum()),
                int(df_point.reindex(t10)["is_anom_point"].fillna(0).sum()),
                float(df_point["is_anom_point"].sum()) / n_days)

    # baseline (atual, endpoint == "on" só)
    anomaly_seq_atual = anomaly_seq_raw.copy()
    seq_end_pos2 = np.arange(time_steps - 1, time_steps - 1 + len(anomaly_seq_raw))
    valid2 = seq_end_pos2 < len(pt_idx)
    allowed_atual = np.ones(len(anomaly_seq_raw), dtype=bool)
    allowed_atual[valid2] = state.reindex(pt_idx[seq_end_pos2[valid2]]).fillna("on").eq("on").values
    anomaly_seq_atual = anomaly_seq_raw & allowed_atual
    df_atual = map_seq_to_point_anomalies(anomaly_seq_atual, pt_idx, time_steps, point_rule, point_window, point_min_count)
    hit48_atual, hit10_atual, rate_atual = eval_point_df(df_atual, allow_transiente=False)

    # candidato: aceita "transiente" também (nos dois níveis, seq e ponto)
    anomaly_seq_cand = mask_allow_transiente(anomaly_seq_raw, pt_idx, time_steps, state)
    df_cand = map_seq_to_point_anomalies(anomaly_seq_cand, pt_idx, time_steps, point_rule, point_window, point_min_count)
    hit48_cand, hit10_cand, rate_cand = eval_point_df(df_cand, allow_transiente=True)

    return {
        "mode": mode, "equip": eq,
        "hit48_atual": hit48_atual, "hit48_cand": hit48_cand,
        "hit10_atual": hit10_atual, "hit10_cand": hit10_cand,
        "rate_atual": round(rate_atual, 2), "rate_cand": round(rate_cand, 2),
    }


def main() -> None:
    rows = [run_case(mode, eq) for mode, eq in CASES]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = Path("analysis/SIMULACAO_MASCARA_ALLOW_TRANSIENTE.csv")
    df.to_csv(out, index=False)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
