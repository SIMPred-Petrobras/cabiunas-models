#!/usr/bin/env python3
"""
eval_by_condition.py
Recall por tipo de condição de alarme (HI, HIHI, UNDER, LOLO) por sensor.

Responde: o modelo é mais cego para temperatura baixa (UNDER) ou alta (HI/HIHI)?
E para vibração (LOLO)?

Uso:
    PYTHONPATH=. python scripts/eval_by_condition.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7 \
        --eval_start 2025-01-01 --eval_end 2025-12-31 \
        --label inSample_2025
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from clearml import Task

SENSORS = [
    "T5_AVG_A",
    "TC382_01_A", "TC382_02_A", "TC382_03_A",
    "TC382_04_A", "TC382_05_A", "TC382_06_A",
    "TV_351X_A",  "TV_351Y_A",
    "TV_352X_A",  "TV_352Y_A",
    "TV_353X_A",  "TV_353Y_A",
    "TV_354X_A",  "TV_354Y_A",
    "TV_355X_A",  "TV_355Y_A",
]

ALARM_CSV_DEFAULT = "../dados/alarmes_selecionados_turbina_a.csv"
SAMPLING_INTERVAL = "5min"
DEBOUNCE_HOURS    = 8.0
HORIZON_HOURS     = 8.0
GAP_HOURS         = 4.0
CONDITIONS        = ["HI", "HIHI", "UNDER", "LOLO", "OVER", "CFN"]


def load_mae_series(task: Task) -> Dict[str, pd.Series]:
    arts = task.artifacts
    series: Dict[str, pd.Series] = {}
    for sensor in SENSORS:
        key = next((k for k in arts if "sequence_scores_all" in k and sensor in k), None)
        if key is None:
            continue
        path = arts[key].get_local_copy()
        df = pd.read_csv(path)
        df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
        series[sensor] = df.set_index("seq_start_time")["mae_seq"]
    return series


def ewma_quantile(mae: pd.Series, half_life_hours: float = 4.0) -> pd.Series:
    hl_pts = int(round(pd.Timedelta(hours=half_life_hours) / pd.Timedelta(SAMPLING_INTERVAL)))
    return mae.ewm(halflife=max(1, hl_pts)).mean().rank(pct=True)


def load_alarms_by_condition(alarm_csv: str,
                              t0=None, t1=None) -> Dict[str, Dict[str, List[pd.Timestamp]]]:
    """Retorna {sensor: {condição: [timestamps onset]}}."""
    df = pd.read_csv(alarm_csv)
    date_col = next(
        (c for c in df.columns if "ocorr" in c.lower()
         or ("data" in c.lower() and "ação" not in c.lower())),
        df.columns[0],
    )
    cond_col = next((c for c in df.columns if "condi" in c.lower()), None)
    tag_col  = next((c for c in df.columns if "tag" in c.lower()
                     and "alarm" in c.lower()), None)

    df = df.copy()
    df["_time"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=["_time"])

    # Exclui OKs
    if cond_col:
        df = df[df[cond_col].str.upper().fillna("").ne("OK")]

    # Filtro de período
    if t0: df = df[df["_time"] >= t0]
    if t1: df = df[df["_time"] <= t1]

    result: Dict[str, Dict[str, List[pd.Timestamp]]] = {}
    for sensor in SENSORS:
        sensor_df = df[df[tag_col] == sensor] if tag_col else pd.DataFrame()
        result[sensor] = {}
        if cond_col:
            for cond in CONDITIONS:
                rows = sensor_df[sensor_df[cond_col].str.upper() == cond]
                if len(rows):
                    result[sensor][cond] = sorted(
                        pd.to_datetime(rows["_time"], utc=True).tolist()
                    )
        # "ALL" = todos os onsets juntos
        result[sensor]["ALL"] = sorted(
            pd.to_datetime(sensor_df["_time"], utc=True).tolist()
        ) if len(sensor_df) else []
    return result


def cluster_incidents(times: List[pd.Timestamp]) -> List[pd.Timestamp]:
    if not times:
        return []
    s = pd.Series(times).sort_values().reset_index(drop=True)
    g = (s.diff().dt.total_seconds() / 3600 > GAP_HOURS).cumsum()
    return s.groupby(g).first().tolist()


def detect_episodes_gap(alert: pd.Series) -> List[Tuple]:
    debounce = pd.Timedelta(hours=DEBOUNCE_HOURS)
    on_idx = alert.index[alert]
    if not len(on_idx):
        return []
    episodes, cs, ce = [], on_idx[0], on_idx[0]
    for t in on_idx[1:]:
        if (t - ce) <= debounce:
            ce = t
        else:
            episodes.append((cs, ce)); cs = ce = t
    episodes.append((cs, ce))
    return episodes


def evaluate(health: pd.Series, incidents: List[pd.Timestamp],
             horizon_hours: float, fa_budget: float = 1.0) -> dict:
    if not incidents:
        return {"n_incidents": 0, "recall": float("nan"), "fa_per_day": float("nan")}

    horizon_sec = horizon_hours * 3600.0
    total_days  = (health.index[-1] - health.index[0]).total_seconds() / 86400.0
    inc_s       = np.array([t.timestamp() for t in incidents])

    best_rec, best_fa = 0.0, float("inf")

    for q in np.linspace(0.50, 0.999, 80):
        alert   = health >= q
        alert_s = np.array([t.timestamp() for t in health.index[alert]])
        eps     = detect_episodes_gap(alert)

        n_hit = sum(
            1 for ti in inc_s
            if alert_s.size and np.any(
                (alert_s >= ti - horizon_sec) & (alert_s <= ti)
            )
        )
        n_fp = sum(
            1 for (s0, s1) in eps
            if not (np.any(
                (inc_s - horizon_sec <= s1.timestamp()) & (inc_s >= s0.timestamp())
            ) if inc_s.size else False)
        )
        fa  = n_fp / max(total_days, 1.0)
        rec = n_hit / len(incidents)
        if fa <= fa_budget and rec > best_rec:
            best_rec, best_fa = rec, fa

    return {
        "n_incidents": len(incidents),
        "recall":      best_rec,
        "fa_per_day":  best_fa,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id",    required=True)
    parser.add_argument("--label",      default="eval")
    parser.add_argument("--alarm_csv",  default=ALARM_CSV_DEFAULT)
    parser.add_argument("--half_life",  type=float, default=4.0)
    parser.add_argument("--horizon",    type=float, default=HORIZON_HOURS)
    parser.add_argument("--fa_budget",  type=float, default=1.0)
    parser.add_argument("--eval_start", default=None)
    parser.add_argument("--eval_end",   default=None)
    parser.add_argument("--out_dir",    default="eval_predictive_out/by_condition")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = pd.Timestamp(args.eval_start, tz="UTC") if args.eval_start else None
    t1 = pd.Timestamp(args.eval_end,   tz="UTC") if args.eval_end   else None

    print(f"\nCarregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"  {task.name} | {task.get_status()}")

    print("Baixando sequence_scores...")
    mae_dict = load_mae_series(task)

    print(f"EWMA (hl={args.half_life}h) + quantile...")
    health_dict = {s: ewma_quantile(mae, args.half_life) for s, mae in mae_dict.items()}

    print("Carregando alarmes por condição...")
    alarms_by_cond = load_alarms_by_condition(args.alarm_csv, t0, t1)

    rows = []
    for sensor, health in health_dict.items():
        h = health.copy()
        if t0: h = h[h.index >= t0]
        if t1: h = h[h.index <= t1]
        if h.empty:
            continue

        cond_map = alarms_by_cond.get(sensor, {})
        for cond, alarm_times in cond_map.items():
            incidents = cluster_incidents(alarm_times)
            if not incidents:
                continue
            res = evaluate(h, incidents, args.horizon, args.fa_budget)
            rows.append({
                "sensor":    sensor,
                "condition": cond,
                **res,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("Nenhum dado para reportar.")
        return

    # Pivot: sensores × condições
    pivot_rec = df.pivot_table(
        index="sensor", columns="condition",
        values="recall", aggfunc="first"
    ).round(3)

    pivot_n = df.pivot_table(
        index="sensor", columns="condition",
        values="n_incidents", aggfunc="first"
    ).fillna(0).astype(int)

    print(f"\n=== RECALL POR CONDIÇÃO (H={args.horizon}h, {args.label}) ===")
    print("\nRecall:")
    print(pivot_rec.to_string())
    print("\nNúmero de incidentes:")
    print(pivot_n.to_string())

    # Análise agregada por condição
    print(f"\n=== RESUMO POR TIPO DE CONDIÇÃO ===")
    summary = df[df["condition"] != "ALL"].groupby("condition").agg(
        n_sensors    =("sensor", "count"),
        total_inc    =("n_incidents", "sum"),
        avg_recall   =("recall", "mean"),
        min_recall   =("recall", "min"),
    ).round(3)
    print(summary.to_string())

    # Salva
    out_csv = os.path.join(args.out_dir, f"by_condition_{args.label}.csv")
    df.to_csv(out_csv, index=False)
    pivot_rec.to_csv(os.path.join(args.out_dir, f"recall_pivot_{args.label}.csv"))
    print(f"\nSalvo: {out_csv}")


if __name__ == "__main__":
    main()
