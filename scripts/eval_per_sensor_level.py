#!/usr/bin/env python3
"""
eval_per_sensor_level.py
Avalia cada sensor individualmente contra seus próprios alarmes (Tag Alarme == sensor).
Suporta filtro de período para separar avaliação in-sample (2025) de OOS (2026).

Uso:
    # In-sample 2025
    PYTHONPATH=. python scripts/eval_per_sensor_level.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7 \
        --eval_start 2025-01-01 --eval_end 2025-12-31 \
        --label inSample_2025

    # OOS 2026
    PYTHONPATH=. python scripts/eval_per_sensor_level.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7 \
        --eval_start 2026-01-01 --eval_end 2026-04-30 \
        --label OOS_2026
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
SAMPLING_INTERVAL = "5min"   # STRIDE=10 × 30s
DEBOUNCE_HOURS    = 8.0
HORIZON_HOURS     = 8.0
GAP_HOURS         = 4.0


# ---------------------------------------------------------------------------
# Carrega sequence_scores de cada sensor
# ---------------------------------------------------------------------------

def load_mae_series(task: Task, sensors: List[str]) -> Dict[str, pd.Series]:
    arts = task.artifacts
    series: Dict[str, pd.Series] = {}
    for sensor in sensors:
        key = next(
            (k for k in arts if "sequence_scores_all" in k and sensor in k), None
        )
        if key is None:
            print(f"  [WARN] {sensor}: artifact não encontrado — ignorado")
            continue
        path = arts[key].get_local_copy()
        df = pd.read_csv(path)
        df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
        s = df.set_index("seq_start_time")["mae_seq"]
        series[sensor] = s
    print(f"  {len(series)}/{len(sensors)} sensores carregados")
    return series


# ---------------------------------------------------------------------------
# EWMA + normalização por quantile (mesmo que eval_predictive_layer.py)
# ---------------------------------------------------------------------------

def ewma_quantile(mae: pd.Series, half_life_hours: float) -> pd.Series:
    hl_pts = int(round(pd.Timedelta(hours=half_life_hours) / pd.Timedelta(SAMPLING_INTERVAL)))
    health = mae.ewm(halflife=max(1, hl_pts)).mean()
    return health.rank(pct=True)


# ---------------------------------------------------------------------------
# Carrega alarmes por sensor
# ---------------------------------------------------------------------------

def load_alarms_per_sensor(alarm_csv: str) -> Dict[str, List[pd.Timestamp]]:
    """Retorna dict sensor → lista de timestamps de onset alarms (sem OK)."""
    df = pd.read_csv(alarm_csv)

    date_col = next(
        (c for c in df.columns if "ocorr" in c.lower() or ("data" in c.lower() and "ação" not in c.lower())),
        df.columns[0],
    )
    cond_col  = next((c for c in df.columns if "condi" in c.lower()), None)
    tag_col   = next((c for c in df.columns if "tag" in c.lower() and "alarm" in c.lower()), None)

    if tag_col is None:
        raise ValueError("Coluna 'Tag Alarme' não encontrada no CSV de alarmes.")

    df = df.copy()
    df["_time"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    if cond_col:
        mask = df[cond_col].str.upper().fillna("").ne("OK")
        df = df[mask]
    df = df.dropna(subset=["_time"])

    result: Dict[str, List[pd.Timestamp]] = {}
    for sensor in SENSORS:
        rows = df[df[tag_col] == sensor]
        result[sensor] = sorted(pd.to_datetime(rows["_time"], utc=True).tolist())

    return result


# ---------------------------------------------------------------------------
# Clustering de incidentes
# ---------------------------------------------------------------------------

def cluster_incidents(alarm_times: List[pd.Timestamp], gap_hours: float = GAP_HOURS) -> List[pd.Timestamp]:
    if not alarm_times:
        return []
    s = pd.Series(alarm_times, name="t").sort_values().reset_index(drop=True)
    g = (s.diff().dt.total_seconds() / 3600 > gap_hours).cumsum()
    return s.groupby(g).first().tolist()


# ---------------------------------------------------------------------------
# Gap-debounce de episódios de alerta
# ---------------------------------------------------------------------------

def detect_episodes_gap(alert: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    debounce = pd.Timedelta(hours=DEBOUNCE_HOURS)
    on_idx = alert.index[alert]
    if len(on_idx) == 0:
        return []
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_start = on_idx[0]
    cur_end   = on_idx[0]
    for t in on_idx[1:]:
        if (t - cur_end) <= debounce:
            cur_end = t
        else:
            episodes.append((cur_start, cur_end))
            cur_start = cur_end = t
    episodes.append((cur_start, cur_end))
    return episodes


# ---------------------------------------------------------------------------
# Avaliação recall × FA para threshold sweep
# ---------------------------------------------------------------------------

def best_point_for_sensor(
    health: pd.Series,
    incidents: List[pd.Timestamp],
    horizon_hours: float,
    n_thresholds: int = 100,
    fa_budget: float = 1.0,
) -> dict:
    horizon_sec = horizon_hours * 3600.0
    total_days  = (health.index[-1] - health.index[0]).total_seconds() / 86400.0
    inc_s       = np.array([t.timestamp() for t in incidents])

    best = {"recall": 0.0, "fa_per_day": 0.0, "threshold_q": 0.5, "n_incidents": len(incidents)}
    best_recall = -1.0

    for q in np.linspace(0.50, 0.999, n_thresholds):
        alert    = health >= q
        alert_s  = np.array([t.timestamp() for t in health.index[alert]])
        episodes = detect_episodes_gap(alert)

        n_hit = 0
        for ti in inc_s:
            if alert_s.size and np.any((alert_s >= ti - horizon_sec) & (alert_s <= ti)):
                n_hit += 1

        n_fp = 0
        for (s0, s1) in episodes:
            s0_ts = s0.timestamp()
            s1_ts = s1.timestamp()
            useful = bool(np.any((inc_s - horizon_sec <= s1_ts) & (inc_s >= s0_ts))) if inc_s.size else False
            if not useful:
                n_fp += 1

        fa_per_day = n_fp / max(total_days, 1.0)
        recall     = n_hit / len(incidents) if incidents else 0.0

        if fa_per_day <= fa_budget and recall > best_recall:
            best_recall = recall
            best = {
                "recall":       recall,
                "fa_per_day":   fa_per_day,
                "threshold_q":  float(q),
                "n_incidents":  len(incidents),
                "n_hit":        n_hit,
                "n_fp":         n_fp,
                "total_days":   total_days,
            }

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação por sensor da camada preditiva EWMA")
    parser.add_argument("--task_id",    required=True, help="ID da task ClearML v8")
    parser.add_argument("--label",      default="eval",  help="Sufixo para arquivos de saída")
    parser.add_argument("--alarm_csv",  default=ALARM_CSV_DEFAULT)
    parser.add_argument("--half_life",  type=float, default=4.0, help="Half-life EWMA em horas")
    parser.add_argument("--horizon",    type=float, default=HORIZON_HOURS, help="Horizonte de antecipação (h)")
    parser.add_argument("--fa_budget",  type=float, default=1.0, help="FA/dia máximo para ponto de operação")
    parser.add_argument("--eval_start", default=None, help="Início do período de avaliação (ex: 2025-01-01)")
    parser.add_argument("--eval_end",   default=None, help="Fim do período de avaliação (ex: 2025-12-31)")
    parser.add_argument("--out_dir",    default="eval_predictive_out/per_sensor_level")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[1/4] Carregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"      {task.name}  |  status={task.get_status()}")

    print(f"\n[2/4] Baixando sequence_scores de {len(SENSORS)} sensores...")
    mae_dict = load_mae_series(task, SENSORS)

    print(f"\n[3/4] EWMA (half_life={args.half_life}h) + normalização quantile...")
    health_dict = {s: ewma_quantile(mae, args.half_life) for s, mae in mae_dict.items()}

    # Filtro de período
    t0 = pd.Timestamp(args.eval_start, tz="UTC") if args.eval_start else None
    t1 = pd.Timestamp(args.eval_end,   tz="UTC") if args.eval_end   else None

    if t0 or t1:
        label_period = f"{t0.date() if t0 else 'início'} → {t1.date() if t1 else 'fim'}"
        print(f"      Período de avaliação: {label_period}")

    print(f"\n[4/4] Avaliando sensor a sensor (H={args.horizon}h)...")
    alarm_per_sensor = load_alarms_per_sensor(args.alarm_csv)

    rows = []
    for sensor, health in health_dict.items():
        # Filtra série temporal do sensor
        h = health.copy()
        if t0:
            h = h[h.index >= t0]
        if t1:
            h = h[h.index <= t1]

        if h.empty:
            print(f"  {sensor}: sem dados no período — ignorado")
            continue

        # Filtra alarmes do sensor para o período
        alarms_s = alarm_per_sensor.get(sensor, [])
        alarms_s = [
            a for a in alarms_s
            if (t0 is None or a >= t0) and (t1 is None or a <= t1)
        ]
        incidents = cluster_incidents(alarms_s)

        if not incidents:
            print(f"  {sensor}: 0 incidentes no período — FA/dia medido, recall=N/A")
            result = best_point_for_sensor(h, [], args.horizon, fa_budget=args.fa_budget)
            result["sensor"] = sensor
            rows.append(result)
            continue

        result = best_point_for_sensor(h, incidents, args.horizon, fa_budget=args.fa_budget)
        result["sensor"] = sensor
        rows.append(result)

        recall_str = f"{result['recall']:.1%}"
        print(f"  {sensor}: {len(incidents)} incidentes | rec={result['recall']:.2f} FA={result['fa_per_day']:.3f}")

    # Relatório
    df_out = pd.DataFrame(rows).set_index("sensor")
    col_order = ["n_incidents", "recall", "fa_per_day", "threshold_q", "n_hit", "n_fp", "total_days"]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]

    print(f"\n=== RESULTADO POR SENSOR (H={args.horizon}h, {args.label}) ===")
    print(f"{'Sensor':<18} {'Incidentes':>12} {'Recall':>8} {'FA/dia':>8}")
    print("─" * 50)
    for sensor, row in df_out.iterrows():
        n_inc = int(row.get("n_incidents", 0))
        rec   = row.get("recall", float("nan"))
        fa    = row.get("fa_per_day", float("nan"))
        if n_inc == 0:
            rec_str = "  N/A"
        else:
            rec_str = f"{rec:7.1%}"
        print(f"  {sensor:<16} {n_inc:>12}  {rec_str}  {fa:>8.3f}")

    out_csv = os.path.join(args.out_dir, f"per_sensor_eval_{args.label}.csv")
    df_out.to_csv(out_csv)
    print(f"\nSalvo: {out_csv}")


if __name__ == "__main__":
    main()
