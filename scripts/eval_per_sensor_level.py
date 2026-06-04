#!/usr/bin/env python3
"""
eval_per_sensor_level.py
Avalia cada sensor individualmente contra seus próprios alarmes.

Novidades:
  --ok_aware    : cluster por sensor usando OK como reset (HIHI→OK→HIHI = 2 incidentes)
  --sticky_hours: após alertar, mantém alerta ativo por N horas (sticky alert)

Uso:
    PYTHONPATH=. python scripts/eval_per_sensor_level.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7 \
        --eval_start 2025-01-01 --eval_end 2025-12-31 \
        --label inSample_2025 [--ok_aware] [--sticky_hours 8]
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


# ---------------------------------------------------------------------------
# Carregamento de sequence_scores
# ---------------------------------------------------------------------------

def load_mae_series(task: Task, sensors: List[str]) -> Dict[str, pd.Series]:
    arts = task.artifacts
    series: Dict[str, pd.Series] = {}
    for sensor in sensors:
        key = next((k for k in arts if "sequence_scores_all" in k and sensor in k), None)
        if key is None:
            print(f"  [WARN] {sensor}: artifact não encontrado — ignorado")
            continue
        path = arts[key].get_local_copy()
        df = pd.read_csv(path)
        df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
        series[sensor] = df.set_index("seq_start_time")["mae_seq"]
    print(f"  {len(series)}/{len(sensors)} sensores carregados")
    return series


# ---------------------------------------------------------------------------
# EWMA + quantile normalization
# ---------------------------------------------------------------------------

def ewma_quantile(mae: pd.Series, half_life_hours: float) -> pd.Series:
    hl_pts = int(round(pd.Timedelta(hours=half_life_hours) / pd.Timedelta(SAMPLING_INTERVAL)))
    return mae.ewm(halflife=max(1, hl_pts)).mean().rank(pct=True)


# ---------------------------------------------------------------------------
# Sticky alert: mantém alerta ativo por sticky_hours após último disparo
# ---------------------------------------------------------------------------

def apply_sticky(health: pd.Series, threshold_q: float,
                 sticky_hours: float) -> pd.Series:
    """Retorna série booleana com sticky alert aplicado."""
    alert = (health >= threshold_q).copy()
    if sticky_hours <= 0 or not alert.any():
        return alert
    sticky_td  = pd.Timedelta(hours=sticky_hours)
    idx        = alert.index
    result     = alert.values.copy()
    alert_pos  = np.where(result)[0]
    for pos in alert_pos:
        end_t = idx[pos] + sticky_td
        end_pos = idx.searchsorted(end_t, side="right")
        result[pos:end_pos] = True
    return pd.Series(result, index=idx)


# ---------------------------------------------------------------------------
# Carregamento de alarmes — gap-based e OK-aware
# ---------------------------------------------------------------------------

def _parse_alarm_df(alarm_csv: str) -> Tuple[pd.DataFrame, str, str, str]:
    df = pd.read_csv(alarm_csv)
    date_col = next(
        (c for c in df.columns if "ocorr" in c.lower()
         or ("data" in c.lower() and "ação" not in c.lower())),
        df.columns[0],
    )
    cond_col = next((c for c in df.columns if "condi" in c.lower()), None)
    tag_col  = next((c for c in df.columns if "tag" in c.lower()
                     and "alarm" in c.lower()), None)
    if tag_col is None:
        raise ValueError("Coluna 'Tag Alarme' não encontrada.")
    df = df.copy()
    df["_time"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=["_time"]).sort_values("_time")
    return df, date_col, cond_col, tag_col


def load_alarms_gap(alarm_csv: str) -> Dict[str, List[pd.Timestamp]]:
    """Onset alarms por sensor (exclui OK). Clustering feito depois por gap."""
    df, _, cond_col, tag_col = _parse_alarm_df(alarm_csv)
    if cond_col:
        df = df[df[cond_col].str.upper().fillna("").ne("OK")]
    result: Dict[str, List[pd.Timestamp]] = {}
    for sensor in SENSORS:
        rows = df[df[tag_col] == sensor]
        result[sensor] = sorted(pd.to_datetime(rows["_time"], utc=True).tolist())
    return result


def load_alarms_ok_aware(alarm_csv: str) -> Dict[str, List[pd.Timestamp]]:
    """OK-aware: qualquer onset após um OK = novo incidente (reset pelo OK).
    Não usa gap — o próprio OK define o fim do evento."""
    df, _, cond_col, tag_col = _parse_alarm_df(alarm_csv)
    result: Dict[str, List[pd.Timestamp]] = {}
    for sensor in SENSORS:
        sensor_df = df[df[tag_col] == sensor].sort_values("_time")
        incidents: List[pd.Timestamp] = []
        in_incident = False
        for _, row in sensor_df.iterrows():
            cond = str(row[cond_col]).upper() if cond_col else "CFN"
            if cond == "OK":
                in_incident = False
            else:
                if not in_incident:
                    incidents.append(row["_time"])
                    in_incident = True
        result[sensor] = incidents
    return result


# ---------------------------------------------------------------------------
# Clustering gap-based (usado apenas no modo padrão)
# ---------------------------------------------------------------------------

def cluster_incidents(alarm_times: List[pd.Timestamp],
                      gap_hours: float = GAP_HOURS) -> List[pd.Timestamp]:
    if not alarm_times:
        return []
    s = pd.Series(alarm_times).sort_values().reset_index(drop=True)
    g = (s.diff().dt.total_seconds() / 3600 > gap_hours).cumsum()
    return s.groupby(g).first().tolist()


# ---------------------------------------------------------------------------
# Gap-debounce de episódios de alerta (FA)
# ---------------------------------------------------------------------------

def detect_episodes_gap(alert: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    debounce = pd.Timedelta(hours=DEBOUNCE_HOURS)
    on_idx = alert.index[alert]
    if len(on_idx) == 0:
        return []
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cs, ce = on_idx[0], on_idx[0]
    for t in on_idx[1:]:
        if (t - ce) <= debounce:
            ce = t
        else:
            episodes.append((cs, ce))
            cs = ce = t
    episodes.append((cs, ce))
    return episodes


# ---------------------------------------------------------------------------
# Avaliação: recall × FA com sweep de threshold
# ---------------------------------------------------------------------------

def best_point_for_sensor(
    health: pd.Series,
    incidents: List[pd.Timestamp],
    horizon_hours: float,
    sticky_hours: float = 0.0,
    n_thresholds: int = 100,
    fa_budget: float = 1.0,
) -> dict:
    horizon_sec = horizon_hours * 3600.0
    total_days  = (health.index[-1] - health.index[0]).total_seconds() / 86400.0
    inc_s       = np.array([t.timestamp() for t in incidents])

    best = {"recall": 0.0, "fa_per_day": 0.0, "threshold_q": 0.5,
            "n_incidents": len(incidents)}
    best_recall = -1.0

    for q in np.linspace(0.50, 0.999, n_thresholds):
        alert    = apply_sticky(health, q, sticky_hours)
        alert_s  = np.array([t.timestamp() for t in health.index[alert]])
        episodes = detect_episodes_gap(alert)

        n_hit = sum(
            1 for ti in inc_s
            if alert_s.size and np.any(
                (alert_s >= ti - horizon_sec) & (alert_s <= ti)
            )
        )
        n_fp = sum(
            1 for (s0, s1) in episodes
            if not (np.any(
                (inc_s - horizon_sec <= s1.timestamp()) &
                (inc_s >= s0.timestamp())
            ) if inc_s.size else False)
        )

        fa    = n_fp / max(total_days, 1.0)
        recall = n_hit / len(incidents) if incidents else 0.0

        if fa <= fa_budget and recall > best_recall:
            best_recall = recall
            best = {
                "recall":      recall,
                "fa_per_day":  fa,
                "threshold_q": float(q),
                "n_incidents": len(incidents),
                "n_hit":       n_hit,
                "n_fp":        n_fp,
                "total_days":  total_days,
            }
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id",      required=True)
    parser.add_argument("--label",        default="eval")
    parser.add_argument("--alarm_csv",    default=ALARM_CSV_DEFAULT)
    parser.add_argument("--half_life",    type=float, default=4.0)
    parser.add_argument("--horizon",      type=float, default=HORIZON_HOURS)
    parser.add_argument("--fa_budget",    type=float, default=1.0)
    parser.add_argument("--eval_start",   default=None)
    parser.add_argument("--eval_end",     default=None)
    parser.add_argument("--ok_aware",     action="store_true",
                        help="Usa OK como reset de incidente (HIHI→OK→HIHI = 2 inc)")
    parser.add_argument("--sticky_hours", type=float, default=0.0,
                        help="Horas que o alerta fica ativo após o último disparo")
    parser.add_argument("--out_dir",      default="eval_predictive_out/per_sensor_level")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = pd.Timestamp(args.eval_start, tz="UTC") if args.eval_start else None
    t1 = pd.Timestamp(args.eval_end,   tz="UTC") if args.eval_end   else None

    print(f"\n[1/4] Carregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"      {task.name}  |  status={task.get_status()}")
    if args.ok_aware:
        print("      Modo: OK-aware clustering")
    if args.sticky_hours > 0:
        print(f"      Sticky alert: {args.sticky_hours}h")

    print(f"\n[2/4] Baixando sequence_scores...")
    mae_dict = load_mae_series(task, SENSORS)

    print(f"\n[3/4] EWMA (hl={args.half_life}h) + quantile normalization...")
    health_dict = {s: ewma_quantile(mae, args.half_life) for s, mae in mae_dict.items()}

    if t0 or t1:
        print(f"      Período: {t0.date() if t0 else 'início'} → {t1.date() if t1 else 'fim'}")

    print(f"\n[4/4] Avaliando sensor a sensor (H={args.horizon}h)...")
    if args.ok_aware:
        raw_alarms = load_alarms_ok_aware(args.alarm_csv)
        # OK-aware já retorna incidentes diretamente
        def get_incidents(sensor, alarms_s):
            return alarms_s  # já clusterizado por OK
    else:
        raw_alarms = load_alarms_gap(args.alarm_csv)
        def get_incidents(sensor, alarms_s):
            return cluster_incidents(alarms_s)

    rows = []
    for sensor, health in health_dict.items():
        h = health.copy()
        if t0: h = h[h.index >= t0]
        if t1: h = h[h.index <= t1]
        if h.empty:
            continue

        alarms_s = raw_alarms.get(sensor, [])
        alarms_s = [a for a in alarms_s
                    if (t0 is None or a >= t0) and (t1 is None or a <= t1)]
        incidents = get_incidents(sensor, alarms_s)

        if not incidents:
            print(f"  {sensor}: 0 incidentes — FA/dia medido, recall=N/A")
            result = best_point_for_sensor(h, [], args.horizon,
                                           args.sticky_hours, fa_budget=args.fa_budget)
        else:
            result = best_point_for_sensor(h, incidents, args.horizon,
                                           args.sticky_hours, fa_budget=args.fa_budget)
            print(f"  {sensor}: {len(incidents)} inc | "
                  f"rec={result['recall']:.2f} FA={result['fa_per_day']:.3f}")

        result["sensor"] = sensor
        rows.append(result)

    df_out = pd.DataFrame(rows).set_index("sensor")
    col_order = ["n_incidents", "recall", "fa_per_day", "threshold_q",
                 "n_hit", "n_fp", "total_days"]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]

    mode_tag = ("_ok_aware" if args.ok_aware else "") + \
               (f"_sticky{int(args.sticky_hours)}h" if args.sticky_hours > 0 else "")
    out_label = f"{args.label}{mode_tag}"

    print(f"\n=== RESULTADO POR SENSOR (H={args.horizon}h, {out_label}) ===")
    print(f"{'Sensor':<18} {'Inc':>5} {'Recall':>8} {'FA/dia':>8}")
    print("─" * 44)
    for sensor, row in df_out.iterrows():
        n   = int(row.get("n_incidents", 0))
        rec = row.get("recall", float("nan"))
        fa  = row.get("fa_per_day", float("nan"))
        rec_str = "  N/A" if n == 0 else f"{rec:7.1%}"
        print(f"  {sensor:<16} {n:>5}  {rec_str}  {fa:>8.3f}")

    out_csv = os.path.join(args.out_dir, f"per_sensor_eval_{out_label}.csv")
    df_out.to_csv(out_csv)
    print(f"\nSalvo: {out_csv}")


if __name__ == "__main__":
    main()
