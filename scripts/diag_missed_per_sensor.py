#!/usr/bin/env python3
"""
diag_missed_per_sensor.py
Identifica incidentes perdidos por sensor e gera zoom plots para diagnóstico visual.

Uso:
    PYTHONPATH=. python scripts/diag_missed_per_sensor.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7 \
        --sensors TC382_01_A T5_AVG_A \
        --eval_start 2025-01-01 --eval_end 2025-12-31 \
        --label 2025

    # Para T5_AVG_A OOS:
    PYTHONPATH=. python scripts/diag_missed_per_sensor.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7 \
        --sensors T5_AVG_A \
        --eval_start 2026-01-01 --eval_end 2026-04-30 \
        --label OOS_2026
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from clearml import Task

ALARM_CSV_DEFAULT = "../dados/alarmes_selecionados_turbina_a.csv"
SAMPLING_INTERVAL = "5min"
DEBOUNCE_HOURS    = 8.0
HORIZON_HOURS     = 8.0
GAP_HOURS         = 4.0
ZOOM_HOURS        = 72.0    # janela de zoom em torno do incidente
N_THRESHOLDS      = 100


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_mae(task: Task, sensor: str) -> pd.Series:
    arts = task.artifacts
    key = next((k for k in arts if "sequence_scores_all" in k and sensor in k), None)
    if key is None:
        raise ValueError(f"Artifact não encontrado para sensor '{sensor}'")
    path = arts[key].get_local_copy()
    df = pd.read_csv(path)
    df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
    return df.set_index("seq_start_time")["mae_seq"]


def ewma_quantile(mae: pd.Series, half_life_hours: float) -> pd.Series:
    hl_pts = int(round(pd.Timedelta(hours=half_life_hours) / pd.Timedelta(SAMPLING_INTERVAL)))
    return mae.ewm(halflife=max(1, hl_pts)).mean().rank(pct=True)


def load_sensor_alarms(alarm_csv: str, sensor: str) -> List[pd.Timestamp]:
    df = pd.read_csv(alarm_csv)
    date_col = next(
        (c for c in df.columns if "ocorr" in c.lower() or ("data" in c.lower() and "ação" not in c.lower())),
        df.columns[0],
    )
    cond_col = next((c for c in df.columns if "condi" in c.lower()), None)
    tag_col  = next((c for c in df.columns if "tag" in c.lower() and "alarm" in c.lower()), None)
    df["_time"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    if cond_col:
        df = df[df[cond_col].str.upper().fillna("").ne("OK")]
    if tag_col:
        df = df[df[tag_col] == sensor]
    return sorted(pd.to_datetime(df["_time"], utc=True).dropna().tolist())


def cluster_incidents(alarm_times: List[pd.Timestamp]) -> List[pd.Timestamp]:
    if not alarm_times:
        return []
    s = pd.Series(alarm_times).sort_values().reset_index(drop=True)
    g = (s.diff().dt.total_seconds() / 3600 > GAP_HOURS).cumsum()
    return s.groupby(g).first().tolist()


# ---------------------------------------------------------------------------
# Identificação dos missed
# ---------------------------------------------------------------------------

def find_missed_incidents(
    health: pd.Series,
    incidents: List[pd.Timestamp],
    threshold_q: float,
    horizon_hours: float,
) -> List[pd.Timestamp]:
    horizon_sec = horizon_hours * 3600.0
    alert_s = np.array([t.timestamp() for t in health.index[health >= threshold_q]])
    missed = []
    for inc in incidents:
        ti = inc.timestamp()
        hit = alert_s.size > 0 and np.any((alert_s >= ti - horizon_sec) & (alert_s <= ti))
        if not hit:
            missed.append(inc)
    return missed


def best_threshold(
    health: pd.Series,
    incidents: List[pd.Timestamp],
    horizon_hours: float,
    fa_budget: float = 1.0,
) -> float:
    """Retorna o threshold_q que maximiza recall dentro do FA budget."""
    horizon_sec = horizon_hours * 3600.0
    total_days  = (health.index[-1] - health.index[0]).total_seconds() / 86400.0
    inc_s       = np.array([t.timestamp() for t in incidents])

    best_q, best_rec = 0.5, 0.0
    debounce = pd.Timedelta(hours=DEBOUNCE_HOURS)

    for q in np.linspace(0.50, 0.999, N_THRESHOLDS):
        alert    = health >= q
        alert_s  = np.array([t.timestamp() for t in health.index[alert]])

        # gap-debounce episodes
        on_idx = health.index[alert]
        episodes = []
        if len(on_idx) > 0:
            cs, ce = on_idx[0], on_idx[0]
            for t in on_idx[1:]:
                if (t - ce) <= debounce:
                    ce = t
                else:
                    episodes.append((cs, ce))
                    cs = ce = t
            episodes.append((cs, ce))

        n_hit = sum(
            1 for ti in inc_s
            if alert_s.size and np.any((alert_s >= ti - horizon_sec) & (alert_s <= ti))
        )
        n_fp = sum(
            1 for (s0, s1) in episodes
            if not (np.any((inc_s - horizon_sec <= s1.timestamp()) & (inc_s >= s0.timestamp())) if inc_s.size else False)
        )
        fa = n_fp / max(total_days, 1.0)
        rec = n_hit / len(incidents) if incidents else 0.0
        if fa <= fa_budget and rec > best_rec:
            best_rec, best_q = rec, q

    return best_q


# ---------------------------------------------------------------------------
# Plot de zoom de incidente perdido
# ---------------------------------------------------------------------------

def plot_missed(
    mae: pd.Series,
    health: pd.Series,
    threshold_q: float,
    incident: pd.Timestamp,
    sensor: str,
    horizon_hours: float,
    zoom_hours: float,
    out_path: str,
) -> None:
    zoom = pd.Timedelta(hours=zoom_hours)
    t0 = incident - zoom
    t1 = incident + pd.Timedelta(hours=24)

    mae_w    = mae[(mae.index >= t0) & (mae.index <= t1)]
    health_w = health[(health.index >= t0) & (health.index <= t1)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Painel 1: health score
    ax1.plot(health_w.index, health_w.values, color="steelblue", linewidth=1.2, label="Health (EWMA quantile)")
    ax1.axhline(threshold_q, color="red", linestyle="--", linewidth=1.2, label=f"Threshold={threshold_q:.3f}")
    # Janela [inc-H, inc]
    win_start = incident - pd.Timedelta(hours=horizon_hours)
    ax1.axvspan(win_start, incident, alpha=0.12, color="orange", label=f"Janela [{horizon_hours:.0f}h antes do incidente]")
    # Linha do incidente
    ymin1, ymax1 = 0.0, 1.0
    ax1.plot([incident, incident], [ymin1, ymax1], color="red", linewidth=2.0, label="Incidente (alarme)")
    ax1.set_ylabel("Health score")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(f"{sensor} — Incidente PERDIDO em {incident.strftime('%Y-%m-%d %H:%M')} UTC\n"
                  f"(threshold={threshold_q:.3f}, H={horizon_hours:.0f}h)", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Painel 2: MAE bruto
    ax2.plot(mae_w.index, mae_w.values, color="gray", linewidth=1.0, label="MAE raw")
    ymin2 = float(np.nanmin(mae_w.values)) if len(mae_w) > 0 else 0.0
    ymax2 = float(np.nanmax(mae_w.values)) if len(mae_w) > 0 else 1.0
    ax2.plot([incident, incident], [ymin2, ymax2], color="red", linewidth=2.0, label="Incidente")
    ax2.axvspan(win_start, incident, alpha=0.12, color="orange")
    ax2.set_ylabel("MAE")
    ax2.set_xlabel("Tempo (UTC)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id",    required=True)
    parser.add_argument("--sensors",    nargs="+", required=True, help="Sensores a analisar")
    parser.add_argument("--alarm_csv",  default=ALARM_CSV_DEFAULT)
    parser.add_argument("--half_life",  type=float, default=4.0)
    parser.add_argument("--horizon",    type=float, default=HORIZON_HOURS)
    parser.add_argument("--fa_budget",  type=float, default=1.0)
    parser.add_argument("--eval_start", default=None)
    parser.add_argument("--eval_end",   default=None)
    parser.add_argument("--zoom",       type=float, default=ZOOM_HOURS, help="Horas de zoom antes do incidente")
    parser.add_argument("--label",      default="diag")
    parser.add_argument("--out_dir",    default="eval_predictive_out/missed_diag")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = pd.Timestamp(args.eval_start, tz="UTC") if args.eval_start else None
    t1 = pd.Timestamp(args.eval_end,   tz="UTC") if args.eval_end   else None

    print(f"\nCarregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"  {task.name} | {task.get_status()}")

    for sensor in args.sensors:
        print(f"\n--- {sensor} ---")
        mae = load_mae(task, sensor)

        # Aplica filtro de período
        mae_eval = mae.copy()
        if t0:
            mae_eval = mae_eval[mae_eval.index >= t0]
        if t1:
            mae_eval = mae_eval[mae_eval.index <= t1]

        # EWMA quantile — calcula sobre série COMPLETA (normalização estável), filtra depois
        health_full = ewma_quantile(mae, args.half_life)
        health_eval = health_full.copy()
        if t0:
            health_eval = health_eval[health_eval.index >= t0]
        if t1:
            health_eval = health_eval[health_eval.index <= t1]

        # Alarmes do sensor no período
        alarms = load_sensor_alarms(args.alarm_csv, sensor)
        alarms = [a for a in alarms if (t0 is None or a >= t0) and (t1 is None or a <= t1)]
        incidents = cluster_incidents(alarms)
        print(f"  {len(alarms)} alarmes → {len(incidents)} incidentes no período")

        if not incidents:
            print("  Nenhum incidente — pulando.")
            continue

        # Threshold ótimo
        thr = best_threshold(health_eval, incidents, args.horizon, args.fa_budget)
        print(f"  Threshold ótimo: {thr:.3f}")

        # Missed
        missed = find_missed_incidents(health_eval, incidents, thr, args.horizon)
        print(f"  Incidentes perdidos ({len(missed)}/{len(incidents)}):")
        for m in missed:
            print(f"    {m.strftime('%Y-%m-%d %H:%M')} UTC")

        if not missed:
            print("  Nenhum incidente perdido neste período.")
            continue

        # Plot de cada missed
        for i, inc in enumerate(missed):
            out_path = os.path.join(
                args.out_dir,
                f"missed_{sensor}_{args.label}_{inc.strftime('%Y%m%d_%H%M')}.png"
            )
            plot_missed(
                mae=mae,           # série completa para contexto
                health=health_full,
                threshold_q=thr,
                incident=inc,
                sensor=sensor,
                horizon_hours=args.horizon,
                zoom_hours=args.zoom,
                out_path=out_path,
            )

    print(f"\nDiagnóstico salvo em: {args.out_dir}")


if __name__ == "__main__":
    main()
