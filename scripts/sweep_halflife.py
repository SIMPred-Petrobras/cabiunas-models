#!/usr/bin/env python3
"""
sweep_halflife.py
Varre valores de half-life do EWMA para sensores específicos e mede recall x FA.
Útil para diagnosticar misses por detecção precoce (EWMA decaiu antes do alarme).

Uso:
    PYTHONPATH=. python scripts/sweep_halflife.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7 \
        --sensors TC382_01_A T5_AVG_A \
        --eval_start 2025-01-01 --eval_end 2025-12-31 \
        --label 2025

    # T5_AVG_A OOS:
    PYTHONPATH=. python scripts/sweep_halflife.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7 \
        --sensors T5_AVG_A \
        --eval_start 2026-01-01 --eval_end 2026-04-30 \
        --label OOS_2026
"""
from __future__ import annotations

import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from clearml import Task

ALARM_CSV_DEFAULT = "../dados/alarmes_selecionados_turbina_a.csv"
SAMPLING_INTERVAL = "5min"
DEBOUNCE_HOURS    = 8.0
HORIZON_HOURS     = 8.0
GAP_HOURS         = 4.0
N_THRESHOLDS      = 80

HALF_LIVES = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 48.0]


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
    df = df.copy()
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


def best_point(
    health: pd.Series,
    incidents: List[pd.Timestamp],
    horizon_hours: float,
    fa_budget: float,
) -> dict:
    horizon_sec = horizon_hours * 3600.0
    total_days  = (health.index[-1] - health.index[0]).total_seconds() / 86400.0
    inc_s       = np.array([t.timestamp() for t in incidents])
    debounce    = pd.Timedelta(hours=DEBOUNCE_HOURS)

    best = {"recall": 0.0, "fa_per_day": 0.0, "threshold_q": 0.5}
    best_rec = -1.0

    for q in np.linspace(0.50, 0.999, N_THRESHOLDS):
        alert   = health >= q
        alert_s = np.array([t.timestamp() for t in health.index[alert]])

        on_idx   = health.index[alert]
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
        fa  = n_fp / max(total_days, 1.0)
        rec = n_hit / len(incidents) if incidents else 0.0

        if fa <= fa_budget and rec > best_rec:
            best_rec = rec
            best = {"recall": rec, "fa_per_day": fa, "threshold_q": float(q), "n_hit": n_hit}

    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id",    required=True)
    parser.add_argument("--sensors",    nargs="+", required=True)
    parser.add_argument("--alarm_csv",  default=ALARM_CSV_DEFAULT)
    parser.add_argument("--half_lives", type=float, nargs="+", default=HALF_LIVES)
    parser.add_argument("--horizon",    type=float, default=HORIZON_HOURS)
    parser.add_argument("--fa_budget",  type=float, default=1.0)
    parser.add_argument("--eval_start", default=None)
    parser.add_argument("--eval_end",   default=None)
    parser.add_argument("--label",      default="sweep")
    parser.add_argument("--out_dir",    default="eval_predictive_out/halflife_sweep")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = pd.Timestamp(args.eval_start, tz="UTC") if args.eval_start else None
    t1 = pd.Timestamp(args.eval_end,   tz="UTC") if args.eval_end   else None

    print(f"\nCarregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"  {task.name} | {task.get_status()}")

    all_results: Dict[str, list] = {}

    for sensor in args.sensors:
        print(f"\n=== {sensor} ===")
        mae = load_mae(task, sensor)

        alarms = load_sensor_alarms(args.alarm_csv, sensor)
        alarms = [a for a in alarms if (t0 is None or a >= t0) and (t1 is None or a <= t1)]
        incidents = cluster_incidents(alarms)
        print(f"  {len(incidents)} incidentes no período")
        if not incidents:
            print("  Sem incidentes — pulando.")
            continue

        rows = []
        print(f"  {'Half-life':>10} | {'Recall':>8} | {'FA/dia':>8} | {'n_hit':>6}")
        print(f"  {'-'*42}")
        for hl in args.half_lives:
            # EWMA sobre série completa, filtra para avaliação
            health_full = ewma_quantile(mae, hl)
            health_eval = health_full.copy()
            if t0:
                health_eval = health_eval[health_eval.index >= t0]
            if t1:
                health_eval = health_eval[health_eval.index <= t1]

            res = best_point(health_eval, incidents, args.horizon, args.fa_budget)
            res["half_life"] = hl
            rows.append(res)
            print(f"  {hl:>8.1f}h  | {res['recall']:>8.1%} | {res['fa_per_day']:>8.3f} | {res.get('n_hit',0):>6}/{len(incidents)}")

        all_results[sensor] = rows

        # Salva CSV por sensor
        df_s = pd.DataFrame(rows).set_index("half_life")
        csv_path = os.path.join(args.out_dir, f"halflife_sweep_{sensor}_{args.label}.csv")
        df_s.to_csv(csv_path)

    # Plot comparativo
    n_sensors = len(all_results)
    if n_sensors == 0:
        print("\nNenhum sensor com incidentes — sem plot.")
        return

    fig, axes = plt.subplots(1, n_sensors, figsize=(6 * n_sensors, 4), squeeze=False)

    for ax, (sensor, rows) in zip(axes[0], all_results.items()):
        df_p = pd.DataFrame(rows)
        hl_vals  = df_p["half_life"].values
        rec_vals = df_p["recall"].values
        fa_vals  = df_p["fa_per_day"].values

        color_rec = "steelblue"
        color_fa  = "tomato"

        ax2 = ax.twinx()
        ax.plot(hl_vals, rec_vals, "o-", color=color_rec, linewidth=2, markersize=6, label="Recall")
        ax2.plot(hl_vals, fa_vals,  "s--", color=color_fa,  linewidth=1.5, markersize=5, label="FA/dia")

        # Marca hl=4 (atual)
        ax.axvline(4.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Half-life EWMA (h)")
        ax.set_ylabel("Recall", color=color_rec)
        ax2.set_ylabel("FA/dia", color=color_fa)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{sensor}\n({args.label}, H={args.horizon:.0f}h)", fontsize=10)
        ax.tick_params(axis="y", labelcolor=color_rec)
        ax2.tick_params(axis="y", labelcolor=color_fa)
        ax.grid(True, alpha=0.3)

        # Legenda combinada
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    plt.suptitle(f"Sweep de Half-life EWMA — {args.label}  (linha pontilhada = atual 4h)", fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"fig_halflife_sweep_{args.label}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Figura: {fig_path}")
    print(f"[OK] CSVs em: {args.out_dir}")


if __name__ == "__main__":
    main()
