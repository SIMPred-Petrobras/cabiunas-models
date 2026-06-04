#!/usr/bin/env python3
"""
diag_cross_sensor.py
Diagnóstico cruzado entre sensores:
  1. TV_354X/Y_A — os misses são o mesmo incidente? O companheiro X/Y cobriu?
  2. TC382_01_A Ago/08 — algum outro sensor detectou algo naquela data?
  3. Gera overview de health score de todos os sensores em janelas de tempo específicas.

Uso:
    PYTHONPATH=. python scripts/diag_cross_sensor.py \
        --task_id e6f1a38c8f5e4154b747e6aae9d6dfc7
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def load_all_mae(task: Task) -> Dict[str, pd.Series]:
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
    print(f"  {len(series)}/{len(SENSORS)} sensores carregados")
    return series


def ewma_quantile(mae: pd.Series, half_life_hours: float = 4.0) -> pd.Series:
    hl_pts = int(round(pd.Timedelta(hours=half_life_hours) / pd.Timedelta(SAMPLING_INTERVAL)))
    return mae.ewm(halflife=max(1, hl_pts)).mean().rank(pct=True)


def load_sensor_alarms(alarm_csv: str, sensor: str,
                       t0=None, t1=None) -> List[pd.Timestamp]:
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
    times = pd.to_datetime(df["_time"], utc=True).dropna()
    if t0:
        times = times[times >= t0]
    if t1:
        times = times[times <= t1]
    return sorted(times.tolist())


def cluster_incidents(alarm_times: List[pd.Timestamp]) -> List[pd.Timestamp]:
    if not alarm_times:
        return []
    s = pd.Series(alarm_times).sort_values().reset_index(drop=True)
    g = (s.diff().dt.total_seconds() / 3600 > GAP_HOURS).cumsum()
    return s.groupby(g).first().tolist()


def find_missed(health: pd.Series, incidents: List[pd.Timestamp],
                threshold_q: float = 0.5) -> List[pd.Timestamp]:
    horizon_sec = HORIZON_HOURS * 3600.0
    alert_s = np.array([t.timestamp() for t in health.index[health >= threshold_q]])
    return [
        inc for inc in incidents
        if not (alert_s.size and np.any(
            (alert_s >= inc.timestamp() - horizon_sec) & (alert_s <= inc.timestamp())
        ))
    ]


# ---------------------------------------------------------------------------
# Diagnóstico 1 — TV_354 cross X/Y
# ---------------------------------------------------------------------------

def diag_tv354(health_dict: Dict[str, pd.Series], alarm_csv: str,
               t0, t1, out_dir: str) -> None:
    print("\n=== DIAG 1: TV_354X/Y_A — cross-check de misses ===")

    pairs = [("TV_354X_A", "TV_354Y_A"), ("TV_355X_A", "TV_355Y_A")]

    for sx, sy in pairs:
        for focal, companion in [(sx, sy), (sy, sx)]:
            if focal not in health_dict or companion not in health_dict:
                continue

            h_focal     = health_dict[focal]
            h_focal     = h_focal[(h_focal.index >= t0) & (h_focal.index <= t1)]
            h_companion = health_dict[companion]
            h_companion = h_companion[(h_companion.index >= t0) & (h_companion.index <= t1)]

            alarms   = load_sensor_alarms(alarm_csv, focal, t0, t1)
            incidents = cluster_incidents(alarms)
            if not incidents:
                continue
            missed = find_missed(h_focal, incidents)
            if not missed:
                print(f"  {focal}: sem misses")
                continue

            print(f"\n  {focal}: {len(missed)} missed, verificando {companion}...")
            for inc in missed:
                win_s = inc - pd.Timedelta(hours=HORIZON_HOURS)
                h_comp_win = h_companion[(h_companion.index >= win_s) & (h_companion.index <= inc)]
                max_comp = h_comp_win.max() if len(h_comp_win) > 0 else 0.0
                covered = max_comp >= 0.5
                print(f"    {inc.strftime('%Y-%m-%d %H:%M')} UTC  | "
                      f"{companion} max health={max_comp:.3f} → {'COBERTO ✓' if covered else 'NÃO COBERTO ✗'}")

                # Plot zoom
                zoom = pd.Timedelta(hours=48)
                t_start = inc - zoom
                t_end   = inc + pd.Timedelta(hours=12)

                fig, ax = plt.subplots(figsize=(13, 4))
                h_f = health_dict[focal]
                h_c = health_dict[companion]
                h_f_w = h_f[(h_f.index >= t_start) & (h_f.index <= t_end)]
                h_c_w = h_c[(h_c.index >= t_start) & (h_c.index <= t_end)]

                ax.plot(h_f_w.index, h_f_w.values, color="steelblue", lw=1.5, label=f"{focal} (PERDEU)")
                ax.plot(h_c_w.index, h_c_w.values, color="darkorange", lw=1.5, label=f"{companion} (companheiro)")
                ax.axhline(0.5, color="red", ls="--", lw=1.0, label="Threshold=0.5")
                ax.axvspan(inc - pd.Timedelta(hours=HORIZON_HOURS), inc, alpha=0.12, color="orange")
                ymin, ymax = 0.0, 1.0
                ax.plot([inc, inc], [ymin, ymax], color="red", lw=2.0, label="Incidente")
                ax.set_ylim(0, 1.05)
                ax.set_title(f"Cross-check: {focal} (missed) vs {companion}\n"
                             f"Incidente: {inc.strftime('%Y-%m-%d %H:%M')} UTC  |  "
                             f"{companion} max={max_comp:.3f} → {'COBERTO' if covered else 'NÃO COBERTO'}", fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
                plt.tight_layout()

                fname = f"cross_{focal}_{inc.strftime('%Y%m%d_%H%M')}.png"
                plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
                plt.close()
                print(f"    Salvo: {fname}")


# ---------------------------------------------------------------------------
# Diagnóstico 2 — TC382_01_A Aug/08: todos os sensores naquele dia
# ---------------------------------------------------------------------------

def diag_tc382_aug08(health_dict: Dict[str, pd.Series], out_dir: str) -> None:
    print("\n=== DIAG 2: TC382_01_A Ago/08 — overview cross-sensor ===")

    incident = pd.Timestamp("2025-08-08 15:31", tz="UTC")
    zoom      = pd.Timedelta(hours=72)
    t_start   = incident - zoom
    t_end     = incident + pd.Timedelta(hours=24)

    n = len(health_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 2.8), sharey=True)
    axes_flat = axes.flatten()

    for i, (sensor, health) in enumerate(health_dict.items()):
        ax = axes_flat[i]
        h_w = health[(health.index >= t_start) & (health.index <= t_end)]
        color = "steelblue" if sensor != "TC382_01_A" else "tomato"
        ax.plot(h_w.index, h_w.values, color=color, lw=1.2)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.axvspan(incident - pd.Timedelta(hours=HORIZON_HOURS), incident,
                   alpha=0.15, color="orange")
        ymin, ymax = 0.0, 1.0
        ax.plot([incident, incident], [ymin, ymax], color="red", lw=1.5)

        max_in_window = h_w[
            (h_w.index >= incident - pd.Timedelta(hours=HORIZON_HOURS)) &
            (h_w.index <= incident)
        ].max() if len(h_w) > 0 else 0.0

        covered = max_in_window >= 0.5
        title_color = "green" if covered else "black"
        ax.set_title(f"{sensor}\nmax={max_in_window:.2f} {'✓' if covered else '✗'}",
                     fontsize=8, color=title_color)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    # Esconde eixos extras
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Cross-sensor: TC382_01_A miss em 2025-08-08 15:31 UTC\n"
        f"(vermelho=TC382_01_A, verde no título=sensor alertou na janela H=8h)",
        fontsize=11
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "cross_tc382_01a_aug08_all_sensors.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out_path}")

    # Resumo textual
    print(f"  Sensores com alerta na janela [inc-8h, inc]:")
    for sensor, health in health_dict.items():
        h_w = health[
            (health.index >= incident - pd.Timedelta(hours=HORIZON_HOURS)) &
            (health.index <= incident)
        ]
        mx = h_w.max() if len(h_w) > 0 else 0.0
        if mx >= 0.5:
            print(f"    ✓ {sensor}: max health={mx:.3f}")
    print(f"  Sensores SEM alerta:")
    for sensor, health in health_dict.items():
        h_w = health[
            (health.index >= incident - pd.Timedelta(hours=HORIZON_HOURS)) &
            (health.index <= incident)
        ]
        mx = h_w.max() if len(h_w) > 0 else 0.0
        if mx < 0.5:
            print(f"    ✗ {sensor}: max health={mx:.3f}")


# ---------------------------------------------------------------------------
# Diagnóstico 3 — TV_355Y_A: detalhe do único incidente perdido
# ---------------------------------------------------------------------------

def diag_tv355y(health_dict: Dict[str, pd.Series], alarm_csv: str,
                t0, t1, out_dir: str) -> None:
    print("\n=== DIAG 3: TV_355Y_A — incidente perdido vs TV_355X_A ===")

    sensor     = "TV_355Y_A"
    companion  = "TV_355X_A"

    alarms    = load_sensor_alarms(alarm_csv, sensor, t0, t1)
    incidents = cluster_incidents(alarms)
    if not incidents:
        print("  Sem incidentes no período.")
        return

    h_y = health_dict.get(sensor)
    h_x = health_dict.get(companion)
    if h_y is None:
        print(f"  {sensor} não carregado.")
        return

    h_y_eval = h_y[(h_y.index >= t0) & (h_y.index <= t1)]
    missed = find_missed(h_y_eval, incidents)
    print(f"  {sensor}: {len(incidents)} incidentes, {len(missed)} perdidos")

    for inc in incidents:
        win_s = inc - pd.Timedelta(hours=HORIZON_HOURS)
        h_y_win = h_y[(h_y.index >= win_s) & (h_y.index <= inc)]
        max_y   = h_y_win.max() if len(h_y_win) > 0 else 0.0

        h_x_win = h_x[(h_x.index >= win_s) & (h_x.index <= inc)] if h_x is not None else pd.Series(dtype=float)
        max_x   = h_x_win.max() if len(h_x_win) > 0 else 0.0

        is_missed = inc in missed
        print(f"  {inc.strftime('%Y-%m-%d %H:%M')} UTC  |  "
              f"{sensor}={max_y:.3f} {'PERDIDO ✗' if is_missed else 'PEGO ✓'}  |  "
              f"{companion}={max_x:.3f} {'COBERTO ✓' if max_x >= 0.5 else 'NÃO COBERTO ✗'}")

        # Plot zoom
        zoom    = pd.Timedelta(hours=72)
        t_start = inc - zoom
        t_end   = inc + pd.Timedelta(hours=12)

        fig, ax = plt.subplots(figsize=(13, 4))
        h_y_w = h_y[(h_y.index >= t_start) & (h_y.index <= t_end)]
        ax.plot(h_y_w.index, h_y_w.values, color="steelblue", lw=1.5, label=f"{sensor} (0% recall)")
        if h_x is not None:
            h_x_w = h_x[(h_x.index >= t_start) & (h_x.index <= t_end)]
            ax.plot(h_x_w.index, h_x_w.values, color="darkorange", lw=1.5, label=f"{companion} (companheiro X)")
        ax.axhline(0.5, color="red", ls="--", lw=1.0, label="Threshold=0.5")
        ax.axvspan(win_s, inc, alpha=0.12, color="orange", label="Janela H=8h")
        ax.plot([inc, inc], [0.0, 1.0], color="red", lw=2.0, label="Incidente")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"TV_355Y_A (0% recall) — único incidente: {inc.strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"TV_355Y_A max={max_y:.3f} (PERDIDO) | TV_355X_A max={max_x:.3f} "
            f"({'COBERTO' if max_x >= 0.5 else 'NÃO COBERTO'})", fontsize=10
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.tight_layout()

        fname = f"cross_TV355Y_{inc.strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Salvo: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id",    required=True)
    parser.add_argument("--alarm_csv",  default=ALARM_CSV_DEFAULT)
    parser.add_argument("--half_life",  type=float, default=4.0)
    parser.add_argument("--eval_start", default="2025-01-01")
    parser.add_argument("--eval_end",   default="2025-12-31")
    parser.add_argument("--out_dir",    default="eval_predictive_out/cross_sensor_diag")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = pd.Timestamp(args.eval_start, tz="UTC")
    t1 = pd.Timestamp(args.eval_end,   tz="UTC")

    print(f"\nCarregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"  {task.name} | {task.get_status()}")

    print(f"\nBaixando sequence_scores de todos os sensores...")
    mae_dict = load_all_mae(task)

    print(f"\nCalculando EWMA (hl={args.half_life}h) + quantile normalization...")
    health_dict = {s: ewma_quantile(mae, args.half_life) for s, mae in mae_dict.items()}

    diag_tv354(health_dict, args.alarm_csv, t0, t1, args.out_dir)
    diag_tv355y(health_dict, args.alarm_csv, t0, t1, args.out_dir)
    diag_tc382_aug08(health_dict, args.out_dir)

    print(f"\nTodos os diagnósticos salvos em: {args.out_dir}")


if __name__ == "__main__":
    main()
