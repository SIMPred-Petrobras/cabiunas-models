#!/usr/bin/env python
"""Analise de sensibilidade: distribuicao do erro de reconstrucao + threshold.

PLOT 1: grid 17 sensores — histograma do MAE cru com threshold q=0.715 marcado.
  Mostra "onde caemos" na distribuicao = sensibilidade global.

PLOT 2: TC382_03_A detalhado — MAE cru vs EWMA suavizado ao longo do tempo.
  Mostra o efeito do smoothing entre o erro cru e o sinal de alerta.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import load, SENSORS, TIME_COL, RUNNING_COL
from cnn1d_ae.predictive import compute_health_index_ewma

TS = 60; STRIDE = 10; HL = 4.0; Q_OP = 0.715
OUT_DIR = "relatorio_anexos"
MAE_CACHE = "improve_halflife_out/mae_per_sensor_cache.npz"
FOCUS_SENSOR = "TC382_03_A"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, _ = load(priority=None)
    n = len(df)
    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    starts = np.arange(0, n - TS + 1, STRIDE)
    ends = starts + TS - 1
    t_end_pd = pd.DatetimeIndex(df[TIME_COL].to_numpy()[ends])
    seq_run_frac = np.array([op[s:s+TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999

    print("[VIZ] carregando MAE cache")
    mae = np.load(MAE_CACHE)["mae"]
    print(f"[VIZ] MAE: {mae.shape}")

    # ===== PLOT 1: grid 17 distribuicoes =====
    print("[VIZ] gerando grid de distribuicoes...")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 4, figsize=(16, 14))
    axes_flat = axes.flatten()
    for j, sensor in enumerate(SENSORS):
        ax = axes_flat[j]
        mae_run = mae[seq_run_full, j]
        thr = float(np.quantile(mae_run, Q_OP))
        # histograma com escala log no Y pra ver a cauda
        ax.hist(mae_run, bins=80, color="steelblue", alpha=0.8)
        ax.axvline(thr, color="red", ls="--", lw=1.2,
                   label=f"thr q={Q_OP}\n=({thr:.4f})")
        pct_above = 100 * (mae_run >= thr).mean()
        # stats overlay
        ax.text(0.98, 0.92, f"% acima thr: {pct_above:.0f}%\n"
                            f"med: {np.median(mae_run):.4f}\n"
                            f"p99: {np.quantile(mae_run, 0.99):.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        ax.set_title(sensor, fontsize=10)
        ax.set_yscale("log")
        ax.set_xlabel("MAE/seq", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    # subplots vazios
    for k in range(len(SENSORS), len(axes_flat)):
        axes_flat[k].set_visible(False)
    fig.suptitle(
        f"Distribuição do erro de reconstrução (MAE) por sensor — operação somente\n"
        f"linha vermelha = threshold individual no quantil {Q_OP} (define alerta)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_mae_distribution_grid.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] salvo fig_mae_distribution_grid.png")

    # ===== PLOT 2: TC382_03_A detalhado MAE cru vs EWMA =====
    sensor_idx = SENSORS.index(FOCUS_SENSOR)
    dt_seconds = STRIDE * 30.0
    health = compute_health_index_ewma(
        mae[:, sensor_idx], seq_run_frac,
        half_life_hours=HL, dt_seconds=dt_seconds,
    )
    thr_mae = float(np.quantile(mae[seq_run_full, sensor_idx], Q_OP))
    thr_health = float(np.quantile(health[seq_run_full], Q_OP))

    # alarmes do tag
    df_alarm = pd.read_csv("../dados/alarmes_selecionados_turbina_a.csv")
    df_alarm["t"] = pd.to_datetime(df_alarm["Data da Ocorrência"], errors="coerce")
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    al_sensor = df_alarm[(df_alarm["t"] >= t_min) & (df_alarm["t"] <= t_max)
                         & (df_alarm["Tag Alarme"] == FOCUS_SENSOR)
                         & (df_alarm["Condição do Alarme"] != "OK")].copy()
    al_sensor = al_sensor.sort_values("t").reset_index(drop=True)
    grp = (al_sensor["t"].diff().dt.total_seconds()/3600 > 4).cumsum()
    incidents = al_sensor.groupby(grp)["t"].min().reset_index(drop=True)
    print(f"[VIZ] {FOCUS_SENSOR}: {len(incidents)} incidentes")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    t_dt = t_end_pd.to_pydatetime()

    # Painel 1: MAE cru
    mae_focus = mae[:, sensor_idx].copy()
    mae_focus_run = np.where(seq_run_full, mae_focus, np.nan)
    ax1.plot(t_dt, mae_focus_run, lw=0.3, color="gray", alpha=0.7, label="MAE cru (por janela)")
    ax1.axhline(thr_mae, color="red", ls="--", lw=1,
                label=f"thr cru (q={Q_OP}) = {thr_mae:.4f}")
    ax1.set_ylabel("MAE cru\n(reconstrução)")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize="small")
    ax1.set_title(
        f"{FOCUS_SENSOR}: MAE cru × health-index suavizado\n"
        f"linhas verdes = alarmes oficiais ({len(incidents)} incidentes)",
        fontsize=11,
    )
    for ti in incidents:
        ax1.axvline(ti, color="green", alpha=0.4, lw=1)

    # Painel 2: health (EWMA)
    health_run = np.where(seq_run_full, health, np.nan)
    ax2.plot(t_dt, health_run, lw=0.5, color="steelblue", label="health = EWMA(MAE), half-life=4h")
    ax2.axhline(thr_health, color="red", ls="--", lw=1,
                label=f"thr health (q={Q_OP}) = {thr_health:.4f}")
    above = health >= thr_health
    above_in_op = above & seq_run_full
    ax2.scatter(np.array(t_dt)[above_in_op], health[above_in_op], s=2, color="orange",
                label="acima do threshold (alerta)", zorder=3)
    pct_alert = 100 * above_in_op.sum() / max(seq_run_full.sum(), 1)
    ax2.set_ylabel(f"health-index\n(EWMA)")
    ax2.set_xlabel("tempo")
    ax2.legend(loc="upper left", fontsize="small")
    ax2.text(0.98, 0.95, f"alerta ativo em {pct_alert:.1f}% do tempo de operação",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    for ti in incidents:
        ax2.axvline(ti, color="green", alpha=0.4, lw=1)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_mae_vs_ewma_{FOCUS_SENSOR}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] salvo fig_mae_vs_ewma_{FOCUS_SENSOR}.png")

    # Stats globais para resumo:
    print("\n[VIZ] === resumo de sensibilidade por sensor ===")
    print(f"{'sensor':<14}{'thr q=0.715':>12}{'% acima':>10}{'media MAE':>12}{'p99 MAE':>12}")
    print("-" * 60)
    for j, sensor in enumerate(SENSORS):
        mae_run = mae[seq_run_full, j]
        thr = float(np.quantile(mae_run, Q_OP))
        pct = 100 * (mae_run >= thr).mean()
        med = float(np.median(mae_run))
        p99 = float(np.quantile(mae_run, 0.99))
        print(f"{sensor:<14}{thr:>12.4f}{pct:>9.1f}%{med:>12.4f}{p99:>12.4f}")


if __name__ == "__main__":
    main()
