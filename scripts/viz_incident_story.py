#!/usr/bin/env python
"""Plot 'detective story' de um incidente: sensor bruto + health + multi-sensor.

Mostra 3 paineis sincronizados ao longo de ~7 dias ao redor do incidente:
1. Valor bruto do sensor (o que a operacao ve)
2. Health-index daquele sensor (o que o AE pensa)
3. Contagem multi-sensor (sinal agregado)

A unica fonte de evidencia que convence a operacao: 'olha o sensor real subindo'.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import load, SENSORS, TIME_COL, RUNNING_COL
from cnn1d_ae.predictive import compute_health_index_ewma

TS = 60; STRIDE = 10
HL = 4.0; Q_OP = 0.715; GAP_H = 12.0
OUT_DIR = "relatorio_anexos"
MAE_CACHE = "improve_halflife_out/mae_per_sensor_cache.npz"
DAYS_BEFORE = 7
DAYS_AFTER = 1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, inc_full = load(priority=None)
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    n = len(df)
    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    starts = np.arange(0, n - TS + 1, STRIDE)
    ends = starts + TS - 1
    t_end_pd = pd.DatetimeIndex(df[TIME_COL].to_numpy()[ends])
    t_end_sec = t_end_pd.values.astype("datetime64[s]").astype("int64").astype(float)
    seq_run_frac = np.array([op[s:s+TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999

    print(f"[STORY] carregando MAE cache")
    mae = np.load(MAE_CACHE)["mae"]
    n_sens = mae.shape[1]

    dt_seconds = STRIDE * 30.0
    health = np.empty_like(mae)
    for j in range(n_sens):
        health[:, j] = compute_health_index_ewma(
            mae[:, j], seq_run_frac, half_life_hours=HL, dt_seconds=dt_seconds,
        )
    # thresholds in-sample
    thr = np.array([float(np.quantile(health[seq_run_full, j], Q_OP)) for j in range(n_sens)])
    above = (health >= thr[None, :])
    n_above = above.sum(axis=1).astype(float)

    # escolhe incidente: o mesmo do zoom anterior — pico de TV_354Y_A em 2025-04
    target_ti = pd.Timestamp("2025-04-09 07:40:35")
    print(f"[STORY] incidente alvo: {target_ti}")
    # janela de plot
    t_start = target_ti - pd.Timedelta(days=DAYS_BEFORE)
    t_end = target_ti + pd.Timedelta(days=DAYS_AFTER)

    # idx do sensor que mais disparou nesse periodo
    win = (t_end_pd >= t_start) & (t_end_pd <= t_end)
    ratio_in_win = (health[win] / thr[None, :])
    sensor_idx = int(np.nanargmax(np.nanmax(ratio_in_win, axis=0)))
    sensor_name = SENSORS[sensor_idx]
    print(f"[STORY] sensor com maior pico na janela: {sensor_name}")

    # === PLOT ===
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 1.5]})
    # janela em datetime
    df_win_mask = (df[TIME_COL] >= t_start) & (df[TIME_COL] <= t_end)
    t_raw = df.loc[df_win_mask, TIME_COL].to_numpy()
    raw_vals = pd.to_numeric(df.loc[df_win_mask, sensor_name], errors="coerce").to_numpy()
    op_vals = pd.to_numeric(df.loc[df_win_mask, RUNNING_COL], errors="coerce").fillna(0).to_numpy()

    # painel 1: valor bruto do sensor
    ax0 = axes[0]
    # OFF como faixa
    off = op_vals < 0.5
    if off.any():
        ax0.fill_between(t_raw, raw_vals.min() if not np.all(np.isnan(raw_vals)) else 0,
                         raw_vals.max() if not np.all(np.isnan(raw_vals)) else 1,
                         where=off, color="lightgray", alpha=0.4, step="mid",
                         label="máquina OFF")
    ax0.plot(t_raw, raw_vals, lw=0.6, color="steelblue", label=f"{sensor_name} (bruto)")
    # janela de detecao do sensor (health acima do thr)
    t_seq_win = t_end_pd[win].to_numpy()
    sensor_above_win = above[win, sensor_idx]
    if sensor_above_win.any():
        # mapeia idx de seq para idx de raw aproximado (cobertura)
        for i in range(len(sensor_above_win)-1):
            if sensor_above_win[i]:
                t0 = t_seq_win[i]
                t1 = t_seq_win[i+1] if i+1 < len(t_seq_win) else t0
                ax0.axvspan(t0, t1, color="orange", alpha=0.15)
    ax0.axvline(target_ti, color="red", lw=2, label=f"ALARME REAL ({target_ti:%d/%m %H:%M})")
    ax0.set_ylabel(f"{sensor_name}\n(valor bruto)")
    ax0.set_title(
        f"Detective story: sensor {sensor_name} ao redor do alarme {target_ti:%Y-%m-%d %H:%M}\n"
        f"faixa laranja = AE deteceu anomalia | linha vermelha = alarme real do sistema",
        fontsize=11
    )
    ax0.legend(loc="upper left", fontsize="small")

    # painel 2: health-index do sensor
    ax1 = axes[1]
    t_seq_full = t_end_pd[win].to_pydatetime()
    h_win = health[win, sensor_idx]
    ax1.plot(t_seq_full, h_win, lw=1.2, color="steelblue", label="health-index EWMA")
    ax1.axhline(thr[sensor_idx], color="red", ls="--", lw=1,
                label=f"threshold individual (thr={thr[sensor_idx]:.4f})")
    above_mask = h_win >= thr[sensor_idx]
    ax1.fill_between(t_seq_full, 0, h_win, where=above_mask, color="orange", alpha=0.3,
                     step="mid", label="acima do threshold")
    ax1.axvline(target_ti, color="red", lw=2)
    ax1.set_ylabel("health-index\n(EWMA do MAE)")
    ax1.legend(loc="upper left", fontsize="small")

    # painel 3: contagem multi-sensor
    ax2 = axes[2]
    n_above_win = n_above[win]
    ax2.plot(t_seq_full, n_above_win, lw=0.8, color="purple", label="n. sensores acima do thr")
    expected = 17 * (1 - Q_OP)
    ax2.axhline(expected, color="gray", ls=":", lw=1, label=f"esperado por chance ({expected:.1f})")
    ax2.axhline(expected*2, color="red", ls="--", lw=1, label=f"alerta forte (≥{expected*2:.0f})")
    ax2.axvline(target_ti, color="red", lw=2)
    ax2.set_ylim(0, 17.5)
    ax2.set_ylabel("multi-sensor")
    ax2.set_xlabel("tempo")
    ax2.legend(loc="upper left", fontsize="small")

    # formato datas
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_incident_story_{sensor_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[STORY] salvo fig_incident_story_{sensor_name}.png")

    # quantos pontos do sensor estavam acima nas 24h, 48h, 72h antes
    for hours_back in [12, 24, 48, 72]:
        mask_pre = (t_end_pd >= target_ti - pd.Timedelta(hours=hours_back)) & (t_end_pd <= target_ti)
        pct = 100 * above[mask_pre, sensor_idx].mean() if mask_pre.any() else 0
        print(f"[STORY] {sensor_name} acima do thr nas ultimas {hours_back}h: {pct:.1f}%")


if __name__ == "__main__":
    main()
