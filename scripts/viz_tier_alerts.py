#!/usr/bin/env python
"""Visualizacoes do sistema tier: panorama anual + zoom em incidente.

Plot 1: ano inteiro com 3 tiers (warn/alarm/critical) sobrepostos + alarmes oficiais
Plot 2: zoom num incidente mostrando escalada warn → alarm → critical
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import load, SENSORS, TIME_COL, RUNNING_COL
from cnn1d_ae.predictive import compute_health_index_ewma

TS = 60; STRIDE = 10; HL = 4.0
WARN_Q, WARN_K = 0.80, 2
ALARM_Q, ALARM_K = 0.90, 2
CRITICAL_Q, CRITICAL_K = 0.95, 1
OUT_DIR = "relatorio_anexos"
MAE_CACHE = "improve_halflife_out/mae_per_sensor_cache.npz"
FOCUS_SENSOR = "TC382_03_A"


def compute_tier(health, seq_run_full, q, k):
    n_sens = health.shape[1]
    thr = np.empty(n_sens, dtype=float)
    for j in range(n_sens):
        valid = health[seq_run_full, j]
        thr[j] = float(np.quantile(valid, q)) if valid.size else float("inf")
    above = (health >= thr[None, :])
    n_above = above.sum(axis=1)
    return (n_above >= k) & seq_run_full


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
    seq_run_frac = np.array([op[s:s+TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999

    print("[VIZ] carregando MAE cache")
    mae = np.load(MAE_CACHE)["mae"]
    dt_seconds = STRIDE * 30.0
    health = np.empty_like(mae)
    for j in range(mae.shape[1]):
        health[:, j] = compute_health_index_ewma(mae[:, j], seq_run_frac,
                                                  half_life_hours=HL, dt_seconds=dt_seconds)

    warn = compute_tier(health, seq_run_full, WARN_Q, WARN_K)
    alarm = compute_tier(health, seq_run_full, ALARM_Q, ALARM_K)
    critical = compute_tier(health, seq_run_full, CRITICAL_Q, CRITICAL_K)
    print(f"[VIZ] WARN={warn.sum()} ALARM={alarm.sum()} CRITICAL={critical.sum()}")

    # Sensor bruto + alarmes oficiais do tag
    raw = pd.to_numeric(df[FOCUS_SENSOR], errors="coerce").to_numpy()
    t_raw = df[TIME_COL].to_numpy()

    df_alarm = pd.read_csv("../dados/alarmes_selecionados_turbina_a.csv")
    df_alarm["t"] = pd.to_datetime(df_alarm["Data da Ocorrência"], errors="coerce")
    al = df_alarm[(df_alarm["t"] >= t_min) & (df_alarm["t"] <= t_max)
                  & (df_alarm["Tag Alarme"] == FOCUS_SENSOR)
                  & (df_alarm["Condição do Alarme"] != "OK")].sort_values("t").reset_index(drop=True)
    grp = (al["t"].diff().dt.total_seconds()/3600 > 4).cumsum()
    incidents = al.groupby(grp)["t"].min().reset_index(drop=True)
    print(f"[VIZ] {FOCUS_SENSOR}: {len(incidents)} incidentes")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    t_dt = t_end_pd.to_pydatetime()

    # ===== PLOT 1: panorama anual =====
    fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 0.7, 0.7, 0.7]})
    # painel sensor bruto
    ax_raw = axes[0]
    off_mask = op < 0.5
    if off_mask.any():
        ymin = float(np.nanmin(raw)); ymax = float(np.nanmax(raw))
        ax_raw.fill_between(t_raw, ymin, ymax, where=off_mask, color="lightgray", alpha=0.35,
                            step="mid")
    ax_raw.plot(t_raw, raw, lw=0.4, color="steelblue", label=f"{FOCUS_SENSOR} (bruto)")
    for ti in incidents:
        ax_raw.axvline(ti, color="red", lw=1.4, alpha=0.85)
    ax_raw.set_ylabel(f"{FOCUS_SENSOR}\n(valor bruto)")
    ax_raw.set_title(
        f"Sistema tier de alertas | warn (q=0.70, k=1) | alarm (q=0.85, k=2) | critical (q=0.95, k=1)\n"
        f"sensor de referência: {FOCUS_SENSOR} | linhas vermelhas = alarmes oficiais do mesmo tag ({len(incidents)})",
        fontsize=12,
    )
    ax_raw.legend(loc="upper right", fontsize="small")

    # 3 painéis para os tiers, cada um como bandas de status
    tier_data = [
        ("WARN", warn, "gold"),
        ("ALARM", alarm, "darkorange"),
        ("CRITICAL", critical, "red"),
    ]
    for ax, (name, mask, color) in zip(axes[1:], tier_data):
        # bandas verticais onde o tier está ativo
        ax.fill_between(t_dt, 0, 1, where=mask, color=color, alpha=0.7, step="mid")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        pct = 100 * mask.mean()
        ax.set_ylabel(f"{name}\n({pct:.0f}%)", rotation=0, ha="right", va="center", fontsize=10)
        # incidentes oficiais
        for ti in incidents:
            ax.axvline(ti, color="red", lw=1.0, alpha=0.4)

    axes[-1].set_xlabel("tempo")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_tier_panorama_ano.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[VIZ] salvo fig_tier_panorama_ano.png")

    # ===== PLOT 2: zoom num incidente com bom lead-up =====
    if len(incidents):
        # acha incidente onde critical disparou nas 72h antes
        DAYS_BEFORE = 7; DAYS_AFTER = 1
        best_ti = None; best_critical_count = -1
        for ti in incidents:
            t0 = pd.Timestamp(ti) - pd.Timedelta(days=3)
            t1 = pd.Timestamp(ti)
            mask_pre = (t_end_pd >= t0) & (t_end_pd <= t1)
            crit_count = int(critical[mask_pre].sum())
            if crit_count > best_critical_count:
                best_critical_count = crit_count; best_ti = ti
        ti = best_ti
        print(f"[VIZ] zoom em {ti} (critical: {best_critical_count} pontos nas 72h antes)")
        t0 = pd.Timestamp(ti) - pd.Timedelta(days=DAYS_BEFORE)
        t1 = pd.Timestamp(ti) + pd.Timedelta(days=DAYS_AFTER)
        mask_raw_w = (df[TIME_COL] >= t0) & (df[TIME_COL] <= t1)
        mask_seq_w = (t_end_pd >= t0) & (t_end_pd <= t1)

        fig, axes = plt.subplots(4, 1, figsize=(11, 5.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 0.7, 0.7, 0.7]})
        t_w = df.loc[mask_raw_w, TIME_COL].to_numpy()
        r_w = raw[mask_raw_w.to_numpy()]
        op_w = op[mask_raw_w.to_numpy()]
        t_seq_w = np.array(t_dt)[mask_seq_w]

        ax_raw = axes[0]
        off_w = op_w < 0.5
        if off_w.any():
            ymin_w = float(np.nanmin(r_w)) if not np.all(np.isnan(r_w)) else 0
            ymax_w = float(np.nanmax(r_w)) if not np.all(np.isnan(r_w)) else 1
            ax_raw.fill_between(t_w, ymin_w, ymax_w, where=off_w, color="lightgray", alpha=0.35,
                                step="mid")
        ax_raw.plot(t_w, r_w, lw=0.7, color="steelblue", label=f"{FOCUS_SENSOR}")
        ax_raw.axvline(ti, color="red", lw=2, label="alarme oficial")
        ax_raw.set_ylabel(f"{FOCUS_SENSOR}")
        ax_raw.set_title(
            f"Zoom em incidente {pd.Timestamp(ti):%Y-%m-%d %H:%M} | escalada warn → alarm → critical",
            fontsize=12,
        )
        ax_raw.legend(loc="upper left", fontsize="small")

        for ax, (name, mask_full, color) in zip(axes[1:], tier_data):
            mask_w = mask_full[mask_seq_w]
            ax.fill_between(t_seq_w, 0, 1, where=mask_w, color=color, alpha=0.7, step="mid")
            ax.set_ylim(0, 1); ax.set_yticks([])
            pct_w = 100 * mask_w.mean() if mask_w.size else 0
            ax.set_ylabel(f"{name}\n({pct_w:.0f}%)", rotation=0, ha="right", va="center", fontsize=10)
            ax.axvline(ti, color="red", lw=2, alpha=0.7)

        axes[-1].set_xlabel("tempo")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/fig_tier_zoom_incidente.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        print("[VIZ] salvo fig_tier_zoom_incidente.png")


if __name__ == "__main__":
    main()
