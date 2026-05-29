#!/usr/bin/env python
"""Plot focado: TC382_03_A serie + anomalias detectadas + alarmes do proprio tag.

1 painel grande: ano inteiro com sensor bruto, anomalias e alarmes
N paineis menores: zoom em cada incidente real do tag TC382_03_A
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import load, SENSORS, TIME_COL, RUNNING_COL
from cnn1d_ae.predictive import compute_health_index_ewma

SENSOR = "TC382_03_A"
TS = 60; STRIDE = 10
HL = 4.0; Q_OP = 0.715
DAYS_BEFORE_ZOOM = 5
DAYS_AFTER_ZOOM = 1
OUT_DIR = "relatorio_anexos"
MAE_CACHE = "improve_halflife_out/mae_per_sensor_cache.npz"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, _ = load(priority=None)  # nao filtra incidentes — usa alarmes raw
    n = len(df)
    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)

    # Alarmes do TAG TC382_03_A (proprio do sensor)
    df_alarm = pd.read_csv("../dados/alarmes_selecionados_turbina_a.csv")
    df_alarm["t"] = pd.to_datetime(df_alarm["Data da Ocorrência"], errors="coerce")
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    mask_2025 = (df_alarm["t"] >= t_min) & (df_alarm["t"] <= t_max)
    mask_sensor = df_alarm["Tag Alarme"].astype(str) == SENSOR
    mask_not_clear = df_alarm["Condição do Alarme"].astype(str) != "OK"
    alarms_sensor = df_alarm[mask_2025 & mask_sensor & mask_not_clear].copy()
    print(f"[VIZ] alarmes (onsets) do tag {SENSOR} em 2025: {len(alarms_sensor)}")
    # dedup em incidentes (gap > 4h)
    alarms_sensor = alarms_sensor.sort_values("t").reset_index(drop=True)
    grp = (alarms_sensor["t"].diff().dt.total_seconds()/3600 > 4).cumsum()
    incidents_sensor = alarms_sensor.groupby(grp)["t"].min().reset_index(drop=True)
    print(f"[VIZ] incidentes dedup (gap>4h) do {SENSOR}: {len(incidents_sensor)}")

    # Health do sensor
    starts = np.arange(0, n - TS + 1, STRIDE)
    ends = starts + TS - 1
    t_end_pd = pd.DatetimeIndex(df[TIME_COL].to_numpy()[ends])
    seq_run_frac = np.array([op[s:s+TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999
    mae = np.load(MAE_CACHE)["mae"]
    sensor_idx = SENSORS.index(SENSOR)
    dt_seconds = STRIDE * 30.0
    health = compute_health_index_ewma(
        mae[:, sensor_idx], seq_run_frac,
        half_life_hours=HL, dt_seconds=dt_seconds,
    )
    thr = float(np.quantile(health[seq_run_full], Q_OP))
    above = (health >= thr) & seq_run_full
    print(f"[VIZ] threshold individual: {thr:.4f} | total acima na operacao: "
          f"{100*above.sum()/max(seq_run_full.sum(),1):.1f}%")

    # Valor bruto
    raw = pd.to_numeric(df[SENSOR], errors="coerce").to_numpy()
    t_raw = df[TIME_COL].to_numpy()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # =================== PLOT 1: ano inteiro ===================
    fig, ax = plt.subplots(figsize=(18, 5))
    # OFF como faixa cinza
    off_mask = op < 0.5
    ymin = float(np.nanmin(raw)) if not np.all(np.isnan(raw)) else 0
    ymax = float(np.nanmax(raw)) if not np.all(np.isnan(raw)) else 1
    if off_mask.any():
        ax.fill_between(t_raw, ymin, ymax, where=off_mask, color="lightgray", alpha=0.35,
                        step="mid", label="máquina OFF")
    # anomalias detectadas (faixa laranja)
    t_seq = t_end_pd.to_numpy()
    for i in range(len(above)-1):
        if above[i]:
            ax.axvspan(t_seq[i], t_seq[i+1] if i+1<len(t_seq) else t_seq[i],
                       color="orange", alpha=0.18)
    # série bruta
    ax.plot(t_raw, raw, lw=0.4, color="steelblue", label=f"{SENSOR} (valor bruto)")
    # alarmes do próprio tag
    for ti in incidents_sensor:
        ax.axvline(ti, color="red", lw=1.2, alpha=0.85)
    # legenda via proxy artists (sem afetar limites do eixo)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(color="lightgray", alpha=0.35, label="máquina OFF"),
        Line2D([], [], color="steelblue", lw=1.0, label=f"{SENSOR} (valor bruto)"),
        Patch(color="orange", alpha=0.4, label="AE detectou anomalia"),
        Line2D([], [], color="red", lw=1.5, label=f"alarme oficial do {SENSOR}"),
    ]
    ax.set_ylabel(f"{SENSOR}\n(valor bruto)")
    ax.set_xlabel("tempo")
    ax.set_title(
        f"Serie temporal do {SENSOR} ao longo de 2025\n"
        f"laranja = AE detectou anomalia | vermelho = alarme oficial do mesmo tag "
        f"({len(incidents_sensor)} incidentes) | cinza = máquina OFF",
        fontsize=12,
    )
    ax.legend(handles=handles, loc="upper right", fontsize="small")
    # garante limites do eixo no range dos dados (anti-1970)
    ax.set_xlim(t_raw[0], t_raw[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    out1 = f"{OUT_DIR}/fig_serie_{SENSOR}_ano.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] salvo {out1}")

    # =================== PLOT 2: zoom em cada incidente ===================
    n_inc = len(incidents_sensor)
    if n_inc == 0:
        print("[VIZ] sem incidentes pra zoom")
        return
    # grid de zooms
    ncols = 2
    nrows = (n_inc + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2*nrows), squeeze=False)
    for k, ti in enumerate(incidents_sensor):
        r, c = k // ncols, k % ncols
        ax = axes[r][c]
        t0 = ti - pd.Timedelta(days=DAYS_BEFORE_ZOOM)
        t1 = ti + pd.Timedelta(days=DAYS_AFTER_ZOOM)
        mask_raw = (df[TIME_COL] >= t0) & (df[TIME_COL] <= t1)
        mask_seq = (t_end_pd >= t0) & (t_end_pd <= t1)
        if not mask_raw.any():
            ax.set_visible(False); continue
        mask_raw_np = mask_raw.to_numpy() if hasattr(mask_raw, "to_numpy") else mask_raw
        t_w = df.loc[mask_raw, TIME_COL].to_numpy()
        r_w = raw[mask_raw_np]
        op_w = op[mask_raw_np]
        # OFF
        ymin_w = float(np.nanmin(r_w)) if not np.all(np.isnan(r_w)) else 0
        ymax_w = float(np.nanmax(r_w)) if not np.all(np.isnan(r_w)) else 1
        off_w = op_w < 0.5
        if off_w.any():
            ax.fill_between(t_w, ymin_w, ymax_w, where=off_w, color="lightgray", alpha=0.35,
                            step="mid")
        # anomalias na janela
        mask_seq_np = mask_seq.to_numpy() if hasattr(mask_seq, "to_numpy") else mask_seq
        idx_seq_w = np.where(mask_seq_np)[0]
        for i in idx_seq_w:
            if above[i] and i+1 < len(t_seq):
                ax.axvspan(t_seq[i], t_seq[i+1], color="orange", alpha=0.25)
        ax.plot(t_w, r_w, lw=0.7, color="steelblue")
        ax.axvline(ti, color="red", lw=2, alpha=0.85)
        # quantos % do periodo anterior em que AE estava em anomalia
        idx_pre = idx_seq_w[t_seq[idx_seq_w] < pd.Timestamp(ti).to_datetime64()]
        if len(idx_pre):
            pct_pre = 100 * above[idx_pre].mean()
        else:
            pct_pre = 0
        ax.set_title(f"incidente {ti:%Y-%m-%d %H:%M} | AE em anomalia: "
                     f"{pct_pre:.0f}% dos {DAYS_BEFORE_ZOOM} dias antes", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    # remove subplots vazios
    for k in range(n_inc, nrows*ncols):
        r, c = k // ncols, k % ncols
        axes[r][c].set_visible(False)
    fig.suptitle(f"Zoom em cada incidente do {SENSOR} (5 dias antes a 1 depois)\n"
                 f"laranja = AE detectou anomalia | vermelho = alarme oficial",
                 fontsize=13)
    fig.tight_layout()
    out2 = f"{OUT_DIR}/fig_serie_{SENSOR}_zoom_incidentes.png"
    fig.savefig(out2, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] salvo {out2}")


if __name__ == "__main__":
    main()
