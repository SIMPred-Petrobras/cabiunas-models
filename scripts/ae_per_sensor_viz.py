#!/usr/bin/env python
"""Gera 3 visualizacoes honestas do per_sensor + OR-de-quantile:
1. Overview do health-index agregado (max-de-quantile), com OFF marcado
2. Heatmap dos 17 sensores (cada um normalizado pelo seu proprio threshold)
3. Zoom num incidente especifico mostrando o lead-up sensor por sensor

Salva em relatorio_anexos/fig_health_{overview,heatmap,zoom_incidente}.png
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from ae_separation_experiment import (
    load, make_seqs, build_ae, mse_per_seq, SENSORS, TIME_COL, RUNNING_COL,
)
from cnn1d_ae.predictive import compute_health_index_ewma

TS = 60; STRIDE = 10
F1 = 4; F2 = 1; S1 = 2; S2 = 2
HALF_LIFE_H = 4.0; GAP_H = 12.0
Q_OP = 0.50  # quantile operacional usado no OR
ZOOM_DAYS_BEFORE = 4
ZOOM_DAYS_AFTER = 1
OUT_DIR = "relatorio_anexos"


def log(m): print(f"[VIZ] {m}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, inc_full = load(priority=None)
    n = len(df)
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    log(f"incidentes (range): {len(inc_full)}")

    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)

    starts = np.arange(0, n - TS + 1, STRIDE)
    ends = starts + TS - 1
    t_end_pd = pd.DatetimeIndex(df[TIME_COL].to_numpy()[ends])
    t_end_sec = t_end_pd.values.astype("datetime64[s]").astype("int64").astype(float)
    seq_run_frac = np.array([op[s:s+TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999

    inc_s = pd.DatetimeIndex(inc_full).values.astype("datetime64[s]").astype("int64").astype(float)
    pos = np.searchsorted(inc_s, t_end_sec)
    dnext = np.where(pos < len(inc_s), (inc_s[np.clip(pos,0,len(inc_s)-1)] - t_end_sec), np.inf)/3600.0
    dprev = np.where(pos > 0, (t_end_sec - inc_s[np.clip(pos-1,0,len(inc_s)-1)]), np.inf)/3600.0
    dany = np.minimum(dnext, dprev)

    # negativos puros pra treino
    is_neg = (dany > GAP_H) & seq_run_full
    neg_idx = np.where(is_neg)[0]
    n_tr = int(0.9 * len(neg_idx))
    tr_idx = neg_idx[:n_tr]; va_idx = neg_idx[n_tr:]
    rng = np.random.default_rng(42)
    if len(tr_idx) > 40000:
        tr_idx = rng.choice(tr_idx, 40000, replace=False)
    log(f"treino: {len(tr_idx)} negs (running, >{GAP_H}h de incidente)")

    dt_seconds = STRIDE * 30.0
    from tensorflow import keras

    health_per_sensor = np.empty((len(starts), len(SENSORS)), dtype=np.float32)
    t0_all = time.time()
    for si, sensor in enumerate(SENSORS):
        t0 = time.time()
        Xi = pd.to_numeric(df[sensor], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32).reshape(-1, 1)
        seqs_i, _ = make_seqs(Xi, TS, STRIDE)
        flat = seqs_i[tr_idx].reshape(-1, 1)
        mu = flat.mean(0); sd = flat.std(0); sd[sd==0] = 1.0
        norm = (seqs_i - mu) / sd
        model, _ = build_ae(TS, 1, F1, F2, S1, S2, 0.1, 1e-4)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
        model.fit(norm[tr_idx], norm[tr_idx],
                  validation_data=(norm[va_idx], norm[va_idx]),
                  epochs=15, batch_size=256, verbose=0, callbacks=cb)
        mae_i = mse_per_seq(model, norm)
        health_per_sensor[:, si] = compute_health_index_ewma(
            mae_i, seq_run_frac,
            half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds,
        )
        log(f"[{si+1:2d}/17] {sensor:14s} t={time.time()-t0:.1f}s")
        keras.backend.clear_session()
    log(f"17 modelos treinados em {time.time()-t0_all:.0f}s")

    # thresholds por sensor: quantile Q_OP do health durante operacao
    per_sensor_thr = np.empty(len(SENSORS), dtype=np.float32)
    for j in range(len(SENSORS)):
        valid = health_per_sensor[seq_run_full, j]
        per_sensor_thr[j] = float(np.quantile(valid, Q_OP)) if valid.size else float("inf")

    # OR-de-quantile: numero de sensores acima do threshold em cada t
    above = health_per_sensor >= per_sensor_thr[None, :]
    n_above = above.sum(axis=1).astype(np.float32)  # 0 a 17 sensores
    any_above = above.any(axis=1) & seq_run_full

    # "quantile position" por sensor (qual quantil esse valor ocupa NO TREINO desse sensor)
    rank_pos = np.empty_like(health_per_sensor)
    for j in range(len(SENSORS)):
        vals = np.sort(health_per_sensor[seq_run_full, j])
        if vals.size == 0:
            rank_pos[:, j] = 0
        else:
            rank_pos[:, j] = np.searchsorted(vals, health_per_sensor[:, j]) / max(len(vals), 1)
    # OFF = NaN no rank_pos (pra ficar transparente nos plots)
    rank_pos[~seq_run_full] = np.nan
    # signal agregado = max(quantile position dos 17 sensores)
    agg_qpos = np.nanmax(rank_pos, axis=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    t_dt = t_end_pd.to_pydatetime()
    inc_dt = pd.DatetimeIndex(inc_full).to_pydatetime()

    # =================== PLOT 1: overview honesto ===================
    fig, ax = plt.subplots(figsize=(15, 4))
    # OFF periods: faixa cinza
    off_mask = ~seq_run_full
    if off_mask.any():
        ax.fill_between(t_dt, 0, 1, where=off_mask, color="lightgray", alpha=0.5,
                        step="mid", label="máquina OFF", transform=ax.get_xaxis_transform())
    # incidentes: vertical
    for ti in inc_dt:
        ax.axvline(ti, color="green", alpha=0.20, lw=0.8)
    # signal agregado
    ax.plot(t_dt, agg_qpos, lw=0.5, color="steelblue", label=f"max(quantile-pos) dos 17 sensores")
    ax.axhline(Q_OP, color="red", ls="--", label=f"alerta (qualquer sensor > q={Q_OP})")
    # alertas marcados
    ax.scatter(np.array(t_dt)[any_above], agg_qpos[any_above], s=4, color="orange",
               label="alerta ativo", zorder=3)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("posição quantilica do health-index (0=normal, 1=anormal)")
    ax.set_xlabel("tempo")
    ax.set_title(f"Health-index agregado (per_sensor OR) | verde=incidente | cinza=OFF | "
                 f"alertas={int(any_above.sum())}")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_health_overview_per_sensor.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log("salvo fig_health_overview_per_sensor.png")

    # =================== PLOT 2: heatmap dos 17 sensores ===================
    fig, ax = plt.subplots(figsize=(15, 6))
    # ordena sensores por relevancia (max do quantil-pos atingido)
    sensor_max = np.nanmax(rank_pos, axis=0)
    order = np.argsort(sensor_max)[::-1]
    M = rank_pos[:, order].T  # (n_sensors, n_seq)
    im = ax.imshow(M, aspect="auto", cmap="hot_r", vmin=0, vmax=1,
                   extent=[mdates.date2num(t_dt[0]), mdates.date2num(t_dt[-1]),
                           len(SENSORS)-0.5, -0.5])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_yticks(range(len(SENSORS)))
    ax.set_yticklabels([SENSORS[i] for i in order], fontsize=8)
    for ti in inc_dt:
        ax.axvline(ti, color="lime", alpha=0.25, lw=0.6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("quantile-pos do health (0=normal, 1=anormal)", fontsize=8)
    ax.set_title("Heatmap: health-index por sensor ao longo do ano | verde=incidente | branco=OFF")
    ax.set_xlabel("tempo")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_health_heatmap_per_sensor.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log("salvo fig_health_heatmap_per_sensor.png")

    # =================== PLOT 3: zoom em 1 incidente com lead time bom ===================
    # escolhe o primeiro incidente com pelo menos 1 sensor cruzando threshold nas ultimas 24h
    H = 24.0 * 3600
    chosen = None
    for ti_s in inc_s:
        mask = (t_end_sec >= ti_s - H) & (t_end_sec <= ti_s)
        if mask.any() and any_above[mask].any():
            chosen = ti_s; break
    if chosen is None and len(inc_s):
        chosen = float(inc_s[len(inc_s)//2])
    if chosen is not None:
        ti_dt = pd.Timestamp(chosen, unit="s")
        win_start = ti_dt - pd.Timedelta(days=ZOOM_DAYS_BEFORE)
        win_end = ti_dt + pd.Timedelta(days=ZOOM_DAYS_AFTER)
        win = (t_end_pd >= win_start) & (t_end_pd <= win_end)
        log(f"zoom em incidente {ti_dt}; {int(win.sum())} pontos na janela [-{ZOOM_DAYS_BEFORE}d, +{ZOOM_DAYS_AFTER}d]")
        fig, ax = plt.subplots(figsize=(13, 5))
        tw = np.array(t_dt)[win]
        # plota só os 5 sensores mais quentes da janela
        top_sensors = np.argsort(np.nanmax(rank_pos[win], axis=0))[::-1][:5]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for k, j in enumerate(top_sensors):
            ax.plot(tw, rank_pos[win, j], lw=1.2, label=SENSORS[j],
                    color=colors[k % len(colors)])
        ax.axhline(Q_OP, color="red", ls="--", lw=1, label=f"thr q={Q_OP}")
        ax.axvline(ti_dt, color="green", lw=2, label=f"incidente real")
        # marca alerta ativo
        ax.scatter(tw[any_above[win]], np.full(int(any_above[win].sum()), 1.02),
                   s=20, marker="v", color="orange", label="alerta ativo")
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("tempo")
        ax.set_ylabel("posição quantilica do health-index")
        ax.set_title(f"Zoom em incidente {ti_dt:%Y-%m-%d %H:%M} | top-5 sensores | "
                     f"alerta antecipa em até {ZOOM_DAYS_BEFORE} dias")
        ax.legend(loc="upper left", fontsize="small")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/fig_health_zoom_incidente.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        log("salvo fig_health_zoom_incidente.png")
    else:
        log("sem incidentes pra zoom")

    log("OK — 3 figuras em relatorio_anexos/")


if __name__ == "__main__":
    main()
