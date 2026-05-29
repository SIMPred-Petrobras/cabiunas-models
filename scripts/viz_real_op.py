#!/usr/bin/env python
"""Visualizacoes no PONTO OPERACIONAL REAL (q=0.715).
- Overview: numero de sensores acima do threshold (mostra intensidade real)
- Zoom: 1 incidente com lead-up sensor por sensor
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
Q_OP = 0.715  # ponto operacional real (FA/dia 0.03, recall 0.67 a H=8h)
OUT_DIR = "relatorio_anexos"


def log(m): print(f"[VIZ-R] {m}", flush=True)


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
    inc_s = pd.DatetimeIndex(inc_full).values.astype("datetime64[s]").astype("int64").astype(float)
    pos = np.searchsorted(inc_s, t_end_sec)
    dnext = np.where(pos < len(inc_s), (inc_s[np.clip(pos,0,len(inc_s)-1)] - t_end_sec), np.inf)/3600.0
    dprev = np.where(pos > 0, (t_end_sec - inc_s[np.clip(pos-1,0,len(inc_s)-1)]), np.inf)/3600.0
    dany = np.minimum(dnext, dprev)
    is_neg = (dany > GAP_H) & seq_run_full
    neg_idx = np.where(is_neg)[0]
    rng = np.random.default_rng(42)
    n_tr = int(0.9 * len(neg_idx))
    tr_pool = neg_idx[:n_tr]; va_idx = neg_idx[n_tr:]
    if len(tr_pool) > 40000:
        tr_pool = rng.choice(tr_pool, 40000, replace=False)
    log(f"treino: {len(tr_pool)} negs")
    dt_seconds = STRIDE * 30.0
    from tensorflow import keras

    health_ps = np.empty((len(starts), len(SENSORS)), dtype=np.float32)
    t0 = time.time()
    for si, sensor in enumerate(SENSORS):
        Xi = pd.to_numeric(df[sensor], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32).reshape(-1, 1)
        seqs_i, _ = make_seqs(Xi, TS, STRIDE)
        flat = seqs_i[tr_pool].reshape(-1, 1)
        mu = flat.mean(0); sd = flat.std(0); sd[sd==0] = 1.0
        norm = (seqs_i - mu) / sd
        model, _ = build_ae(TS, 1, F1, F2, S1, S2, 0.1, 1e-4)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
        model.fit(norm[tr_pool], norm[tr_pool], validation_data=(norm[va_idx], norm[va_idx]),
                  epochs=15, batch_size=256, verbose=0, callbacks=cb)
        mae_i = mse_per_seq(model, norm)
        health_ps[:, si] = compute_health_index_ewma(
            mae_i, seq_run_frac, half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds,
        )
        keras.backend.clear_session()
        if (si+1) % 5 == 0: log(f"  {si+1}/17")
    log(f"17 modelos em {time.time()-t0:.0f}s")

    # thresholds por sensor no quantil Q_OP do RUNNING
    per_sensor_thr = np.empty(len(SENSORS), dtype=np.float32)
    for j in range(len(SENSORS)):
        valid = health_ps[seq_run_full, j]
        per_sensor_thr[j] = float(np.quantile(valid, Q_OP))
    above = (health_ps >= per_sensor_thr[None, :])
    n_above = above.sum(axis=1).astype(np.float32)  # 0..17
    n_above[~seq_run_full] = np.nan
    any_above = above.any(axis=1) & seq_run_full
    log(f"q_op={Q_OP} → media n_sensores acima na operacao = {np.nanmean(n_above):.1f}/17")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    t_dt = t_end_pd.to_pydatetime()
    inc_dt = pd.DatetimeIndex(inc_full).to_pydatetime()

    # ===== PLOT 1: numero de sensores acima do threshold (INTENSIDADE) =====
    fig, ax = plt.subplots(figsize=(15, 4))
    off_mask = ~seq_run_full
    if off_mask.any():
        ax.fill_between(t_dt, 0, 17, where=off_mask, color="lightgray", alpha=0.5, step="mid",
                        label="máquina OFF")
    # baseline esperado: q_op=0.715 → cada sensor 28.5% prob → media 17*0.285 = 4.8
    expected = 17 * (1 - Q_OP)
    ax.axhline(expected, color="gray", ls=":", lw=1, label=f"esperado por chance ({expected:.1f})")
    # threshold de alerta: 2x acima do esperado = sinal real de degradacao
    alert_thr = expected * 2
    ax.axhline(alert_thr, color="red", ls="--", lw=1, label=f"alerta forte (≥{alert_thr:.0f} sensores)")
    # signal
    ax.plot(t_dt, n_above, lw=0.4, color="steelblue", label=f"n. sensores acima do quantile {Q_OP:.2f}")
    # alertas fortes
    strong = n_above >= alert_thr
    ax.scatter(np.array(t_dt)[strong], n_above[strong], s=3, color="orange", label="alerta forte ativo", zorder=3)
    # incidentes
    for ti in inc_dt:
        ax.axvline(ti, color="green", alpha=0.18, lw=0.7)
    ax.set_ylim(-0.5, 17.5)
    ax.set_ylabel("n. sensores com health > quantile op")
    ax.set_xlabel("tempo")
    ax.set_title(f"Intensidade do alerta (per_sensor, q_op={Q_OP}) | "
                 f"verde=incidente | cinza=OFF | n_alertas_fortes={int(strong.sum())}")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_health_intensidade_per_sensor.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log("salvo fig_health_intensidade_per_sensor.png")

    # ===== PLOT 2: zoom 4 dias antes ate 1 dia depois de um incidente bom =====
    # acha incidente com pico de n_above >= alert_thr nas 48h antes
    H_pre = 48 * 3600
    best_score = -1; chosen = None
    for ti_s in inc_s:
        mask = (t_end_sec >= ti_s - H_pre) & (t_end_sec <= ti_s) & seq_run_full
        if not mask.any(): continue
        peak = np.nanmax(n_above[mask])
        if peak > best_score:
            best_score = peak; chosen = ti_s
    if chosen is not None:
        ti_dt = pd.Timestamp(chosen, unit="s")
        win_start = ti_dt - pd.Timedelta(days=4)
        win_end = ti_dt + pd.Timedelta(days=1)
        win = (t_end_pd >= win_start) & (t_end_pd <= win_end)
        log(f"zoom em {ti_dt} (pico de {best_score:.0f}/17 sensores nas 48h antes)")

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 2]})
        tw = np.array(t_dt)[win]
        # painel topo: contagem n_above
        axes[0].plot(tw, n_above[win], lw=1, color="steelblue", label="n. sensores acima")
        axes[0].axhline(expected, color="gray", ls=":", lw=1, label=f"chance ({expected:.1f})")
        axes[0].axhline(alert_thr, color="red", ls="--", lw=1, label=f"alerta (≥{alert_thr:.0f})")
        axes[0].axvline(ti_dt, color="green", lw=2, label="incidente real")
        axes[0].fill_between(tw, 0, alert_thr, where=(n_above[win] >= alert_thr), color="orange", alpha=0.3,
                             step="mid")
        axes[0].set_ylim(0, 17.5)
        axes[0].set_ylabel("n. sensores")
        axes[0].set_title(f"Lead-up para incidente {ti_dt:%Y-%m-%d %H:%M}")
        axes[0].legend(loc="upper left", fontsize="small")
        # painel base: top 6 sensores na janela
        top_sensors = np.argsort(np.nanmax(health_ps[win] / per_sensor_thr[None, :], axis=0))[::-1][:6]
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for k, j in enumerate(top_sensors):
            # razao health/thr — >1 significa "acima do seu threshold"
            ratio = health_ps[win, j] / max(per_sensor_thr[j], 1e-9)
            axes[1].plot(tw, ratio, lw=1.2, label=f"{SENSORS[j]} (thr={per_sensor_thr[j]:.3f})",
                         color=colors[k % len(colors)])
        axes[1].axhline(1.0, color="red", ls="--", lw=1, label="threshold individual")
        axes[1].axvline(ti_dt, color="green", lw=2)
        axes[1].set_ylabel("health / threshold (>1 = acima)")
        axes[1].set_xlabel("tempo")
        axes[1].set_ylim(0, max(2.5, axes[1].get_ylim()[1]))
        axes[1].legend(loc="upper left", fontsize="small", ncols=2)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/fig_health_zoom_real_op.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        log("salvo fig_health_zoom_real_op.png")

    log("OK")


if __name__ == "__main__":
    main()
