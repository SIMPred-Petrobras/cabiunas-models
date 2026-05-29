#!/usr/bin/env python
"""Plot final do estado validado: intensidade ao longo do ano com split train/test."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import load, SENSORS, TIME_COL, RUNNING_COL
from cnn1d_ae.predictive import compute_health_index_ewma, _detect_episodes

TS = 60; STRIDE = 10
HL = 4.0; Q_OP = 0.715; GAP_H = 12.0; DEBOUNCE_H = 8.0; HORIZON = 8.0
TRAIN_FRAC = 0.66
OUT_DIR = "relatorio_anexos"
MAE_CACHE = "improve_halflife_out/mae_per_sensor_cache.npz"


def evaluate(alert, t_end_sec, inc_s, span_days):
    H = HORIZON * 3600; deb = DEBOUNCE_H * 3600
    idx = np.where(alert)[0]
    episodes = _detect_episodes(idx, t_end_sec, deb)
    alert_s = t_end_sec[idx]
    hits = 0
    for ti in inc_s:
        w = alert_s[(alert_s >= ti - H) & (alert_s <= ti)]
        if w.size: hits += 1
    fa = 0
    for (s0, s1) in episodes:
        if not bool((((inc_s - H) <= s1) & (inc_s >= s0)).any()): fa += 1
    return dict(recall=hits/max(len(inc_s),1), fa_per_day=fa/max(span_days,1e-9),
                n_episodes=len(episodes), n_incidents=len(inc_s))


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
    inc_s_all = pd.DatetimeIndex(inc_full).values.astype("datetime64[s]").astype("int64").astype(float)
    dt_seconds = STRIDE * 30.0

    print(f"[VIZ] carregando MAE cache: {MAE_CACHE}")
    mae = np.load(MAE_CACHE)["mae"]
    print(f"[VIZ] mae shape: {mae.shape}")

    n_sens = mae.shape[1]
    health = np.empty_like(mae)
    for j in range(n_sens):
        health[:, j] = compute_health_index_ewma(
            mae[:, j], seq_run_frac, half_life_hours=HL, dt_seconds=dt_seconds,
        )

    # split temporal
    n_train_pts = int(TRAIN_FRAC * n)
    t_split = df[TIME_COL].iloc[n_train_pts]
    is_train = t_end_pd <= t_split
    is_test = ~is_train
    inc_tr_s = inc_s_all[inc_s_all <= pd.Timestamp(t_split).value//10**9]
    inc_te_s = inc_s_all[inc_s_all > pd.Timestamp(t_split).value//10**9]
    span_days_tr = (t_end_sec[is_train].max() - t_end_sec[is_train].min())/86400
    span_days_te = (t_end_sec[is_test].max() - t_end_sec[is_test].min())/86400

    # thresholds CALCULADOS EM TRAIN
    thr = np.array([float(np.quantile(health[is_train & seq_run_full, j], Q_OP))
                     for j in range(n_sens)])
    above = (health >= thr[None, :])
    n_above = above.sum(axis=1).astype(np.float32)
    n_above[~seq_run_full] = np.nan
    alert_full = above.any(axis=1) & seq_run_full

    # metricas
    alert_tr = alert_full[is_train]
    alert_te = alert_full[is_test]
    m_tr = evaluate(alert_tr, t_end_sec[is_train], inc_tr_s, span_days_tr)
    m_te = evaluate(alert_te, t_end_sec[is_test], inc_te_s, span_days_te)
    print(f"[VIZ] TRAIN (jan-ago, {len(inc_tr_s)} incidentes):")
    print(f"  recall={m_tr['recall']:.2f} FA/d={m_tr['fa_per_day']:.3f} eps={m_tr['n_episodes']}")
    print(f"[VIZ] TEST (set-dez, {len(inc_te_s)} incidentes):")
    print(f"  recall={m_te['recall']:.2f} FA/d={m_te['fa_per_day']:.3f} eps={m_te['n_episodes']}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    expected = 17 * (1 - Q_OP)
    alert_thr = expected * 2

    t_dt = t_end_pd.to_pydatetime()
    inc_dt = pd.DatetimeIndex(inc_full).to_pydatetime()
    t_split_dt = pd.Timestamp(t_split).to_pydatetime()

    fig, ax = plt.subplots(figsize=(16, 5))
    off_mask = ~seq_run_full
    if off_mask.any():
        ax.fill_between(t_dt, 0, 17, where=off_mask, color="lightgray", alpha=0.4,
                        step="mid", label="máquina OFF")
    # incidentes
    for ti in inc_dt:
        ax.axvline(ti, color="green", alpha=0.15, lw=0.6)
    # baseline e threshold
    ax.axhline(expected, color="gray", ls=":", lw=1, label=f"esperado por chance ({expected:.1f})")
    ax.axhline(alert_thr, color="red", ls="--", lw=1, label=f"alerta forte (≥{alert_thr:.0f})")
    # signal
    ax.plot(t_dt, n_above, lw=0.4, color="steelblue")
    # alertas em train (cor azul claro) vs test (cor laranja)
    strong = n_above >= alert_thr
    strong_tr = strong & is_train
    strong_te = strong & is_test
    ax.scatter(np.array(t_dt)[strong_tr], n_above[strong_tr], s=3, color="dodgerblue",
               label="alerta forte (treino)", zorder=3)
    ax.scatter(np.array(t_dt)[strong_te], n_above[strong_te], s=3, color="orangered",
               label="alerta forte (teste OOS)", zorder=3)
    # linha de split
    ax.axvline(t_split_dt, color="purple", lw=2, alpha=0.8, label="train/test split")
    ax.set_ylim(-0.5, 17.5)
    ax.set_ylabel("n. sensores acima do threshold individual")
    ax.set_xlabel("tempo")
    ax.set_title(
        f"Validação temporal honesta | thresholds calibrados em jan-ago, aplicados em set-dez\n"
        f"TRAIN: recall={m_tr['recall']:.0%} FA/d={m_tr['fa_per_day']:.3f} ({m_tr['n_episodes']} eps) | "
        f"TEST: recall={m_te['recall']:.0%} FA/d={m_te['fa_per_day']:.3f} ({m_te['n_episodes']} eps)"
    )
    ax.legend(loc="upper right", fontsize="small", ncols=2)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_temporal_validation_final.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] salvo {OUT_DIR}/fig_temporal_validation_final.png")


if __name__ == "__main__":
    main()
