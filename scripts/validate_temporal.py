#!/usr/bin/env python
"""Validacao temporal HONESTA: treina AE em jan-ago, congela, mede a curva
preditiva em set-dez sem retuning. E o unico teste que separa um sistema preditivo
real de um overfit dressado de validacao.

Saidas em temporal_validation_out/: results.json + per-horizon curve CSVs.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

# imports do projeto
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import (
    load, make_seqs, build_ae, mse_per_seq, SENSORS, TIME_COL, RUNNING_COL,
)
from cnn1d_ae.predictive import (
    compute_health_index_ewma, compute_predictive_curve, pick_operating_point,
)

# Hiperparams: mesmos do experimento vencedor (f2=3, ts=60)
TS=60; STRIDE=10; F1=16; F2=3; S1=2; S2=2
HALF_LIFE_H=4.0; DEBOUNCE_H=8.0; GAP_H=12.0
TRAIN_FRAC=0.66           # ~jan-ago vs set-dez
HORIZONS=[8, 24, 72]
FA_BUDGET=1.0
OUT="temporal_validation_out"


def log(m): print(f"[TVAL] {m}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, inc_full = load(priority=None)
    n = len(df)
    n_train = int(TRAIN_FRAC * n)
    t_split = df[TIME_COL].iloc[n_train]
    log(f"Split em {t_split}")
    log(f"  train: {df[TIME_COL].min()} -> {t_split}")
    log(f"  test : {t_split} -> {df[TIME_COL].max()}")

    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    X = df[SENSORS].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32)

    # incidentes ja deduplicados (gap>4h) por load(); filtro temporal
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    incidents_train = inc_full[inc_full <= t_split]
    incidents_test = inc_full[inc_full > t_split]
    log(f"incidentes (range): train={len(incidents_train)} | test={len(incidents_test)}")

    # sequencias e tempos
    seqs, starts = make_seqs(X, TS, STRIDE)
    ends = starts + TS - 1
    t_end_np = df[TIME_COL].to_numpy()[ends]
    t_end_pd = pd.DatetimeIndex(t_end_np)
    t_end_sec = t_end_pd.values.astype("datetime64[s]").astype("int64").astype(float)
    seq_run_frac = np.array([op[s:s+TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999

    # distancia ao incidente mais proximo (qualquer)
    inc_s_full = pd.DatetimeIndex(inc_full).values.astype("datetime64[s]").astype("int64").astype(float)
    pos = np.searchsorted(inc_s_full, t_end_sec)
    dnext = np.where(pos < len(inc_s_full), (inc_s_full[np.clip(pos,0,len(inc_s_full)-1)] - t_end_sec), np.inf)/3600.0
    dprev = np.where(pos > 0, (t_end_sec - inc_s_full[np.clip(pos-1,0,len(inc_s_full)-1)]), np.inf)/3600.0
    dany = np.minimum(dnext, dprev)

    # mascaras temporais
    is_train = t_end_pd <= t_split
    is_test = ~is_train
    log(f"seqs: train={int(is_train.sum())} | test={int(is_test.sum())}")

    # treino = negativos puros DENTRO do periodo de treino
    is_train_neg = is_train & seq_run_full & (dany > GAP_H)
    tr_pool = np.where(is_train_neg)[0]
    rng = np.random.default_rng(42)
    n_split = int(0.9 * len(tr_pool))
    tr_idx = tr_pool[:n_split]
    va_idx = tr_pool[n_split:]
    if len(tr_idx) > 40000:
        tr_idx = rng.choice(tr_idx, 40000, replace=False)
    log(f"treino: {len(tr_idx)} negs (jan-ago, running, >12h de incidente)")

    # normalizacao FIT no treino
    flat = seqs[tr_idx].reshape(-1, seqs.shape[-1])
    mu = flat.mean(0); sd = flat.std(0); sd[sd==0] = 1.0
    norm = lambda a: (a - mu) / sd

    model, latent = build_ae(TS, seqs.shape[-1], F1, F2, S1, S2, 0.1, 1e-4)
    log(f"latente_ratio={latent/(TS*seqs.shape[-1]):.3f}")
    from tensorflow import keras
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    model.fit(
        norm(seqs[tr_idx]), norm(seqs[tr_idx]),
        validation_data=(norm(seqs[va_idx]), norm(seqs[va_idx])),
        epochs=20, batch_size=256, verbose=2, callbacks=cb,
    )

    # health-index sobre TUDO (usando mu/sd do treino, FROZEN)
    mae_all = mse_per_seq(model, norm(seqs))
    dt_seconds = STRIDE * 30.0
    health = compute_health_index_ewma(
        mae_all, seq_run_frac,
        half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds,
    )

    inc_tr_s = pd.DatetimeIndex(incidents_train).values.astype("datetime64[s]").astype("int64").astype(float)
    inc_te_s = pd.DatetimeIndex(incidents_test).values.astype("datetime64[s]").astype("int64").astype(float)

    results = {
        "split_time": str(t_split),
        "n_train_incidents": int(len(incidents_train)),
        "n_test_incidents": int(len(incidents_test)),
        "train_in_sample": {},
        "test_out_of_sample": {},
        "test_at_train_threshold": {},
    }
    for H in HORIZONS:
        curve_tr = compute_predictive_curve(
            health_ewma=health[is_train],
            seq_running_full=seq_run_full[is_train],
            t_end_seconds=t_end_sec[is_train],
            incident_seconds=inc_tr_s,
            horizon_hours=H, debounce_hours=DEBOUNCE_H,
        )
        op_tr = pick_operating_point(curve_tr, FA_BUDGET)
        curve_te = compute_predictive_curve(
            health_ewma=health[is_test],
            seq_running_full=seq_run_full[is_test],
            t_end_seconds=t_end_sec[is_test],
            incident_seconds=inc_te_s,
            horizon_hours=H, debounce_hours=DEBOUNCE_H,
        )
        op_te = pick_operating_point(curve_te, FA_BUDGET)
        # ponto TESTE no MESMO threshold escolhido em treino (deploy honesto)
        row_at_tr_thr = None
        if op_tr and not curve_te.empty:
            i = (curve_te["threshold"] - op_tr["threshold"]).abs().idxmin()
            row_at_tr_thr = {k: float(v) for k, v in curve_te.loc[i].to_dict().items()}
        results["train_in_sample"][f"H{H}h"] = op_tr
        results["test_out_of_sample"][f"H{H}h"] = op_te
        results["test_at_train_threshold"][f"H{H}h"] = row_at_tr_thr
        # salva curvas
        if curve_tr is not None and not curve_tr.empty:
            curve_tr.to_csv(f"{OUT}/curve_train_H{H}h.csv", index=False)
        if curve_te is not None and not curve_te.empty:
            curve_te.to_csv(f"{OUT}/curve_test_H{H}h.csv", index=False)
        log(f"H={H}h")
        log(f"  TRAIN (in-sample)        : recall={op_tr['recall']:.2f} fa/d={op_tr['fa_per_day']:.2f} lead={op_tr['median_lead_hours']:.1f}h thr={op_tr['threshold']:.4f}")
        log(f"  TEST  (out-of-sample)    : recall={op_te['recall']:.2f} fa/d={op_te['fa_per_day']:.2f} lead={op_te['median_lead_hours']:.1f}h thr={op_te['threshold']:.4f}")
        if row_at_tr_thr:
            log(f"  TEST @ TRAIN thr (deploy): recall={row_at_tr_thr['recall']:.2f} fa/d={row_at_tr_thr['fa_per_day']:.2f}")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"salvo {OUT}/results.json + curve_*.csv")


if __name__ == "__main__":
    main()
