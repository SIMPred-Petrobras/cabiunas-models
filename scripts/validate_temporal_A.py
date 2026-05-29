#!/usr/bin/env python
"""Validacao temporal da Strategy A (per-sensor F1-best): jan-ago vs set-dez.

Teste critico: os thresholds F1-best CALCULADOS em jan-ago sobrevivem em set-dez?
Ou sao overfitting?
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import (
    load, make_seqs, build_ae, mse_per_seq, SENSORS, TIME_COL, RUNNING_COL,
)
from cnn1d_ae.predictive import compute_health_index_ewma, _detect_episodes

TS = 60; STRIDE = 10; F1 = 4; F2 = 1; S1 = 2; S2 = 2
HALF_LIFE_H = 4.0; GAP_H = 12.0; DEBOUNCE_H = 8.0
TRAIN_FRAC = 0.66
HORIZONS = [8, 24]
OUT = "validate_A_out"


def log(m): print(f"[VAL-A] {m}", flush=True)


def f1_best_threshold(s, y):
    if y.sum() == 0: return float("inf")
    qs = np.quantile(s, np.linspace(0.3, 0.999, 60))
    best_f1 = 0.0; best_thr = float(np.median(s))
    for thr in qs:
        pred = s >= thr
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
        if tp == 0: continue
        p = tp/(tp+fp); r = tp/(tp+fn); f = 2*p*r/(p+r)
        if f > best_f1: best_f1 = f; best_thr = float(thr)
    return best_thr


def evaluate_alerts(alert_seq, t_end_sec, inc_s, horizon_h, debounce_h, span_days):
    H = horizon_h * 3600; deb = debounce_h * 3600
    idx = np.where(alert_seq)[0]
    episodes = _detect_episodes(idx, t_end_sec, deb)
    alert_s = t_end_sec[idx]
    hits = 0; leads = []
    for ti in inc_s:
        w = alert_s[(alert_s >= ti - H) & (alert_s <= ti)]
        if w.size: hits += 1; leads.append((ti - w.min()) / 3600.0)
    recall = hits / len(inc_s) if len(inc_s) else 0.0
    fa = 0
    for (s0, s1) in episodes:
        useful = bool((((inc_s - H) <= s1) & (inc_s >= s0)).any()) if inc_s.size else False
        if not useful: fa += 1
    return dict(recall=float(recall), fa_per_day=float(fa/max(span_days,1e-9)),
                median_lead_hours=float(np.median(leads)) if leads else 0.0,
                n_episodes=int(len(episodes)))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    df, inc_full = load(priority=None)
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    n = len(df)
    n_train_pts = int(TRAIN_FRAC * n)
    t_split = df[TIME_COL].iloc[n_train_pts]
    log(f"split: {t_split}")
    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    starts = np.arange(0, n - TS + 1, STRIDE)
    ends = starts + TS - 1
    t_end_pd = pd.DatetimeIndex(df[TIME_COL].to_numpy()[ends])
    t_end_sec = t_end_pd.values.astype("datetime64[s]").astype("int64").astype(float)
    seq_run_frac = np.array([op[s:s+TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999
    inc_s_all = pd.DatetimeIndex(inc_full).values.astype("datetime64[s]").astype("int64").astype(float)
    pos = np.searchsorted(inc_s_all, t_end_sec)
    dnext = np.where(pos < len(inc_s_all), (inc_s_all[np.clip(pos,0,len(inc_s_all)-1)] - t_end_sec), np.inf)/3600.0
    dprev = np.where(pos > 0, (t_end_sec - inc_s_all[np.clip(pos-1,0,len(inc_s_all)-1)]), np.inf)/3600.0
    dany = np.minimum(dnext, dprev)
    is_train = (t_end_pd <= t_split)
    is_test = ~is_train
    log(f"seqs: train={int(is_train.sum())} | test={int(is_test.sum())}")
    inc_tr = inc_full[inc_full <= t_split]
    inc_te = inc_full[inc_full > t_split]
    inc_tr_s = pd.DatetimeIndex(inc_tr).values.astype("datetime64[s]").astype("int64").astype(float)
    inc_te_s = pd.DatetimeIndex(inc_te).values.astype("datetime64[s]").astype("int64").astype(float)
    log(f"incidentes: train={len(inc_tr)} | test={len(inc_te)}")
    is_neg = (dany > GAP_H) & seq_run_full & is_train
    neg_idx = np.where(is_neg)[0]
    n_tr = int(0.9 * len(neg_idx))
    tr_pool = neg_idx[:n_tr]; va_idx = neg_idx[n_tr:]
    rng = np.random.default_rng(42)
    if len(tr_pool) > 40000:
        tr_pool = rng.choice(tr_pool, 40000, replace=False)
    log(f"treino: {len(tr_pool)} negs no periodo train")
    dt_seconds = STRIDE * 30.0
    span_days_test = max((t_end_sec[is_test].max() - t_end_sec[is_test].min()) / 86400.0, 1e-9)

    from tensorflow import keras
    health = np.empty((len(starts), len(SENSORS)), dtype=np.float32)
    t1 = time.time()
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
        health[:, si] = compute_health_index_ewma(mae_i, seq_run_frac,
                                                  half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds)
        keras.backend.clear_session()
        if (si+1) % 5 == 0: log(f"  {si+1}/17")
    log(f"17 modelos em {time.time()-t1:.0f}s")

    print()
    print(f"{'H':<5}{'strategy':<14}{'recall':>10}{'FA/d':>10}{'eps':>8}{'lead':>9}")
    print("-" * 56)
    results = {}
    for H in HORIZONS:
        label_pre_tr = (dnext >= 0) & (dnext <= H) & is_train
        # F1-best CALCULADO em TRAIN
        thr_f1 = np.array([f1_best_threshold(health[is_train, j], label_pre_tr[is_train])
                            for j in range(len(SENSORS))])
        # thresholds aplicados no TEST
        alert_A_full = ((health >= thr_f1[None, :]).any(axis=1)) & seq_run_full & is_test
        # Reduz aos pontos do test set para evaluator
        alert_A = alert_A_full[is_test]
        m_A_test = evaluate_alerts(alert_A, t_end_sec[is_test],
                                    inc_te_s, H, DEBOUNCE_H, span_days_test)
        # BASELINE q=0.715 calculado em TRAIN
        thr_unif = np.array([float(np.quantile(health[is_train & seq_run_full, j], 0.715))
                              for j in range(len(SENSORS))])
        alert_base_full = ((health >= thr_unif[None, :]).any(axis=1)) & seq_run_full & is_test
        alert_base = alert_base_full[is_test]
        m_base_test = evaluate_alerts(alert_base, t_end_sec[is_test],
                                       inc_te_s, H, DEBOUNCE_H, span_days_test)
        for name, m in [("BASELINE_uni", m_base_test), ("A (F1-best)", m_A_test)]:
            print(f"{H:<5}{name:<14}{m['recall']:>10.2f}{m['fa_per_day']:>10.3f}"
                  f"{m['n_episodes']:>8d}{m['median_lead_hours']:>9.1f}")
        results[f"H{H}h"] = {"baseline": m_base_test, "A": m_A_test}
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    log(f"total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
