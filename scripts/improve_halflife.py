#!/usr/bin/env python
"""Iteracao 4: sweep do EWMA half-life pra otimizar smoothing.

Hipotese: half-life de 4h pode estar errado. Testa 1, 2, 4, 8, 16h.
Avalia com baseline OR q=0.715 (robusto). Vencedor passa por validacao temporal.
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

TS = 60; STRIDE = 10
F1 = 4; F2 = 1; S1 = 2; S2 = 2
GAP_H = 12.0; DEBOUNCE_H = 8.0
HORIZON = 8.0; Q_OP = 0.715
TRAIN_FRAC = 0.66
OUT = "improve_halflife_out"
MAE_CACHE = f"{OUT}/mae_per_sensor_cache.npz"


def log(m): print(f"[HL] {m}", flush=True)


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


def get_or_train_mae(df, sensors, tr_pool, va_idx, starts):
    if os.path.exists(MAE_CACHE):
        log(f"carregando MAE cache: {MAE_CACHE}")
        return np.load(MAE_CACHE)["mae"]
    from tensorflow import keras
    mae_mat = np.empty((len(starts), len(sensors)), dtype=np.float32)
    t0 = time.time()
    for si, sensor in enumerate(sensors):
        Xi = pd.to_numeric(df[sensor], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32).reshape(-1, 1)
        seqs_i, _ = make_seqs(Xi, TS, STRIDE)
        flat = seqs_i[tr_pool].reshape(-1, 1)
        mu = flat.mean(0); sd = flat.std(0); sd[sd==0] = 1.0
        norm = (seqs_i - mu) / sd
        model, _ = build_ae(TS, 1, F1, F2, S1, S2, 0.1, 1e-4)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
        model.fit(norm[tr_pool], norm[tr_pool], validation_data=(norm[va_idx], norm[va_idx]),
                  epochs=15, batch_size=256, verbose=0, callbacks=cb)
        mae_mat[:, si] = mse_per_seq(model, norm)
        keras.backend.clear_session()
        if (si+1) % 5 == 0: log(f"  {si+1}/17")
    log(f"17 modelos em {time.time()-t0:.0f}s; cache em {MAE_CACHE}")
    np.savez_compressed(MAE_CACHE, mae=mae_mat)
    return mae_mat


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    df, inc_full = load(priority=None)
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    n_inc = len(inc_full)
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
    dt_seconds = STRIDE * 30.0
    span_days = max((t_end_sec.max() - t_end_sec.min()) / 86400.0, 1e-9)

    mae = get_or_train_mae(df, SENSORS, tr_pool, va_idx, starts)
    log(f"mae: {mae.shape}")

    # === SWEEP HALF-LIFE in-sample ===
    log("=== Sweep half-life (in-sample, OR uniforme q=0.715) ===")
    print(f"{'half-life':>10}{'recall':>10}{'FA/d':>10}{'eps':>8}{'lead':>8}")
    print("-" * 46)
    rows = []
    for hl in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        health = np.empty_like(mae)
        for j in range(mae.shape[1]):
            health[:, j] = compute_health_index_ewma(
                mae[:, j], seq_run_frac, half_life_hours=hl, dt_seconds=dt_seconds,
            )
        thr = np.array([float(np.quantile(health[seq_run_full, j], Q_OP)) for j in range(mae.shape[1])])
        alert = ((health >= thr[None, :]).any(axis=1)) & seq_run_full
        m = evaluate_alerts(alert, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
        m["hl"] = hl
        rows.append(m)
        print(f"{hl:>10.1f}{m['recall']:>10.2f}{m['fa_per_day']:>10.3f}"
              f"{m['n_episodes']:>8d}{m['median_lead_hours']:>8.1f}")
    df_r = pd.DataFrame(rows)
    # vencedor: melhor recall com FA/d <= 0.05
    feas = df_r[df_r["fa_per_day"] <= 0.05]
    if len(feas):
        winner = feas.loc[feas["recall"].idxmax()].to_dict()
        log(f"VENCEDOR in-sample: half-life={winner['hl']}h "
            f"(recall={winner['recall']:.2f} FA/d={winner['fa_per_day']:.3f})")
    else:
        log("nenhum candidato com FA/d<=0.05")
        return

    # === VALIDACAO TEMPORAL DO VENCEDOR ===
    if winner["hl"] == 4.0:
        log("vencedor in-sample = baseline (4h). Sem mudanca.")
        return
    log(f"=== validacao temporal: half-life={winner['hl']}h vs baseline 4h ===")
    n_train_pts = int(TRAIN_FRAC * n)
    t_split = df[TIME_COL].iloc[n_train_pts]
    is_train = t_end_pd <= t_split
    is_test = ~is_train
    inc_te = inc_full[inc_full > t_split]
    inc_te_s = pd.DatetimeIndex(inc_te).values.astype("datetime64[s]").astype("int64").astype(float)
    span_days_test = max((t_end_sec[is_test].max() - t_end_sec[is_test].min()) / 86400.0, 1e-9)
    # retreinar SO com negs de train period
    is_neg_tr = is_neg & is_train
    neg_idx_tr = np.where(is_neg_tr)[0]
    n_tr2 = int(0.9 * len(neg_idx_tr))
    tr_pool2 = neg_idx_tr[:n_tr2]; va_idx2 = neg_idx_tr[n_tr2:]
    if len(tr_pool2) > 40000:
        tr_pool2 = rng.choice(tr_pool2, 40000, replace=False)
    # nova cache pra train period
    log("retreinando 17 AEs em train period...")
    mae_tr = np.empty((len(starts), mae.shape[1]), dtype=np.float32)
    from tensorflow import keras
    t1 = time.time()
    for si, sensor in enumerate(SENSORS):
        Xi = pd.to_numeric(df[sensor], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32).reshape(-1, 1)
        seqs_i, _ = make_seqs(Xi, TS, STRIDE)
        flat = seqs_i[tr_pool2].reshape(-1, 1)
        mu = flat.mean(0); sd = flat.std(0); sd[sd==0] = 1.0
        norm = (seqs_i - mu) / sd
        model, _ = build_ae(TS, 1, F1, F2, S1, S2, 0.1, 1e-4)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
        model.fit(norm[tr_pool2], norm[tr_pool2], validation_data=(norm[va_idx2], norm[va_idx2]),
                  epochs=15, batch_size=256, verbose=0, callbacks=cb)
        mae_tr[:, si] = mse_per_seq(model, norm)
        keras.backend.clear_session()
        if (si+1) % 5 == 0: log(f"  {si+1}/17")
    log(f"retreinado em {time.time()-t1:.0f}s")

    print()
    print(f"{'hl':>5}{'phase':<8}{'recall':>10}{'FA/d':>10}{'eps':>8}")
    print("-" * 41)
    for hl in [4.0, winner["hl"]]:
        health = np.empty_like(mae_tr)
        for j in range(mae_tr.shape[1]):
            health[:, j] = compute_health_index_ewma(
                mae_tr[:, j], seq_run_frac, half_life_hours=hl, dt_seconds=dt_seconds,
            )
        # threshold calculado em TRAIN
        thr = np.array([float(np.quantile(health[is_train & seq_run_full, j], Q_OP))
                         for j in range(mae_tr.shape[1])])
        alert_full = ((health >= thr[None, :]).any(axis=1)) & seq_run_full & is_test
        alert = alert_full[is_test]
        m_test = evaluate_alerts(alert, t_end_sec[is_test],
                                  inc_te_s, HORIZON, DEBOUNCE_H, span_days_test)
        print(f"{hl:>5.1f}{'TEST':<8}{m_test['recall']:>10.2f}{m_test['fa_per_day']:>10.3f}{m_test['n_episodes']:>8d}")
        rows.append({"hl": hl, "phase": "TEST", **m_test})
    log(f"total: {time.time()-t0:.0f}s")
    json.dump({"in_sample": rows[:6], "temporal_test": rows[-2:]},
              open(f"{OUT}/results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
