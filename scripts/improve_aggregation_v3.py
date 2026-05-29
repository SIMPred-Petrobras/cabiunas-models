#!/usr/bin/env python
"""Iteracao 3: Strategy D - Group-based AE (2 grupos fisicos).

Hipotese: 2 modelos (temperatura=7ch, vibracao=10ch) capturam correlacoes
intra-grupo melhor que 17 individuais OU 1 multi de 17, com atribuicao
por subsistema fisico.

Compara com Strategy A (per-sensor F1-best) e BASELINE.
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

TS = 60; STRIDE = 10; HALF_LIFE_H = 4.0; GAP_H = 12.0; DEBOUNCE_H = 8.0
HORIZON = 8.0
OUT = "improve_agg_v3_out"
HEALTH_CACHE = "improve_agg_out/health_per_sensor_cache.npz"  # per-sensor cache da iter1

# Grupos fisicos
TEMP = ["T5_AVG_A", "TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A", "TC382_06_A"]
VIB = ["TV_351X_A", "TV_351Y_A", "TV_352X_A", "TV_352Y_A", "TV_353X_A", "TV_353Y_A",
       "TV_354X_A", "TV_354Y_A", "TV_355X_A", "TV_355Y_A"]


def log(m): print(f"[IT3] {m}", flush=True)


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


def f1_best_threshold(health_col, label_pre, mask_running):
    s = health_col[mask_running]; y = label_pre[mask_running]
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


def train_group_ae(df, group_sensors, tr_pool, va_idx, seq_run_frac, dt_seconds):
    """Treina AE de canal multiplo so dos sensores do grupo."""
    from tensorflow import keras
    n = len(df)
    starts = np.arange(0, n - TS + 1, STRIDE)
    X = df[group_sensors].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32)
    seqs, _ = make_seqs(X, TS, STRIDE)
    flat = seqs[tr_pool].reshape(-1, seqs.shape[-1])
    mu = flat.mean(0); sd = flat.std(0); sd[sd==0] = 1.0
    norm = (seqs - mu) / sd
    # mesma config do multivariate vencedor (f1=16, f2=3) mas adaptada ao grupo
    n_ch = len(group_sensors)
    model, latent = build_ae(TS, n_ch, 8, 2, 2, 2, 0.1, 1e-4)
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    model.fit(norm[tr_pool], norm[tr_pool], validation_data=(norm[va_idx], norm[va_idx]),
              epochs=15, batch_size=256, verbose=0, callbacks=cb)
    mae = mse_per_seq(model, norm)
    health = compute_health_index_ewma(mae, seq_run_frac,
                                       half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds)
    keras.backend.clear_session()
    return health, latent


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
    label_pre = (dnext >= 0) & (dnext <= HORIZON)
    span_days = max((t_end_sec.max() - t_end_sec.min()) / 86400.0, 1e-9)
    is_neg = (np.minimum(dnext, np.where(pos>0, (t_end_sec - inc_s[np.clip(pos-1,0,len(inc_s)-1)]), np.inf)/3600.0) > GAP_H) & seq_run_full
    neg_idx = np.where(is_neg)[0]
    rng = np.random.default_rng(42)
    n_tr = int(0.9 * len(neg_idx))
    tr_pool = neg_idx[:n_tr]; va_idx = neg_idx[n_tr:]
    if len(tr_pool) > 40000:
        tr_pool = rng.choice(tr_pool, 40000, replace=False)
    dt_seconds = STRIDE * 30.0

    log("treinando AE de temperatura (7 canais)...")
    health_T, latent_T = train_group_ae(df, TEMP, tr_pool, va_idx, seq_run_frac, dt_seconds)
    log(f"  latente={latent_T} | dim={TS*len(TEMP)} | ratio={latent_T/(TS*len(TEMP)):.3f}")
    log("treinando AE de vibracao (10 canais)...")
    health_V, latent_V = train_group_ae(df, VIB, tr_pool, va_idx, seq_run_frac, dt_seconds)
    log(f"  latente={latent_V} | dim={TS*len(VIB)} | ratio={latent_V/(TS*len(VIB)):.3f}")

    # === Strategy D variantes ===
    log("=== Strategy D: 2 grupos (T+V), avaliando varias agregacoes ===")
    # D1: OR ambos no F1-best individual
    thr_T = f1_best_threshold(health_T, label_pre, seq_run_full)
    thr_V = f1_best_threshold(health_V, label_pre, seq_run_full)
    alert_D1 = ((health_T >= thr_T) | (health_V >= thr_V)) & seq_run_full
    m_D1 = evaluate_alerts(alert_D1, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
    log(f"  D1 (OR F1-best): recall={m_D1['recall']:.2f} FA/d={m_D1['fa_per_day']:.3f} eps={m_D1['n_episodes']}")

    # D2: OR uniforme q=0.715
    thr_T_unif = float(np.quantile(health_T[seq_run_full], 0.715))
    thr_V_unif = float(np.quantile(health_V[seq_run_full], 0.715))
    alert_D2 = ((health_T >= thr_T_unif) | (health_V >= thr_V_unif)) & seq_run_full
    m_D2 = evaluate_alerts(alert_D2, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
    log(f"  D2 (OR q=0.715): recall={m_D2['recall']:.2f} FA/d={m_D2['fa_per_day']:.3f} eps={m_D2['n_episodes']}")

    # D3: AND (ambos os grupos juntos -- mais conservador)
    alert_D3 = ((health_T >= thr_T) & (health_V >= thr_V)) & seq_run_full
    m_D3 = evaluate_alerts(alert_D3, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
    log(f"  D3 (AND F1):    recall={m_D3['recall']:.2f} FA/d={m_D3['fa_per_day']:.3f} eps={m_D3['n_episodes']}")

    # === E: hibrido per_sensor + grupos (UNION dos alertas) ===
    log("=== Strategy E: hibrido per_sensor F1 + group D1 (UNION) ===")
    health = np.load(HEALTH_CACHE)["health"]
    thr_f1_per = np.array([f1_best_threshold(health[:, j], label_pre, seq_run_full)
                            for j in range(len(SENSORS))])
    alert_per = ((health >= thr_f1_per[None, :]).any(axis=1)) & seq_run_full
    alert_E = (alert_per | alert_D1) & seq_run_full
    m_E = evaluate_alerts(alert_E, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
    log(f"  E (union persensor+grupo): recall={m_E['recall']:.2f} FA/d={m_E['fa_per_day']:.3f} eps={m_E['n_episodes']}")

    # === comparacao final ===
    m_A = evaluate_alerts(alert_per, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
    thr_unif = np.array([float(np.quantile(health[seq_run_full, j], 0.715)) for j in range(len(SENSORS))])
    alert_base = ((health >= thr_unif[None, :]).any(axis=1)) & seq_run_full
    m_base = evaluate_alerts(alert_base, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
    print()
    print(f"{'strat':<14}{'recall':>10}{'FA/d':>10}{'eps':>8}{'lead':>8}{'detalhe':>22}")
    print("-" * 74)
    for name, m, det in [
        ("BASELINE_uni", m_base, "OR q=0.715 / 17 sensores"),
        ("A (F1 17)", m_A, "OR F1-best per sensor"),
        ("D1 (OR T+V F1)", m_D1, "grupo T or V F1"),
        ("D2 (OR T+V q)", m_D2, "grupo T or V q=0.715"),
        ("D3 (AND T+V)", m_D3, "grupo T and V F1"),
        ("E (union)", m_E, "A OR D1"),
    ]:
        print(f"{name:<14}{m['recall']:>10.2f}{m['fa_per_day']:>10.3f}"
              f"{m['n_episodes']:>8d}{m['median_lead_hours']:>8.1f}{det:>22}")

    # decisao por Pareto (recall>baseline E FA/d<=baseline*1.5)
    cands = {"A": m_A, "D1": m_D1, "D2": m_D2, "D3": m_D3, "E": m_E}
    winner = None; best_gain = -1
    for name, m in cands.items():
        if m["recall"] >= m_base["recall"] and m["fa_per_day"] <= max(m_base["fa_per_day"]*1.5, 0.06):
            gain = m["recall"] - m_base["recall"]
            if gain > best_gain: best_gain = gain; winner = name
    log(f"VENCEDOR: {winner} (+{best_gain*100:.1f}pp recall vs BASELINE)" if winner else "MANTER BASELINE")
    json.dump({"baseline": m_base, "A": m_A, "D1": m_D1, "D2": m_D2, "D3": m_D3, "E": m_E,
               "winner": winner},
              open(f"{OUT}/results.json", "w"), indent=2)
    log(f"total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
