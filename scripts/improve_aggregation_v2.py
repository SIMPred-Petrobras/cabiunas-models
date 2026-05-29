#!/usr/bin/env python
"""Iteracao 2: refina a vencedora (Per-sensor F1-best) sweepando scaling factor.

Resultado da iter1: A vence em recall mas com so 23 episodios/ano (vs 51 do baseline).
Pode estar saturada. Aqui faco sweep do scaling factor sobre os F1-best thresholds
pra achar o ponto operacional MATCHED a baseline (similar n_episodes), e tambem
explora pontos mais agressivos (mais episodios = mais alertas discretos).
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import load, make_seqs, SENSORS, TIME_COL, RUNNING_COL
from cnn1d_ae.predictive import _detect_episodes

TS = 60; STRIDE = 10; HALF_LIFE_H = 4.0; GAP_H = 12.0; DEBOUNCE_H = 8.0
HORIZON = 8.0
OUT = "improve_agg_v2_out"
HEALTH_CACHE = "improve_agg_out/health_per_sensor_cache.npz"


def log(m): print(f"[IT2] {m}", flush=True)


def f1_best_threshold(health_col: np.ndarray, label_pre: np.ndarray, mask_running: np.ndarray):
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

    log(f"incidentes={n_inc} | carregando health cache")
    health = np.load(HEALTH_CACHE)["health"]
    n_sens = health.shape[1]
    log(f"health: {health.shape}")

    # F1-best por sensor
    thr_f1 = np.empty(n_sens)
    for j in range(n_sens):
        thr_f1[j] = f1_best_threshold(health[:, j], label_pre, seq_run_full)

    # baseline (q=0.715 uniforme)
    thr_unif = np.array([float(np.quantile(health[seq_run_full, j], 0.715)) for j in range(n_sens)])
    alert_base = ((health >= thr_unif[None, :]).any(axis=1)) & seq_run_full
    m_base = evaluate_alerts(alert_base, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
    log(f"BASELINE q=0.715 → recall={m_base['recall']:.2f} FA/d={m_base['fa_per_day']:.3f} eps={m_base['n_episodes']}")

    # SWEEP A: escala os thresholds F1-best por fator k
    # k>1 = mais conservador (thresholds maiores, alertas raros)
    # k<1 = mais permissivo
    log("=== Sweep A (per-sensor F1 scaled by k) ===")
    print(f"{'k':>6}{'recall':>10}{'FA/d':>10}{'eps':>8}{'lead':>8}")
    print("-" * 42)
    rows = []
    for k in np.linspace(0.5, 1.8, 27):
        thr_k = thr_f1 * float(k)
        alert = ((health >= thr_k[None, :]).any(axis=1)) & seq_run_full
        m = evaluate_alerts(alert, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
        m["k"] = float(k)
        rows.append(m)
        print(f"{k:>6.2f}{m['recall']:>10.2f}{m['fa_per_day']:>10.3f}"
              f"{m['n_episodes']:>8d}{m['median_lead_hours']:>8.1f}")
    df_curve = pd.DataFrame(rows)
    df_curve.to_csv(f"{OUT}/A_sweep_curve.csv", index=False)

    # ponto operacional matched: n_episodes proximo do baseline
    target_eps = m_base["n_episodes"]
    df_curve["eps_diff"] = (df_curve["n_episodes"] - target_eps).abs()
    matched = df_curve.loc[df_curve["eps_diff"].idxmin()].to_dict()
    log(f"--- ponto MATCHED (eps≈{target_eps}) ---")
    log(f"  A_matched (k={matched['k']:.2f}): recall={matched['recall']:.2f} FA/d={matched['fa_per_day']:.3f} eps={int(matched['n_episodes'])}")
    log(f"  delta vs baseline: recall {matched['recall']-m_base['recall']:+.2f} | FA/d {matched['fa_per_day']-m_base['fa_per_day']:+.3f}")

    # ponto operacional realista (50-150 eps, melhor recall)
    feas = df_curve[(df_curve["n_episodes"] >= 30) & (df_curve["n_episodes"] <= 150)]
    if len(feas):
        best_real = feas.loc[feas["recall"].idxmax()].to_dict()
        log(f"--- ponto REAL (30-150 eps, melhor recall) ---")
        log(f"  A_real (k={best_real['k']:.2f}): recall={best_real['recall']:.2f} FA/d={best_real['fa_per_day']:.3f} eps={int(best_real['n_episodes'])}")
        log(f"  delta vs baseline: recall {best_real['recall']-m_base['recall']:+.2f} | FA/d {best_real['fa_per_day']-m_base['fa_per_day']:+.3f}")

    # decisao
    cands = []
    if 'matched' in dir(): cands.append(("A_matched", matched))
    if len(feas): cands.append(("A_real", best_real))
    winner = None; best_gain = 0
    for name, m in cands:
        if m["recall"] > m_base["recall"] and m["fa_per_day"] <= max(m_base["fa_per_day"] * 1.3, 0.06):
            gain = m["recall"] - m_base["recall"]
            if gain > best_gain: best_gain = gain; winner = (name, m)
    if winner:
        log(f"=== VENCEDOR: {winner[0]} (+{best_gain*100:.1f}pp recall, FA/d {winner[1]['fa_per_day']:.3f}) ===")
        decision = winner
    else:
        log("=== MANTER BASELINE (nenhum ponto matched supera) ===")
        decision = None
    json.dump({"baseline": m_base, "matched": matched if 'matched' in dir() else None,
               "real": best_real if len(feas) else None,
               "winner": winner[0] if winner else None,
               "winner_metrics": winner[1] if winner else None,
               "sweep": rows},
              open(f"{OUT}/results.json", "w"), indent=2)
    log(f"salvo {OUT}/results.json")


if __name__ == "__main__":
    main()
