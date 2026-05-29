#!/usr/bin/env python
"""Iteracao autonoma: testar estrategias de agregacao alternativas vs baseline OR-uniforme.

Baseline atual: OR-de-quantile uniforme em q=0.715 -> recall 67% / FA/d 0.03 (H=8h).
Objetivo: subir recall mantendo FA/d <= 0.10.

Estrategias testadas:
  A. Per-sensor optimal threshold (F1-best individual; OR aggregator)
  B. Weighted sum (peso = AUC do sensor) + threshold no signal escalar
  C. Voting >=2 (uniforme, controle)

Para cada: computa curva preditiva real (H=8h), acha ponto operacional real.
Se alguma vencer baseline em recall E FA/d, atualiza recomendacao.
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
HALF_LIFE_H = 4.0; GAP_H = 12.0; DEBOUNCE_H = 8.0
HORIZON = 8.0
OUT = "improve_agg_out"
HEALTH_CACHE = f"{OUT}/health_per_sensor_cache.npz"


def log(m): print(f"[IMP] {m}", flush=True)


def auc_per_sensor(health: np.ndarray, label_pre: np.ndarray, mask_running: np.ndarray):
    """AUC-ROC por sensor (positivo = janela pre-incidente em running)."""
    aucs = np.empty(health.shape[1])
    y = label_pre[mask_running]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.full(health.shape[1], 0.5)
    for j in range(health.shape[1]):
        s = health[mask_running, j]
        # AUC via Mann-Whitney sem sklearn
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s)+1)
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        sum_r = np.zeros(len(cnt)); np.add.at(sum_r, inv, ranks); ranks = (sum_r/cnt)[inv]
        n_pos = int(y.sum()); n_neg = int((~y).sum())
        aucs[j] = float((ranks[y].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg))
    return aucs


def f1_best_threshold(health_col: np.ndarray, label_pre: np.ndarray, mask_running: np.ndarray):
    """Threshold escalar do sensor que maximiza F1 contra label pre-incidente."""
    s = health_col[mask_running]; y = label_pre[mask_running]
    if y.sum() == 0: return float("inf"), 0.0
    qs = np.quantile(s, np.linspace(0.3, 0.999, 60))
    best_f1 = 0.0; best_thr = float(np.median(s))
    for thr in qs:
        pred = s >= thr
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
        if tp == 0: continue
        p = tp/(tp+fp); r = tp/(tp+fn); f = 2*p*r/(p+r)
        if f > best_f1: best_f1 = f; best_thr = float(thr)
    return best_thr, best_f1


def evaluate_alerts(alert_seq: np.ndarray, t_end_sec: np.ndarray,
                    inc_s: np.ndarray, horizon_h: float, debounce_h: float, span_days: float):
    """Mede recall, FA/dia, lead, n_episodes a partir de um vetor binario de alerta."""
    H = horizon_h * 3600
    deb = debounce_h * 3600
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


def find_real_op_point(curve_rows, n_inc):
    df = pd.DataFrame(curve_rows)
    if df.empty: return None
    feas = df[(df["fa_per_day"] >= 0.03) & (df["fa_per_day"] <= 0.30)
              & (df["n_episodes"] >= 0.3 * n_inc)]
    if not len(feas): feas = df[df["fa_per_day"] >= 0.03]
    if not len(feas): return None
    return feas.loc[feas["recall"].idxmax()].to_dict()


def train_or_load_per_sensor(df, sensors, tr_pool, va_idx, seq_run_frac, dt_seconds, starts):
    if os.path.exists(HEALTH_CACHE):
        log(f"carregando cache: {HEALTH_CACHE}")
        d = np.load(HEALTH_CACHE)
        return d["health"]
    from tensorflow import keras
    health_mat = np.empty((len(starts), len(sensors)), dtype=np.float32)
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
        mae_i = mse_per_seq(model, norm)
        health_mat[:, si] = compute_health_index_ewma(
            mae_i, seq_run_frac, half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds,
        )
        keras.backend.clear_session()
        if (si+1) % 5 == 0: log(f"  {si+1}/{len(sensors)}")
    log(f"17 modelos treinados em {time.time()-t0:.0f}s; cachando em {HEALTH_CACHE}")
    np.savez_compressed(HEALTH_CACHE, health=health_mat)
    return health_mat


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    df, inc_full = load(priority=None)
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    n_inc = len(inc_full)
    log(f"incidentes: {n_inc}")

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

    # label "pre-incidente em 8h" (pra otimizar threshold por sensor)
    label_pre = (dnext >= 0) & (dnext <= HORIZON)

    # 1. health per sensor (com cache)
    health = train_or_load_per_sensor(df, SENSORS, tr_pool, va_idx, seq_run_frac, dt_seconds, starts)
    n_sens = health.shape[1]
    log(f"health: {health.shape}")

    # =================== AUC por sensor (para B e relatorio) ===================
    aucs = auc_per_sensor(health, label_pre, seq_run_full)
    log("AUC por sensor (top-5 e bottom-3):")
    order = np.argsort(aucs)[::-1]
    for j in order[:5]: log(f"  + {SENSORS[j]}: AUC={aucs[j]:.3f}")
    for j in order[-3:]: log(f"  - {SENSORS[j]}: AUC={aucs[j]:.3f}")

    # =================== BASELINE: OR uniforme q=0.715 ===================
    log("=== BASELINE: OR uniforme q=0.715 ===")
    thr_unif = np.empty(n_sens)
    for j in range(n_sens):
        thr_unif[j] = float(np.quantile(health[seq_run_full, j], 0.715))
    alert_baseline = ((health >= thr_unif[None, :]).any(axis=1)) & seq_run_full
    m_base = evaluate_alerts(alert_baseline, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
    log(f"  BASELINE: recall={m_base['recall']:.2f} FA/d={m_base['fa_per_day']:.2f} eps={m_base['n_episodes']}")

    # =================== A. Per-sensor F1-best threshold ===================
    log("=== A. Per-sensor F1-best threshold (OR) ===")
    thr_f1 = np.empty(n_sens); f1_per = np.empty(n_sens)
    for j in range(n_sens):
        thr_f1[j], f1_per[j] = f1_best_threshold(health[:, j], label_pre, seq_run_full)
    alert_A = ((health >= thr_f1[None, :]).any(axis=1)) & seq_run_full
    m_A = evaluate_alerts(alert_A, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
    log(f"  A: recall={m_A['recall']:.2f} FA/d={m_A['fa_per_day']:.2f} eps={m_A['n_episodes']}")

    # =================== B. Weighted sum + threshold ===================
    log("=== B. Soma ponderada por AUC (peso = max(AUC-0.5, 0)) + threshold ===")
    weights = np.maximum(aucs - 0.5, 0.0)
    if weights.sum() == 0: weights = np.ones(n_sens)
    weights = weights / weights.sum()
    # normaliza cada sensor por seu desvio na operacao (z-score per sensor)
    health_z = np.copy(health)
    for j in range(n_sens):
        v = health[seq_run_full, j]
        mu_j, sd_j = float(v.mean()), float(v.std()) + 1e-9
        health_z[:, j] = (health[:, j] - mu_j) / sd_j
    score_B = (health_z * weights[None, :]).sum(axis=1)
    # sweep threshold sobre score_B em quantis, escolhe ponto real
    rows_B = []
    qs = np.linspace(0.50, 0.999, 60)
    for q in qs:
        thr = float(np.quantile(score_B[seq_run_full], q))
        alert_B = (score_B >= thr) & seq_run_full
        m = evaluate_alerts(alert_B, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
        m["q"] = float(q); m["thr"] = thr
        rows_B.append(m)
    op_B = find_real_op_point(rows_B, n_inc)
    if op_B:
        log(f"  B: recall={op_B['recall']:.2f} FA/d={op_B['fa_per_day']:.2f} "
            f"eps={int(op_B['n_episodes'])} (q_score={op_B['q']:.3f})")

    # =================== C. Voting >=2 (uniforme q=0.5) ===================
    log("=== C. Voting >=2 (uniforme em varios q) ===")
    rows_C = []
    for q in [0.50, 0.60, 0.70, 0.715, 0.80, 0.85, 0.90, 0.95]:
        thr_c = np.array([float(np.quantile(health[seq_run_full, j], q)) for j in range(n_sens)])
        n_above_c = (health >= thr_c[None, :]).sum(axis=1)
        alert_C = (n_above_c >= 2) & seq_run_full
        m = evaluate_alerts(alert_C, t_end_sec, inc_s, HORIZON, DEBOUNCE_H, span_days)
        m["q"] = float(q)
        rows_C.append(m)
    op_C = find_real_op_point(rows_C, n_inc)
    if op_C:
        log(f"  C: recall={op_C['recall']:.2f} FA/d={op_C['fa_per_day']:.2f} "
            f"eps={int(op_C['n_episodes'])} (q={op_C['q']:.3f})")

    # =================== DECISAO ===================
    log("=== resumo ===")
    print(f"{'strat':<10}{'recall':>10}{'FA/d':>10}{'eps':>8}{'lead_h':>9}{'detalhe':>15}")
    print("-" * 62)
    def fmt(label, m, extra=""):
        if not m: return
        print(f"{label:<10}{m['recall']:>10.2f}{m['fa_per_day']:>10.3f}"
              f"{int(m['n_episodes']):>8d}{m['median_lead_hours']:>9.1f}{extra:>15}")
    fmt("BASELINE", m_base, "q=0.715")
    fmt("A (F1)", m_A, "per-sensor")
    fmt("B (sum)", op_B, f"q_score={op_B['q']:.2f}" if op_B else "")
    fmt("C (vot≥2)", op_C, f"q={op_C['q']:.2f}" if op_C else "")
    print()

    # criterio: vence se recall > baseline E FA/d <= 1.5 x baseline
    cands = {"A": m_A, "B": op_B, "C": op_C}
    winner = None; best_gain = 0
    for name, m in cands.items():
        if not m: continue
        if m["recall"] > m_base["recall"] and m["fa_per_day"] <= max(m_base["fa_per_day"] * 1.5, 0.10):
            gain = m["recall"] - m_base["recall"]
            if gain > best_gain: best_gain = gain; winner = name
    decision = f"VENCEDOR: {winner} (+{best_gain*100:.1f}pp recall vs baseline)" if winner else "MANTER BASELINE (nenhuma melhoria)"
    log(decision)
    json.dump({"baseline": m_base, "A": m_A, "B": op_B, "C": op_C,
               "decision": decision, "aucs_sorted":
               [{"sensor": SENSORS[j], "auc": float(aucs[j])} for j in order]},
              open(f"{OUT}/results.json", "w"), indent=2)
    log(f"total: {time.time()-t0:.0f}s | salvo {OUT}/results.json")


if __name__ == "__main__":
    main()
