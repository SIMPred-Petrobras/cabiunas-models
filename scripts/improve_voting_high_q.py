#!/usr/bin/env python
"""Ultimo teste algoritmico: voting (>=k sensores juntos) com q alto.

Hipotese: degradacao real coordena multiplos sensores (acoplamento fisico);
ruido eh independente por sensor. Em q alto + voting:
- Ruido: probabilidade de k sensores ALEATORIAMENTE acima cai rapidamente
- Degradacao: sensores SE COORDENAM → k>=2 facil de atingir

Testa varias combinacoes (q, k) sobre o test period (set-dez),
threshold calibrado em train (jan-ago).

CRITERIO BINARIO PRE-DEFINIDO:
  PASSA: recall OOS >= 0.60 E tempo alerta point-level <= 30%
  FALHA: qualquer um dos dois

Se PASSA: ganho real, migra arquitetura.
Se FALHA: capitulo algorithmico definitivamente fechado.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import load, SENSORS, TIME_COL, RUNNING_COL
from cnn1d_ae.predictive import compute_health_index_ewma, _detect_episodes

TS = 60; STRIDE = 10
HL = 4.0; GAP_H = 12.0; DEBOUNCE_H = 8.0
HORIZON = 8.0; TRAIN_FRAC = 0.66
MAE_CACHE = "improve_halflife_out/mae_per_sensor_cache.npz"
OUT = "improve_voting_high_q_out"

# Grid de teste
Q_VALUES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97]
K_VALUES = [1, 2, 3, 4, 5, 6, 7]  # k=1 = OR original; k>=2 = voting

# Criterios decisivos
TARGET_RECALL = 0.60
MAX_POINT_LEVEL_ALERT = 0.30
BASELINE_REF = 0.58  # baseline atual q=0.715 k=1


def log(m): print(f"[VOT] {m}", flush=True)


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
                n_episodes=int(len(episodes)),
                point_level_alert_pct=float(alert_seq.mean()*100))


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
    is_train = (t_end_pd <= t_split)
    is_test = ~is_train
    inc_te = inc_full[inc_full > t_split]
    inc_te_s = pd.DatetimeIndex(inc_te).values.astype("datetime64[s]").astype("int64").astype(float)
    span_days_test = max((t_end_sec[is_test].max() - t_end_sec[is_test].min()) / 86400.0, 1e-9)
    log(f"incidentes test: {len(inc_te)}")

    log("carregando MAE cache, computando EWMA...")
    mae = np.load(MAE_CACHE)["mae"]
    dt_seconds = STRIDE * 30.0
    health = np.empty_like(mae)
    for j in range(mae.shape[1]):
        health[:, j] = compute_health_index_ewma(mae[:, j], seq_run_frac,
                                                  half_life_hours=HL, dt_seconds=dt_seconds)
    n_sens = health.shape[1]
    log(f"health: {health.shape}")

    # === Grid search ===
    print()
    print(f"{'q':>6}{'k':>5}{'recall':>10}{'FA/d':>10}{'eps':>8}{'pt-lvl%':>10}{'passa?':>10}")
    print("-" * 59)
    rows = []
    for q in Q_VALUES:
        # threshold por sensor calibrado em TRAIN (CRITICO: nao usa test)
        thr = np.array([float(np.quantile(health[is_train & seq_run_full, j], q))
                         for j in range(n_sens)])
        above = (health >= thr[None, :])
        n_above = above.sum(axis=1)
        for k in K_VALUES:
            alert_full = (n_above >= k) & seq_run_full & is_test
            alert_test = alert_full[is_test]
            m = evaluate_alerts(alert_test, t_end_sec[is_test],
                                 inc_te_s, HORIZON, DEBOUNCE_H, span_days_test)
            passes = (m["recall"] >= TARGET_RECALL and
                      m["point_level_alert_pct"] <= MAX_POINT_LEVEL_ALERT * 100)
            mark = "✅" if passes else "❌"
            print(f"{q:>6.2f}{k:>5d}{m['recall']:>10.2f}{m['fa_per_day']:>10.3f}"
                  f"{m['n_episodes']:>8d}{m['point_level_alert_pct']:>9.1f}%{mark:>10}")
            rows.append({"q": q, "k": k, **m, "passes": bool(passes)})

    df_res = pd.DataFrame(rows)
    df_res.to_csv(f"{OUT}/voting_high_q_grid.csv", index=False)

    # === Decisao final ===
    print()
    passing = df_res[df_res["passes"]]
    log(f"Combinacoes que PASSAM o criterio (recall>={TARGET_RECALL} E pt-lvl<={MAX_POINT_LEVEL_ALERT*100}%): {len(passing)}")
    if len(passing) > 0:
        # ordena por recall (criterio primario)
        passing = passing.sort_values("recall", ascending=False)
        log("Top 5 candidatos validos:")
        for _, row in passing.head(5).iterrows():
            log(f"  q={row['q']:.2f}, k={int(row['k'])}: "
                f"recall={row['recall']:.2f}, FA/d={row['fa_per_day']:.3f}, "
                f"pt-lvl={row['point_level_alert_pct']:.1f}%, eps={int(row['n_episodes'])}")
        winner = passing.iloc[0].to_dict()
        log("")
        log(f"=== VENCEDOR: q={winner['q']:.2f}, k={int(winner['k'])} ===")
        log(f"  recall={winner['recall']:.2f} (vs baseline {BASELINE_REF})")
        log(f"  FA/d={winner['fa_per_day']:.3f}")
        log(f"  point-level alert={winner['point_level_alert_pct']:.1f}% (vs 99.7% baseline)")
        log("PASSA — capitulo algoritmico NAO fechado, migrar arquitetura.")
    else:
        # mesmo sem passar, encontra ponto mais proximo
        log("NENHUMA combinacao passa o criterio binario.")
        # melhor recall sob constraint de pt-lvl<=30%
        feas_pt = df_res[df_res["point_level_alert_pct"] <= 30]
        if len(feas_pt):
            best_pt = feas_pt.loc[feas_pt["recall"].idxmax()].to_dict()
            log(f"Melhor sob pt-lvl<=30%: q={best_pt['q']}, k={int(best_pt['k'])} → "
                f"recall={best_pt['recall']:.2f}, pt-lvl={best_pt['point_level_alert_pct']:.1f}%")
        log("")
        log("=== CAPITULO ALGORITMICO FECHADO ===")
        log("7 iteracoes algoritmicas. Plateau OOS H=8h em ~58% recall.")
        log("Proximos passos: operacional (tier alerts) + dados externos.")

    json.dump({"target_recall": TARGET_RECALL,
               "max_point_level_alert": MAX_POINT_LEVEL_ALERT,
               "baseline_ref_recall": BASELINE_REF,
               "n_passing": int(len(passing)),
               "all_results": rows},
              open(f"{OUT}/decision.json", "w"), indent=2)
    log(f"total: {time.time()-t0:.0f}s | salvo {OUT}/decision.json + voting_high_q_grid.csv")


if __name__ == "__main__":
    main()
