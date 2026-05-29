#!/usr/bin/env python
"""Validacao temporal do per-sensor (OR-de-17 univariados).

Treina 17 AEs em jan-ago (sem retuning), congela, mede curva preditiva em
set-dez. Se o ganho de +7pp do per-sensor sobre o multivariado in-sample
sobrevive aqui, viramos arquitetura. Se nao, era artefato.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import (
    load, make_seqs, build_ae, mse_per_seq, SENSORS, TIME_COL, RUNNING_COL,
)
from cnn1d_ae.predictive import (
    compute_health_index_ewma, pick_operating_point,
)

TS = 60; STRIDE = 10
F1_UNI = 4; F2_UNI = 1; S1_UNI = 2; S2_UNI = 2
HALF_LIFE_H = 4.0; DEBOUNCE_H = 8.0; GAP_H = 12.0
TRAIN_FRAC = 0.66
HORIZONS = [8, 24, 72]
FA_BUDGET = 1.0
OUT = "per_sensor_temporal_out"


def log(m): print(f"[PST] {m}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, inc_full = load(priority=None)
    n = len(df)
    n_train_pts = int(TRAIN_FRAC * n)
    t_split = df[TIME_COL].iloc[n_train_pts]
    log(f"split em {t_split}")

    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    incidents_train = inc_full[inc_full <= t_split]
    incidents_test = inc_full[inc_full > t_split]
    log(f"incidentes: train={len(incidents_train)} | test={len(incidents_test)}")

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

    is_train = t_end_pd <= t_split
    is_test = ~is_train

    # treino = negativos puros DENTRO do periodo de treino
    is_train_neg = is_train & seq_run_full & (dany > GAP_H)
    tr_pool = np.where(is_train_neg)[0]
    n_tr = int(0.9 * len(tr_pool))
    tr_idx = tr_pool[:n_tr]
    va_idx = tr_pool[n_tr:]
    rng = np.random.default_rng(42)
    if len(tr_idx) > 40000:
        tr_idx = rng.choice(tr_idx, 40000, replace=False)
    log(f"treino: {len(tr_idx)} negs (jan-ago, running, >12h de incidente)")

    dt_seconds = STRIDE * 30.0
    from tensorflow import keras

    per_sensor_health = {}
    t0_all = time.time()
    for si, sensor in enumerate(SENSORS):
        t0 = time.time()
        Xi = pd.to_numeric(df[sensor], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32).reshape(-1, 1)
        seqs_i, _ = make_seqs(Xi, TS, STRIDE)
        flat = seqs_i[tr_idx].reshape(-1, 1)
        mu = flat.mean(0); sd = flat.std(0); sd[sd==0] = 1.0
        norm = (seqs_i - mu) / sd

        model, latent = build_ae(TS, 1, F1_UNI, F2_UNI, S1_UNI, S2_UNI, 0.1, 1e-4)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
        model.fit(
            norm[tr_idx], norm[tr_idx],
            validation_data=(norm[va_idx], norm[va_idx]),
            epochs=15, batch_size=256, verbose=0, callbacks=cb,
        )
        mae_i = mse_per_seq(model, norm)
        health_i = compute_health_index_ewma(
            mae_i, seq_run_frac,
            half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds,
        )
        per_sensor_health[sensor] = health_i
        log(f"[{si+1:2d}/17] {sensor:14s} t={time.time()-t0:.1f}s")
        keras.backend.clear_session()
    log(f"17 modelos treinados em {time.time()-t0_all:.0f}s")

    # ---- thresholds ESCOLHIDOS no TREINO (jan-ago), APLICADOS no TESTE ----
    # Para cada quantil q: cada sensor tem seu threshold no q-esimo quantile do
    # seu health calculado SOMENTE no periodo de treino, depois aplica em teste.
    qs = np.linspace(0.50, 0.999, 40)
    span_days_test = max((t_end_sec[is_test].max() - t_end_sec[is_test].min()) / 86400.0, 1e-9)
    span_days_train = max((t_end_sec[is_train].max() - t_end_sec[is_train].min()) / 86400.0, 1e-9)
    inc_tr_s = pd.DatetimeIndex(incidents_train).values.astype("datetime64[s]").astype("int64").astype(float)
    inc_te_s = pd.DatetimeIndex(incidents_test).values.astype("datetime64[s]").astype("int64").astype(float)

    def evaluate_at_q(q, mask, span_days, incidents_s):
        thrs = {}
        for s, h in per_sensor_health.items():
            valid = h[is_train & seq_run_full]  # threshold sempre no TREINO
            thrs[s] = float(np.quantile(valid, q)) if len(valid) else float("inf")
        alert = np.zeros(len(t_end_sec), dtype=bool)
        for s, h in per_sensor_health.items():
            alert |= (h >= thrs[s])
        alert &= seq_run_full & mask
        idx = np.where(alert)[0]
        deb = DEBOUNCE_H * 3600
        episodes = []
        if len(idx):
            cur = [idx[0]]
            for j in idx[1:]:
                if t_end_sec[j] - t_end_sec[cur[-1]] <= deb:
                    cur.append(j)
                else:
                    episodes.append((t_end_sec[cur[0]], t_end_sec[cur[-1]]))
                    cur = [j]
            episodes.append((t_end_sec[cur[0]], t_end_sec[cur[-1]]))
        alert_s = t_end_sec[idx]
        out = {}
        for H in HORIZONS:
            Hs = H * 3600.0
            hits = 0; leads = []
            for ti in incidents_s:
                w = alert_s[(alert_s >= ti - Hs) & (alert_s <= ti)]
                if w.size:
                    hits += 1
                    leads.append((ti - w.min()) / 3600.0)
            recall = hits / len(incidents_s) if len(incidents_s) else 0.0
            fa = 0
            for (s0, s1) in episodes:
                useful = bool((((incidents_s - Hs) <= s1) & (incidents_s >= s0)).any()) if incidents_s.size else False
                if not useful:
                    fa += 1
            out[H] = dict(
                threshold_quantile=float(q),
                recall=float(recall),
                fa_per_day=float(fa / span_days),
                median_lead_hours=float(np.median(leads)) if leads else 0.0,
                n_episodes=int(len(episodes)),
            )
        return out

    # gera curvas in-sample (TREINO) e out-of-sample (TESTE)
    rows_tr = {H: [] for H in HORIZONS}
    rows_te = {H: [] for H in HORIZONS}
    for q in qs:
        tr = evaluate_at_q(q, is_train, span_days_train, inc_tr_s)
        te = evaluate_at_q(q, is_test, span_days_test, inc_te_s)
        for H in HORIZONS:
            rows_tr[H].append(tr[H])
            rows_te[H].append(te[H])

    results = {"train_in_sample": {}, "test_out_of_sample": {}}
    for H in HORIZONS:
        curve_tr = pd.DataFrame(rows_tr[H]).rename(columns={"threshold_quantile": "threshold"})
        curve_te = pd.DataFrame(rows_te[H]).rename(columns={"threshold_quantile": "threshold"})
        op_tr = pick_operating_point(curve_tr, FA_BUDGET)
        op_te = pick_operating_point(curve_te, FA_BUDGET)
        results["train_in_sample"][f"H{H}h"] = op_tr
        results["test_out_of_sample"][f"H{H}h"] = op_te
        curve_te.to_csv(f"{OUT}/per_sensor_OR_test_curve_H{H}h.csv", index=False)
        log(f"H={H:2d}h")
        log(f"  TRAIN: recall={op_tr['recall']:.2f} fa/d={op_tr['fa_per_day']:.2f} lead={op_tr['median_lead_hours']:.1f}h q={op_tr['threshold']:.3f}")
        log(f"  TEST : recall={op_te['recall']:.2f} fa/d={op_te['fa_per_day']:.2f} lead={op_te['median_lead_hours']:.1f}h q={op_te['threshold']:.3f}")

    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    log(f"salvo {OUT}/results.json + per_sensor_OR_test_curve_H{{8,24,72}}h.csv")


if __name__ == "__main__":
    main()
