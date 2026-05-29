#!/usr/bin/env python
"""Experimento: 17 AEs univariados (1 por sensor) vs 1 AE multivariado.

Cada AE detecta no seu canal; combina alertas via OR-de-qualquer-um.
Compara curva preditiva (H=8/24/72h) com o multivariado pra decidir
se modelos por sensor entregam recall maior ao custo de 17x complexidade.
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

# config univariado: bottleneck agressivo (input pequeno: 60×1 = 60 pts)
TS = 60; STRIDE = 10
F1_UNI = 4; F2_UNI = 1; S1_UNI = 2; S2_UNI = 2  # latent = 60/4=15 → ratio 0.25 (s1*s2 deve dividir TS)
HALF_LIFE_H = 4.0; DEBOUNCE_H = 8.0; GAP_H = 12.0
HORIZONS = [8, 24, 72]
FA_BUDGET = 1.0
OUT = "per_sensor_out"


def log(m): print(f"[PSE] {m}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, inc_full = load(priority=None)
    n = len(df)
    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)

    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    log(f"incidentes (range): {len(inc_full)}")

    # tempos das sequências (mesmo conjunto pra todos os sensores)
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
    n_tr = int(0.9 * len(neg_idx))
    tr_pool = neg_idx[:n_tr]
    va_idx = neg_idx[n_tr:]
    rng = np.random.default_rng(42)
    if len(tr_pool) > 40000:
        tr_pool = rng.choice(tr_pool, 40000, replace=False)
    log(f"treino: {len(tr_pool)} negs por sensor (running, >12h de incidente)")

    dt_seconds = STRIDE * 30.0
    from tensorflow import keras

    per_sensor_health = {}
    per_sensor_meta = {}
    t0_all = time.time()
    for si, sensor in enumerate(SENSORS):
        t0 = time.time()
        Xi = pd.to_numeric(df[sensor], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32).reshape(-1, 1)
        seqs_i, _ = make_seqs(Xi, TS, STRIDE)
        flat = seqs_i[tr_pool].reshape(-1, 1)
        mu = flat.mean(0); sd = flat.std(0); sd[sd==0] = 1.0
        norm = (seqs_i - mu) / sd

        model, latent = build_ae(TS, 1, F1_UNI, F2_UNI, S1_UNI, S2_UNI, 0.1, 1e-4)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
        model.fit(
            norm[tr_pool], norm[tr_pool],
            validation_data=(norm[va_idx], norm[va_idx]),
            epochs=15, batch_size=256, verbose=0, callbacks=cb,
        )
        mae_i = mse_per_seq(model, norm)
        health_i = compute_health_index_ewma(
            mae_i, seq_run_frac,
            half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds,
        )
        per_sensor_health[sensor] = health_i
        per_sensor_meta[sensor] = dict(latent=int(latent), train_secs=round(time.time()-t0, 1))
        log(f"[{si+1:2d}/17] {sensor:14s} latente={latent:3d} ratio={latent/TS:.2f} | t={time.time()-t0:.1f}s")
        keras.backend.clear_session()
    log(f"17 modelos treinados em {time.time()-t0_all:.0f}s")

    # combinacao OR: cada sensor com seu proprio threshold no quantile q (varredura)
    qs = np.linspace(0.50, 0.999, 40)
    span_days = max((t_end_sec.max() - t_end_sec.min()) / 86400.0, 1e-9)

    rows = {H: [] for H in HORIZONS}
    for q in qs:
        thrs = {}
        for s, h in per_sensor_health.items():
            valid = h[seq_run_full]
            thrs[s] = float(np.quantile(valid, q)) if len(valid) else float("inf")
        alert = np.zeros(len(t_end_sec), dtype=bool)
        for s, h in per_sensor_health.items():
            alert |= (h >= thrs[s])
        alert &= seq_run_full
        idx = np.where(alert)[0]
        # episodios debounced
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
        for H in HORIZONS:
            Hs = H * 3600.0
            hits = 0; leads = []
            for ti in inc_s:
                w = alert_s[(alert_s >= ti - Hs) & (alert_s <= ti)]
                if w.size:
                    hits += 1
                    leads.append((ti - w.min()) / 3600.0)
            recall = hits / len(inc_s) if len(inc_s) else 0.0
            fa = 0
            for (s0, s1) in episodes:
                useful = bool((((inc_s - Hs) <= s1) & (inc_s >= s0)).any()) if inc_s.size else False
                if not useful:
                    fa += 1
            rows[H].append(dict(
                threshold_quantile=float(q),
                recall=float(recall),
                fa_per_day=float(fa / span_days),
                median_lead_hours=float(np.median(leads)) if leads else 0.0,
                n_episodes=int(len(episodes)),
            ))

    results = {"sensor_meta": per_sensor_meta, "operating_points": {}}
    for H in HORIZONS:
        curve = pd.DataFrame(rows[H])
        # adapta col name para compat com pick_operating_point (que espera 'threshold')
        curve_for_pick = curve.rename(columns={"threshold_quantile": "threshold"})
        op_pt = pick_operating_point(curve_for_pick, FA_BUDGET)
        results["operating_points"][f"H{H}h"] = op_pt
        curve.to_csv(f"{OUT}/per_sensor_OR_curve_H{H}h.csv", index=False)
        if op_pt:
            log(f"H={H:2d}h | OR-de-17 (fa/dia<=1): recall={op_pt['recall']:.2f} "
                f"fa/d={op_pt['fa_per_day']:.2f} lead={op_pt['median_lead_hours']:.1f}h "
                f"q={op_pt['threshold']:.3f}")
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    log(f"salvo {OUT}/per_sensor_OR_curve_H{{8,24,72}}h.csv e results.json")


if __name__ == "__main__":
    main()
