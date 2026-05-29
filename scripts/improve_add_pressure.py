#!/usr/bin/env python
"""QUICK WIN #1: adicionar sensores de pressao ao per_sensor.

Hipotese: ~42% dos incidentes nao detectados em 8h podem ser falhas
hidraulicas/gas que nossos 17 sensores (T + V) nao capturam.

Adiciona 9 sensores de pressao (PI_*, PDI_*) que ja estao no CSV bruto
mas nao no modelo. Compara com baseline (17 sensores) na MESMA validacao
temporal (jan-ago → set-dez).

Criterio de sucesso: ganho >= +3pp recall OOS a H=8h.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from ae_separation_experiment import load, make_seqs, build_ae, mse_per_seq, SENSORS, TIME_COL, RUNNING_COL
from cnn1d_ae.predictive import compute_health_index_ewma, _detect_episodes

TS = 60; STRIDE = 10
F1 = 4; F2 = 1; S1 = 2; S2 = 2
HL = 4.0; GAP_H = 12.0; DEBOUNCE_H = 8.0
Q_OP = 0.715; HORIZON = 8.0
TRAIN_FRAC = 0.66
OUT = "improve_pressure_out"

# 9 sensores de pressao com sinal real (std > 0.05)
PRESSURE = [
    "954005_624_PI_0315", "954005_624_PI_0319",
    "954005_624_PI_0340", "954005_624_PI_0339",
    "954005_624_PI_0307", "954005_624_PI_0308",
    "954005_624_PDI_0317", "954005_624_PDI_0302",
    "954005_624_PDI_0338",
]
SENSORS_EXPANDED = SENSORS + PRESSURE  # 17 + 9 = 26


def log(m): print(f"[PRES] {m}", flush=True)


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
    n_inc = len(inc_full)
    n = len(df)
    n_train_pts = int(TRAIN_FRAC * n)
    t_split = df[TIME_COL].iloc[n_train_pts]
    log(f"split: {t_split}")
    log(f"sensores: 17 original + {len(PRESSURE)} pressao = {len(SENSORS_EXPANDED)} total")

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
    inc_tr = inc_full[inc_full <= t_split]
    inc_te = inc_full[inc_full > t_split]
    inc_tr_s = pd.DatetimeIndex(inc_tr).values.astype("datetime64[s]").astype("int64").astype(float)
    inc_te_s = pd.DatetimeIndex(inc_te).values.astype("datetime64[s]").astype("int64").astype(float)
    log(f"incidentes: train={len(inc_tr)} | test={len(inc_te)}")
    # train period: negativos puros no train period
    is_neg = (dany > GAP_H) & seq_run_full & is_train
    neg_idx = np.where(is_neg)[0]
    n_tr = int(0.9 * len(neg_idx))
    tr_pool = neg_idx[:n_tr]; va_idx = neg_idx[n_tr:]
    rng = np.random.default_rng(42)
    if len(tr_pool) > 40000:
        tr_pool = rng.choice(tr_pool, 40000, replace=False)
    log(f"treino: {len(tr_pool)} negs no train period")
    dt_seconds = STRIDE * 30.0
    span_days_test = max((t_end_sec[is_test].max() - t_end_sec[is_test].min()) / 86400.0, 1e-9)

    from tensorflow import keras
    health = np.empty((len(starts), len(SENSORS_EXPANDED)), dtype=np.float32)
    t1 = time.time()
    for si, sensor in enumerate(SENSORS_EXPANDED):
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
                                                  half_life_hours=HL, dt_seconds=dt_seconds)
        keras.backend.clear_session()
        if (si+1) % 5 == 0: log(f"  {si+1}/{len(SENSORS_EXPANDED)}")
    log(f"{len(SENSORS_EXPANDED)} modelos em {time.time()-t1:.0f}s")
    np.savez_compressed(f"{OUT}/health_expanded.npz", health=health, sensors=SENSORS_EXPANDED)

    # threshold uniforme q=0.715 calibrado em TRAIN
    thr = np.array([float(np.quantile(health[is_train & seq_run_full, j], Q_OP))
                     for j in range(len(SENSORS_EXPANDED))])
    above = (health >= thr[None, :])

    print()
    print(f"{'modelo':<22}{'n_sens':>8}{'recall':>10}{'FA/d':>10}{'eps':>8}{'lead':>8}")
    print("-" * 66)
    results = {}
    for name, sensor_mask in [
        ("BASELINE 17 (T+V)", np.array([s in SENSORS for s in SENSORS_EXPANDED])),
        ("EXPANDED 26 (+P)", np.ones(len(SENSORS_EXPANDED), dtype=bool)),
        ("PRESSURE only 9", np.array([s in PRESSURE for s in SENSORS_EXPANDED])),
    ]:
        # OR sobre apenas os sensores do subset
        above_sub = above[:, sensor_mask].any(axis=1)
        alert_test = above_sub & seq_run_full & is_test
        alert_test_sliced = alert_test[is_test]
        m = evaluate_alerts(alert_test_sliced, t_end_sec[is_test],
                            inc_te_s, HORIZON, DEBOUNCE_H, span_days_test)
        print(f"{name:<22}{int(sensor_mask.sum()):>8d}{m['recall']:>10.2f}{m['fa_per_day']:>10.3f}"
              f"{m['n_episodes']:>8d}{m['median_lead_hours']:>8.1f}")
        results[name] = m

    # delta
    base = results["BASELINE 17 (T+V)"]; expanded = results["EXPANDED 26 (+P)"]
    drecall = expanded["recall"] - base["recall"]
    dfa = expanded["fa_per_day"] - base["fa_per_day"]
    log(f"")
    log(f"DELTA expanded vs baseline (OOS):")
    log(f"  recall: {drecall:+.2f}pp ({drecall*100:+.1f}pp)")
    log(f"  FA/d: {dfa:+.4f}")
    crit = drecall >= 0.03
    log(f"CRITERIO (+3pp recall): {'PASSOU ✅' if crit else 'NAO PASSOU ❌'}")

    # check de incidentes especificos: quais O EXPANDED pega que BASELINE perde?
    alert_base = above[:, np.array([s in SENSORS for s in SENSORS_EXPANDED])].any(axis=1) & seq_run_full
    alert_exp = above.any(axis=1) & seq_run_full
    H_sec = HORIZON * 3600
    base_caught = 0; exp_caught = 0; exp_only = 0; base_only = 0
    for ti in inc_te_s:
        in_base = ((t_end_sec[alert_base] >= ti - H_sec) & (t_end_sec[alert_base] <= ti)).any()
        in_exp = ((t_end_sec[alert_exp] >= ti - H_sec) & (t_end_sec[alert_exp] <= ti)).any()
        if in_base and in_exp: base_caught += 1; exp_caught += 1
        elif in_base: base_caught += 1; base_only += 1
        elif in_exp: exp_caught += 1; exp_only += 1
    log(f"Incidentes pegos: BASELINE={base_caught} | EXPANDED={exp_caught}")
    log(f"  Pegos APENAS por EXPANDED (ganho da pressao): {exp_only}")
    log(f"  Pegos APENAS por BASELINE (perdidos por adicao): {base_only}")

    json.dump({k: v for k, v in results.items()}, open(f"{OUT}/results.json", "w"), indent=2)
    log(f"total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
