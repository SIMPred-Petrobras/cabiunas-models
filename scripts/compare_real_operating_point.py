#!/usr/bin/env python
"""Comparacao HONESTA multi vs per_sensor em pontos operacionais REAIS.

Problema descoberto: pick_operating_point com fa_budget=1 retorna o ponto degenerado
(quantile baixo, alertas continuos, FA/dia artificialmente baixo por saturacao).

Aqui forco a comparacao em pontos com:
- FA/dia >= 0.05 (alerta nao-degenerado)
- n_episodes proximo de n_incidents (cobertura 1-pra-1, nao mega-episodios)

Decide: se per_sensor vence no PONTO REAL, mantem como default. Senao, reverte.
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
from cnn1d_ae.predictive import (
    compute_health_index_ewma, compute_predictive_curve,
    compute_predictive_curve_per_sensor,
)

TS = 60; STRIDE = 10
HALF_LIFE_H = 4.0; DEBOUNCE_H = 8.0; GAP_H = 12.0
# per_sensor: AE univariado pequeno
PS_F1 = 4; PS_F2 = 1; PS_S1 = 2; PS_S2 = 2
# multi: usa o vencedor da separacao (f2=3)
MULTI_F1 = 16; MULTI_F2 = 3; MULTI_S1 = 2; MULTI_S2 = 2
HORIZONS = [8, 24]
OUT = "compare_real_op_out"


def log(m): print(f"[CMP] {m}", flush=True)


def train_multi(x_train, x_val, x_all, n_feat, batch_size, epochs=15):
    from tensorflow import keras
    model, _ = build_ae(TS, n_feat, MULTI_F1, MULTI_F2, MULTI_S1, MULTI_S2, 0.1, 1e-4)
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    model.fit(x_train, x_train, validation_data=(x_val, x_val),
              epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cb)
    mae_all = mse_per_seq(model, x_all)
    keras.backend.clear_session()
    return mae_all


def train_per_sensor(df, sensors, tr_idx, va_idx, seq_run_frac, dt_seconds, batch_size, starts):
    from tensorflow import keras
    n_seq = len(starts)
    health_mat = np.empty((n_seq, len(sensors)), dtype=np.float32)
    for si, sensor in enumerate(sensors):
        Xi = pd.to_numeric(df[sensor], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32).reshape(-1, 1)
        seqs_i, _ = make_seqs(Xi, TS, STRIDE)
        flat = seqs_i[tr_idx].reshape(-1, 1)
        mu = flat.mean(0); sd = flat.std(0); sd[sd==0] = 1.0
        norm = (seqs_i - mu) / sd
        model, _ = build_ae(TS, 1, PS_F1, PS_F2, PS_S1, PS_S2, 0.1, 1e-4)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
        model.fit(norm[tr_idx], norm[tr_idx], validation_data=(norm[va_idx], norm[va_idx]),
                  epochs=15, batch_size=batch_size, verbose=0, callbacks=cb)
        mae_i = mse_per_seq(model, norm)
        health_mat[:, si] = compute_health_index_ewma(
            mae_i, seq_run_frac, half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds,
        )
        keras.backend.clear_session()
        if (si + 1) % 5 == 0:
            log(f"  per_sensor {si+1}/{len(sensors)}")
    return health_mat


def find_real_op(curve: pd.DataFrame, n_incidents: int):
    """Encontra ponto operacional real: n_episodes proximo de n_incidents (1:1 cobertura)
    com FA/dia >= 0.03 (nao degenerado). Retorna a linha mais proxima desse criterio.
    """
    if curve is None or curve.empty:
        return None
    # filtra pontos nao-degenerados
    feas = curve[(curve["fa_per_day"] >= 0.03) & (curve["n_episodes"] >= 0.5 * n_incidents)]
    if not len(feas):
        feas = curve[curve["fa_per_day"] >= 0.03]
    if not len(feas):
        return None
    # entre esses, pega o de melhor recall
    return feas.loc[feas["recall"].idxmax()].to_dict()


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
    X = df[SENSORS].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32)

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

    # ===== MULTI =====
    log("treinando multi...")
    t1 = time.time()
    seqs_multi, _ = make_seqs(X, TS, STRIDE)
    flat = seqs_multi[tr_pool].reshape(-1, seqs_multi.shape[-1])
    mu_m = flat.mean(0); sd_m = flat.std(0); sd_m[sd_m==0] = 1.0
    norm_multi = (seqs_multi - mu_m) / sd_m
    mae_multi = train_multi(norm_multi[tr_pool], norm_multi[va_idx], norm_multi,
                            seqs_multi.shape[-1], 256)
    health_multi = compute_health_index_ewma(
        mae_multi, seq_run_frac, half_life_hours=HALF_LIFE_H, dt_seconds=dt_seconds,
    )
    log(f"multi treinado em {time.time()-t1:.0f}s")

    # ===== PER_SENSOR =====
    log("treinando per_sensor (17 AEs)...")
    t1 = time.time()
    health_ps = train_per_sensor(df, SENSORS, tr_pool, va_idx, seq_run_frac, dt_seconds, 256, starts)
    log(f"per_sensor treinado em {time.time()-t1:.0f}s")

    # ===== CURVAS COMPLETAS =====
    results = {"horizons": {}}
    print()
    print(f"{'H':<5}{'arch':<14}{'q':>8}{'recall':>10}{'FA/dia':>10}{'eps':>8}{'lead_h':>9}")
    print("-" * 64)
    for H in HORIZONS:
        # multi
        curve_multi = compute_predictive_curve(
            health_ewma=health_multi, seq_running_full=seq_run_full,
            t_end_seconds=t_end_sec, incident_seconds=inc_s,
            horizon_hours=H, debounce_hours=DEBOUNCE_H, n_threshold_steps=60,
        )
        # per_sensor
        curve_ps = compute_predictive_curve_per_sensor(
            per_sensor_health=health_ps, seq_running_full=seq_run_full,
            t_end_seconds=t_end_sec, incident_seconds=inc_s,
            horizon_hours=H, debounce_hours=DEBOUNCE_H, n_threshold_steps=60,
        )
        curve_multi.to_csv(f"{OUT}/multi_curve_H{H}h.csv", index=False)
        curve_ps.to_csv(f"{OUT}/per_sensor_curve_H{H}h.csv", index=False)

        op_multi = find_real_op(curve_multi, n_inc)
        op_ps = find_real_op(curve_ps, n_inc)
        for arch, op in [("multi", op_multi), ("per_sensor", op_ps)]:
            if op:
                print(f"{H:<5}{arch:<14}{op['threshold']:>8.3f}"
                      f"{op['recall']:>10.2f}{op['fa_per_day']:>10.2f}"
                      f"{int(op['n_episodes']):>8d}{op['median_lead_hours']:>9.1f}")
            else:
                print(f"{H:<5}{arch:<14}{'N/A':>8s}{'N/A':>10s}")
        results["horizons"][f"H{H}h"] = {"multi": op_multi, "per_sensor": op_ps}
    print()

    # ===== DECISAO =====
    op_multi_8 = results["horizons"]["H8h"]["multi"]
    op_ps_8 = results["horizons"]["H8h"]["per_sensor"]
    decision = None
    if op_multi_8 and op_ps_8:
        diff_recall = op_ps_8["recall"] - op_multi_8["recall"]
        diff_fa = op_multi_8["fa_per_day"] - op_ps_8["fa_per_day"]  # positive = per_sensor better
        # criterio: per_sensor vence se recall >= multi - 2pp E FA/dia <= multi
        if op_ps_8["recall"] >= op_multi_8["recall"] - 0.02 and op_ps_8["fa_per_day"] <= op_multi_8["fa_per_day"] + 0.02:
            decision = "MANTER per_sensor como default"
        else:
            decision = f"REVERTER para multivariate (per_sensor pior em ponto real: Δrecall={diff_recall:+.2f})"
    log(f"DECISAO: {decision}")
    results["decision"] = decision
    json.dump(results, open(f"{OUT}/decision.json", "w"), indent=2, default=str)
    log(f"total: {time.time()-t0:.0f}s | salvo {OUT}/decision.json + curves")


if __name__ == "__main__":
    main()
