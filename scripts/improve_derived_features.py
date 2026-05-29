#!/usr/bin/env python
"""QUICK WIN #2: adicionar features derivadas como 'sensores virtuais'.

QW#1 (pressao) provou: adicionar MAIS sensores do mesmo tipo (valores brutos)
nao ajuda — informacao redundante.

Hipotese: features QUALITATIVAMENTE diferentes (gradiente = dinamica;
rolling std = volatilidade) capturam regimes que valores brutos perdem.

Plug-and-play com o per_sensor existente: cada feature derivada vira um AE
univariado adicional, entra no mesmo OR-de-quantile aggregation. Nao muda
arquitetura.

Para cada um dos 17 sensores original, cria 2 derivados:
- {sensor}_grad: derivada de primeira ordem (rate-of-change)
- {sensor}_std60: rolling std em janela de 60 pts (volatilidade)

Total: 17 + 17*2 = 51 AEs univariados.

Criterio: ganho >= +3pp recall OOS a H=8h. Senao, pivoto para teto informacional
e foco em buscar dados externos.
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
ROLL_WIN = 60  # janela de rolling std
OUT = "improve_derived_out"


def log(m): print(f"[DRV] {m}", flush=True)


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

    # ===== gera os "sensores virtuais" =====
    # Para cada sensor original, computa:
    #   _grad: np.gradient (derivada centrada)
    #   _std60: rolling std de 60 pts
    # Salva tudo num df_ext
    log(f"gerando features derivadas para {len(SENSORS)} sensores...")
    df_ext = df[[TIME_COL, RUNNING_COL]].copy()
    sensor_groups = {"orig": list(SENSORS), "grad": [], "std": []}
    for s in SENSORS:
        v = pd.to_numeric(df[s], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32)
        df_ext[s] = v
        df_ext[f"{s}_grad"] = np.gradient(v).astype(np.float32)
        df_ext[f"{s}_std{ROLL_WIN}"] = pd.Series(v).rolling(ROLL_WIN, min_periods=1).std().fillna(0).astype(np.float32)
        sensor_groups["grad"].append(f"{s}_grad")
        sensor_groups["std"].append(f"{s}_std{ROLL_WIN}")
    all_sensors = sensor_groups["orig"] + sensor_groups["grad"] + sensor_groups["std"]
    log(f"  total: {len(all_sensors)} canais (17 orig + 17 grad + 17 std)")

    op = pd.to_numeric(df_ext[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    starts = np.arange(0, n - TS + 1, STRIDE)
    ends = starts + TS - 1
    t_end_pd = pd.DatetimeIndex(df_ext[TIME_COL].to_numpy()[ends])
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
    inc_te_s = pd.DatetimeIndex(inc_te).values.astype("datetime64[s]").astype("int64").astype(float)
    log(f"incidentes: train={len(inc_tr)} | test={len(inc_te)}")

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

    # ===== treina 51 AEs univariados =====
    from tensorflow import keras
    health = np.empty((len(starts), len(all_sensors)), dtype=np.float32)
    t1 = time.time()
    for si, sensor in enumerate(all_sensors):
        Xi = df_ext[sensor].to_numpy(np.float32).reshape(-1, 1)
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
        if (si+1) % 10 == 0: log(f"  {si+1}/{len(all_sensors)}")
    log(f"{len(all_sensors)} modelos em {time.time()-t1:.0f}s")
    np.savez_compressed(f"{OUT}/health_derived.npz", health=health, sensors=all_sensors)

    # ===== thresholds em train period =====
    thr = np.array([float(np.quantile(health[is_train & seq_run_full, j], Q_OP))
                     for j in range(len(all_sensors))])
    above = (health >= thr[None, :])

    # ===== avaliacao: varios subsets =====
    print()
    print(f"{'modelo':<25}{'n_sens':>8}{'recall':>10}{'FA/d':>10}{'eps':>8}{'lead':>8}")
    print("-" * 69)
    results = {}
    subsets = [
        ("BASELINE 17 (orig)", sensor_groups["orig"]),
        ("ORIG + GRAD (34)", sensor_groups["orig"] + sensor_groups["grad"]),
        ("ORIG + STD (34)", sensor_groups["orig"] + sensor_groups["std"]),
        ("ORIG + GRAD + STD (51)", all_sensors),
        ("GRAD only (17)", sensor_groups["grad"]),
        ("STD only (17)", sensor_groups["std"]),
    ]
    for name, subset in subsets:
        mask = np.array([s in subset for s in all_sensors])
        alert_test_full = above[:, mask].any(axis=1) & seq_run_full & is_test
        alert_test = alert_test_full[is_test]
        m = evaluate_alerts(alert_test, t_end_sec[is_test],
                            inc_te_s, HORIZON, DEBOUNCE_H, span_days_test)
        print(f"{name:<25}{int(mask.sum()):>8d}{m['recall']:>10.2f}{m['fa_per_day']:>10.3f}"
              f"{m['n_episodes']:>8d}{m['median_lead_hours']:>8.1f}")
        results[name] = m

    # ===== decisao =====
    base = results["BASELINE 17 (orig)"]
    best_name, best_gain = None, -1.0
    for name, m in results.items():
        if name == "BASELINE 17 (orig)": continue
        gain = m["recall"] - base["recall"]
        if gain > best_gain and m["fa_per_day"] <= base["fa_per_day"] * 1.5:
            best_gain = gain; best_name = name
    log("")
    log(f"BASELINE: recall={base['recall']:.2f} FA/d={base['fa_per_day']:.3f}")
    if best_name:
        log(f"MELHOR ALTERNATIVA: {best_name} (delta recall {best_gain*100:+.1f}pp)")
        if best_gain >= 0.03:
            log(f"CRITERIO (+3pp recall): PASSOU ✅")
        else:
            log(f"CRITERIO (+3pp recall): NAO PASSOU ❌ (delta {best_gain*100:+.1f}pp insuficiente)")
    else:
        log("NENHUMA alternativa viavel encontrada")

    # incidentes pegos só pelo melhor combo
    if best_name and best_name in results:
        H_sec = HORIZON * 3600
        mask_best = np.array([s in [sub[1] for sub in subsets if sub[0] == best_name][0] for s in all_sensors])
        alert_base = above[:, np.array([s in sensor_groups["orig"] for s in all_sensors])].any(axis=1) & seq_run_full
        alert_best = above[:, mask_best].any(axis=1) & seq_run_full
        base_caught = 0; best_caught = 0; best_only = 0; base_only = 0
        for ti in inc_te_s:
            in_base = ((t_end_sec[alert_base] >= ti - H_sec) & (t_end_sec[alert_base] <= ti)).any()
            in_best = ((t_end_sec[alert_best] >= ti - H_sec) & (t_end_sec[alert_best] <= ti)).any()
            if in_base and in_best: base_caught += 1; best_caught += 1
            elif in_base: base_caught += 1; base_only += 1
            elif in_best: best_caught += 1; best_only += 1
        log(f"Incidentes pegos: BASELINE={base_caught} | {best_name}={best_caught}")
        log(f"  Pegos APENAS pelo melhor (ganho real): {best_only}")
        log(f"  Pegos APENAS pelo BASELINE (perdidos): {base_only}")

    json.dump({k: v for k, v in results.items()}, open(f"{OUT}/results.json", "w"), indent=2)
    log(f"total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
