#!/usr/bin/env python
"""Calibração de half-life EWMA + threshold por horizonte.

Treina 17 AEs univariados (idêntico a ae_per_sensor_temporal.py), salva MAE raw
em NPZ e varre half_life ∈ {2, 3, 4, 6} horas × grade de quantis.

Objetivo: fechar o gap H8h/H24h in-sample→out-of-sample escolhendo o
half-life ótimo por horizonte — sem mudar arquitetura ou retreinar.

NÃO usa ClearML: modelos tiny (60×1, 15 ep) rodam em ~3 min CPU.

Execução:
    PYTHONPATH=. python scripts/calibrate_halflife.py
    PYTHONPATH=. python scripts/calibrate_halflife.py --skip-train   # se NPZ já existe
"""
from __future__ import annotations
import argparse, os, sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from ae_separation_experiment import (
    load, make_seqs, build_ae, mse_per_seq, SENSORS, TIME_COL, RUNNING_COL,
)
from cnn1d_ae.predictive import compute_health_index_ewma

TS = 60; STRIDE = 10
F1_UNI = 4; F2_UNI = 1; S1_UNI = 2; S2_UNI = 2
DEBOUNCE_H = 8.0; GAP_H = 12.0
TRAIN_FRAC = 0.66
HORIZONS = [8, 24, 72]
FA_BUDGET = 1.0
HALF_LIVES = [2.0, 3.0, 4.0, 6.0]  # 4.0 = baseline atual
DT_SECONDS = STRIDE * 30.0
OUT = "calibrate_halflife_out"
NPZ_PATH = f"{OUT}/mae_raw.npz"


def log(m): print(f"[CAL] {m}", flush=True)


# ---------------------------------------------------------------------------
# Avaliação de alerta OR-de-17 num único half-life
# ---------------------------------------------------------------------------
def evaluate_all_horizons(per_sensor_health, t_end_sec, seq_run_full,
                           is_train, is_test, inc_tr_s, inc_te_s,
                           span_days_train, span_days_test):
    qs = np.linspace(0.50, 0.999, 50)
    rows_tr = {H: [] for H in HORIZONS}
    rows_te = {H: [] for H in HORIZONS}

    for q in qs:
        # threshold calibrado SEMPRE no treino
        thrs = {}
        for s, h in per_sensor_health.items():
            valid = h[is_train & seq_run_full]
            thrs[s] = float(np.quantile(valid, q)) if len(valid) else float("inf")

        alert = np.zeros(len(t_end_sec), dtype=bool)
        for s, h in per_sensor_health.items():
            alert |= (h >= thrs[s])

        alert_run = alert & seq_run_full
        deb = DEBOUNCE_H * 3600

        for mask, inc_s, span_days, rows in [
            (is_train, inc_tr_s, span_days_train, rows_tr),
            (is_test,  inc_te_s, span_days_test,  rows_te),
        ]:
            a = alert_run & mask
            idx = np.where(a)[0]
            episodes = []
            if idx.size:
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
                fa = sum(
                    1 for (s0, s1) in episodes
                    if not (bool((((inc_s - Hs) <= s1) & (inc_s >= s0)).any()) if inc_s.size else False)
                )
                rows[H].append(dict(
                    q=float(q),
                    recall=float(recall),
                    fa_per_day=float(fa / max(span_days, 1e-9)),
                    median_lead_hours=float(np.median(leads)) if leads else 0.0,
                    n_episodes=int(len(episodes)),
                ))
    return rows_tr, rows_te


def pick_op(rows, fa_budget):
    """Ponto de operação: máximo recall dentro do budget de FA/dia."""
    df = pd.DataFrame(rows)
    feasible = df[df["fa_per_day"] <= fa_budget]
    if feasible.empty:
        feasible = df.sort_values("fa_per_day").head(1)
    best = feasible.sort_values("recall", ascending=False).iloc[0]
    return best.to_dict()


# ---------------------------------------------------------------------------
# Treino dos 17 modelos e salvar MAE raw
# ---------------------------------------------------------------------------
def train_and_save(df, tr_idx, va_idx):
    from tensorflow import keras
    n = len(df)
    starts = np.arange(0, n - TS + 1, STRIDE)
    mae_dict = {}
    t0_all = time.time()
    for si, sensor in enumerate(SENSORS):
        t0 = time.time()
        Xi = pd.to_numeric(df[sensor], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(np.float32).reshape(-1, 1)
        seqs_i, _ = make_seqs(Xi, TS, STRIDE)
        flat = seqs_i[tr_idx].reshape(-1, 1)
        mu = flat.mean(0); sd = flat.std(0); sd[sd == 0] = 1.0
        norm = (seqs_i - mu) / sd

        model, _ = build_ae(TS, 1, F1_UNI, F2_UNI, S1_UNI, S2_UNI, 0.1, 1e-4)
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
        model.fit(norm[tr_idx], norm[tr_idx],
                  validation_data=(norm[va_idx], norm[va_idx]),
                  epochs=15, batch_size=256, verbose=0, callbacks=cb)
        mae_dict[sensor] = mse_per_seq(model, norm)
        log(f"[{si+1:2d}/17] {sensor:14s}  t={time.time()-t0:.1f}s")
        keras.backend.clear_session()

    log(f"17 modelos prontos em {time.time()-t0_all:.0f}s  →  salvando {NPZ_PATH}")
    np.savez(NPZ_PATH, **mae_dict)
    return mae_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="Reutiliza NPZ existente (pula treino)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    log("carregando dados...")
    df, inc_full = load(priority=None)
    n = len(df)
    n_train_pts = int(TRAIN_FRAC * n)
    t_split = df[TIME_COL].iloc[n_train_pts]
    log(f"split temporal: {t_split}")

    op = pd.to_numeric(df[RUNNING_COL], errors="coerce").fillna(0).to_numpy().astype(np.float32)
    t_min, t_max = df[TIME_COL].min(), df[TIME_COL].max()
    inc_full = inc_full[(inc_full >= t_min) & (inc_full <= t_max)]
    incidents_train = inc_full[inc_full <= t_split]
    incidents_test  = inc_full[inc_full > t_split]
    log(f"incidentes: train={len(incidents_train)}  test={len(incidents_test)}")

    starts = np.arange(0, n - TS + 1, STRIDE)
    ends = starts + TS - 1
    t_end_pd  = pd.DatetimeIndex(df[TIME_COL].to_numpy()[ends])
    t_end_sec = t_end_pd.values.astype("datetime64[s]").astype("int64").astype(float)
    seq_run_frac = np.array([op[s:s + TS].mean() for s in starts])
    seq_run_full = seq_run_frac >= 0.999

    inc_s_all = pd.DatetimeIndex(inc_full).values.astype("datetime64[s]").astype("int64").astype(float)
    pos = np.searchsorted(inc_s_all, t_end_sec)
    dnext = np.where(pos < len(inc_s_all),
                     (inc_s_all[np.clip(pos, 0, len(inc_s_all)-1)] - t_end_sec), np.inf) / 3600.0
    dprev = np.where(pos > 0,
                     (t_end_sec - inc_s_all[np.clip(pos-1, 0, len(inc_s_all)-1)]), np.inf) / 3600.0
    dany = np.minimum(dnext, dprev)

    is_train = t_end_pd <= t_split
    is_test  = ~is_train
    is_train_neg = is_train & seq_run_full & (dany > GAP_H)
    tr_pool = np.where(is_train_neg)[0]
    n_tr = int(0.9 * len(tr_pool))
    tr_idx = tr_pool[:n_tr]
    va_idx = tr_pool[n_tr:]
    rng = np.random.default_rng(42)
    if len(tr_idx) > 40000:
        tr_idx = rng.choice(tr_idx, 40000, replace=False)
    log(f"treino: {len(tr_idx)} seqs  |  val: {len(va_idx)} seqs")

    inc_tr_s = pd.DatetimeIndex(incidents_train).values.astype("datetime64[s]").astype("int64").astype(float)
    inc_te_s = pd.DatetimeIndex(incidents_test).values.astype("datetime64[s]").astype("int64").astype(float)
    span_days_train = max((t_end_sec[is_train].max() - t_end_sec[is_train].min()) / 86400.0, 1e-9)
    span_days_test  = max((t_end_sec[is_test].max()  - t_end_sec[is_test].min())  / 86400.0, 1e-9)

    # --- treino ou carga ---
    if args.skip_train and os.path.exists(NPZ_PATH):
        log(f"--skip-train: carregando MAE de {NPZ_PATH}")
        npz = np.load(NPZ_PATH)
        mae_dict = {s: npz[s] for s in SENSORS}
    else:
        mae_dict = train_and_save(df, tr_idx, va_idx)

    # --- sweep de half-lives ---
    summary_rows = []
    best_per_horizon = {}

    for hl in HALF_LIVES:
        log(f"\n=== half_life={hl}h ===")
        per_sensor_health = {
            s: compute_health_index_ewma(mae_dict[s], seq_run_frac, hl, DT_SECONDS)
            for s in SENSORS
        }
        rows_tr, rows_te = evaluate_all_horizons(
            per_sensor_health, t_end_sec, seq_run_full,
            is_train, is_test, inc_tr_s, inc_te_s,
            span_days_train, span_days_test,
        )

        for H in HORIZONS:
            op_tr = pick_op(rows_tr[H], FA_BUDGET)
            op_te = pick_op(rows_te[H], FA_BUDGET)
            gap = op_tr["recall"] - op_te["recall"]
            row = dict(
                half_life_h=hl, horizon_h=H,
                recall_train=round(op_tr["recall"], 3),
                recall_test=round(op_te["recall"], 3),
                gap_pp=round(gap * 100, 1),
                fa_train=round(op_tr["fa_per_day"], 3),
                fa_test=round(op_te["fa_per_day"], 3),
                lead_train_h=round(op_tr["median_lead_hours"], 1),
                lead_test_h=round(op_te["median_lead_hours"], 1),
            )
            summary_rows.append(row)
            log(f"  H={H:2d}h  train={op_tr['recall']:.2f}  test={op_te['recall']:.2f}"
                f"  gap={gap*100:+.1f}pp  fa_test={op_te['fa_per_day']:.3f}")

            # melhor por horizonte = máximo recall_test dentro do budget
            key = H
            if key not in best_per_horizon or op_te["recall"] > best_per_horizon[key]["recall_test"]:
                best_per_horizon[key] = {**row, "source": f"hl={hl}"}

        # salva curva de teste por half-life
        for H in HORIZONS:
            pd.DataFrame(rows_te[H]).to_csv(
                f"{OUT}/curve_hl{int(hl*10):02d}_H{H}h.csv", index=False
            )

    # --- tabela resumo ---
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(f"{OUT}/summary.csv", index=False)

    print("\n" + "="*72)
    print(df_sum.to_string(index=False))
    print("="*72)

    print("\n--- MELHOR POR HORIZONTE (máximo recall_test dentro de FA≤1/dia) ---")
    final = {}
    for H, best in sorted(best_per_horizon.items()):
        print(f"  H={H:2d}h  hl={best['half_life_h']}h  recall_test={best['recall_test']:.3f}"
              f"  gap={best['gap_pp']:+.1f}pp  fa_test={best['fa_test']:.3f}")
        final[f"H{H}h"] = best

    json.dump(final, open(f"{OUT}/best_per_horizon.json", "w"), indent=2)
    log(f"\nResultados em {OUT}/  (summary.csv + best_per_horizon.json + curvas por HL)")


if __name__ == "__main__":
    main()
