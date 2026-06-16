"""Duty-cycle + sweep de threshold deployável, validado OOS em 2024.

O ponto de operação q=0.5 (piso da calibração) maximiza recall mas deixa o alarme
LIGADO grande parte do tempo — a métrica FA-por-episódio não penaliza isso. Aqui
varremos o quantil q (→ threshold absoluto calibrado em 2025) e medimos, em 2024 OOS:
recall, FA-episódio E **duty-cycle** (fração do tempo-ON com EWMA>=thr). Objetivo:
o menor duty-cycle que mantém recall alto = alarme operável.

Uso: PYTHONPATH=. python scripts/analyze_duty_cycle.py \
       --sens_csv /home/thallys/Downloads/2024.csv --col_prefix bapiha02-
"""
import argparse
import numpy as np
import pandas as pd
from clearml import Task
from tensorflow import keras

import scripts.eval_per_sensor_level as E
from src.cnn1d_ae.inference import load_bundle, score_dataframe

DS = "/home/thallys/.clearml/cache/storage_manager/datasets/ds_424e5b589e13402d9d95371a317e85c9"
ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
SENS2025 = f"{DS}/sensores_filtrados_Interpolados_2025.csv"
TASK = "58bc393c1d7a4e42815236e8897abc88"
RUN_THR, HORIZON, STICKY = 50.0, 8.0, 12.0
HL = {"T5_AVG_A": 0.5, "TC382_01_A": 0.5, "TC382_02_A": 2.0, "TC382_03_A": 4.0,
      "TC382_04_A": 0.5, "TC382_05_A": 1.0, "TC382_06_A": 0.5}
Q_GRID = [0.50, 0.80, 0.90, 0.95, 0.97, 0.99, 0.995]
W0, W1 = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")


def recall_fa_duty(ew, on, athr, incidents, days):
    raw = (ew.to_numpy() >= athr) & on
    duty = raw.sum() / max(on.sum(), 1)            # fração do tempo-ON em alerta (sem sticky)
    al = pd.Series(raw, index=ew.index)
    if STICKY > 0 and al.any():                     # sticky p/ recall/episódio
        idx = al.index; res = al.values.copy(); td = pd.Timedelta(hours=STICKY)
        for p in np.where(res)[0]:
            res[p:idx.searchsorted(idx[p] + td, side="right")] = True
        al = pd.Series(res, index=idx)
    eps = E.detect_episodes_gap(al)
    als = np.array([t.timestamp() for t in al.index[al]]); hs = HORIZON * 3600
    inc_s = np.array([t.timestamp() for t in incidents])
    nhit = sum(1 for ti in inc_s if als.size and np.any((als >= ti - hs) & (als <= ti)))
    nfp = sum(1 for s0, s1 in eps if not (np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())) if inc_s.size else False))
    rec = nhit / len(incidents) if incidents else float("nan")
    return rec, nfp / max(days, 1.0), duty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_csv", required=True)
    ap.add_argument("--col_prefix", default="")
    args = ap.parse_args()

    task = Task.get_task(task_id=TASK)
    alarms = E.load_alarms_gap(ALARM)
    mae2025 = E.load_mae_series(task, list(HL))
    ngp25 = pd.read_csv(SENS2025, usecols=["data_datetime", "NGP_A"])
    ngp25["data_datetime"] = pd.to_datetime(ngp25["data_datetime"], utc=True, errors="coerce")
    ngp25 = ngp25.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()["NGP_A"]

    head = pd.read_csv(args.sens_csv, nrows=1)
    tcol = next(c for c in head.columns if "datetime" in c.lower() or c.lower() == "data")
    want = [tcol] + [args.col_prefix + c for c in list(HL) + ["NGP_A"] if args.col_prefix + c in head.columns]
    df = pd.read_csv(args.sens_csv, usecols=want, low_memory=False)
    df.columns = [c[len(args.col_prefix):] if c.startswith(args.col_prefix) else c for c in df.columns]
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    df = df[(df.index >= W0) & (df.index <= W1)]
    for c in list(HL) + ["NGP_A"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for s, hl in HL.items():
        bundle = load_bundle(f"production_bundles/{s}_inference_bundle.json") if s in ("T5_AVG_A", "TC382_04_A") \
            else load_bundle(task.artifacts[f"{s}_inference_bundle_json"].get_local_copy())
        model = keras.models.load_model(task.artifacts[f"{s}_model_keras"].get_local_copy(), compile=False)
        scored = score_dataframe(model, bundle, df[[s, "NGP_A"]])
        mae24 = pd.Series(scored["mae_seq"].to_numpy(), index=pd.DatetimeIndex(scored["seq_end_time"]))
        dt_s = pd.Series(mae24.index).diff().dt.total_seconds().median() or 300.0
        hl_pts = max(1, int(round(hl * 3600 / dt_s)))
        on24 = (df["NGP_A"].reindex(mae24.index, method="nearest") > RUN_THR).to_numpy()
        ew24 = mae24.ewm(halflife=hl_pts).mean()
        days = (mae24.index[-1] - mae24.index[0]).total_seconds() / 86400.0
        inc = E.cluster_incidents([t for t in alarms.get(s, []) if W0 <= t <= W1], gap_hours=E.GAP_HOURS)
        on_inc = df["NGP_A"].reindex(pd.DatetimeIndex(inc), method="nearest") > RUN_THR if inc else pd.Series([], dtype=bool)
        inc_on = [t for t, o in zip(inc, on_inc.values) if o]
        # threshold calibrado em 2025 (OFF excl)
        m25 = mae2025[s]; on25 = ngp25.reindex(m25.index, method="nearest") > RUN_THR
        ew25 = m25[on25.values].ewm(halflife=hl_pts).mean()
        for q in Q_GRID:
            athr = float(ew25.quantile(q))
            rec, fa, duty = recall_fa_duty(ew24, on24, athr, inc_on, days)
            rows.append({"sensor": s, "q": q, "recall": rec, "fa_per_day": fa,
                         "duty_cycle": duty, "n_inc": len(inc_on)})

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    for s in HL:
        sub = res[res.sensor == s]
        n = int(sub["n_inc"].iloc[0])
        print(f"\n=== {s}  (hl={HL[s]}h, {n} incidentes OOS 2024) ===")
        print(f"{'q':>5} {'recall':>7} {'FA/dia':>7} {'duty%':>7}")
        for _, r in sub.iterrows():
            rc = "  -" if pd.isna(r.recall) else f"{r.recall*100:5.0f}%"
            print(f"{r.q:5.3f} {rc:>7} {r.fa_per_day:7.3f} {r.duty_cycle*100:6.1f}%")
    res.to_csv("eval_predictive_out/duty_cycle_2024.csv", index=False)
    print("\ncsv: eval_predictive_out/duty_cycle_2024.csv")


if __name__ == "__main__":
    main()
