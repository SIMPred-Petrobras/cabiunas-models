"""Validação FINAL OOS 2024 dos bundles deployados: lê o bloco production_alerting
de cada bundle em production_bundles/ (threshold absoluto + half-life DEPLOYÁVEIS) e
mede recall / FA-episódio / duty-cycle em 2024 (nunca visto). É o que produção faria.

Uso: PYTHONPATH=. python scripts/validate_deployed_2024.py \
       --sens_csv /home/thallys/Downloads/2024.csv --col_prefix bapiha02-
"""
import argparse
import numpy as np
import pandas as pd
from clearml import Task
from tensorflow import keras

import scripts.eval_per_sensor_level as E
from src.cnn1d_ae.inference import load_bundle, score_dataframe

ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
TASK = "58bc393c1d7a4e42815236e8897abc88"
RUN_THR, HORIZON, STICKY = 50.0, 8.0, 12.0
SENSORS = ["T5_AVG_A", "TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A", "TC382_06_A"]
W0, W1 = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_csv", required=True)
    ap.add_argument("--col_prefix", default="")
    args = ap.parse_args()
    task = Task.get_task(task_id=TASK)
    alarms = E.load_alarms_gap(ALARM)

    head = pd.read_csv(args.sens_csv, nrows=1)
    tcol = next(c for c in head.columns if "datetime" in c.lower() or c.lower() == "data")
    want = [tcol] + [args.col_prefix + c for c in SENSORS + ["NGP_A"] if args.col_prefix + c in head.columns]
    df = pd.read_csv(args.sens_csv, usecols=want, low_memory=False)
    df.columns = [c[len(args.col_prefix):] if c.startswith(args.col_prefix) else c for c in df.columns]
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    df = df[(df.index >= W0) & (df.index <= W1)]
    for c in SENSORS + ["NGP_A"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"VALIDAÇÃO FINAL OOS 2024 — bundles deployados (production_alerting)\n")
    print(f"{'sensor':11s} {'q':>5} {'hl':>4} {'absThr':>8} | {'recall':>6} {'FA/dia':>6} {'duty%':>6} {'N':>3}")
    rows = []
    for s in SENSORS:
        b = load_bundle(f"production_bundles/{s}_inference_bundle.json")
        pa = b["production_alerting"]; hl = float(pa["half_life_hours"]); athr = float(pa["ewma_abs_threshold"]); q = pa["threshold_q"]
        model = keras.models.load_model(task.artifacts[f"{s}_model_keras"].get_local_copy(), compile=False)
        sc = score_dataframe(model, b, df[[s, "NGP_A"]])
        mae = pd.Series(sc["mae_seq"].to_numpy(), index=pd.DatetimeIndex(sc["seq_end_time"]))
        dt = pd.Series(mae.index).diff().dt.total_seconds().median() or 300.0
        on = (df["NGP_A"].reindex(mae.index, method="nearest") > RUN_THR).to_numpy()
        ew = mae.ewm(halflife=max(1, int(round(hl * 3600 / dt)))).mean()
        raw = (ew.to_numpy() >= athr) & on
        duty = raw.sum() / max(on.sum(), 1)
        al = pd.Series(raw, index=ew.index)
        if al.any():
            idx = al.index; res = al.values.copy(); td = pd.Timedelta(hours=STICKY)
            for p in np.where(res)[0]:
                res[p:idx.searchsorted(idx[p] + td, side="right")] = True
            al = pd.Series(res, index=idx)
        eps = E.detect_episodes_gap(al); als = np.array([t.timestamp() for t in al.index[al]]); hs = HORIZON * 3600
        inc = E.cluster_incidents([t for t in alarms.get(s, []) if W0 <= t <= W1], gap_hours=E.GAP_HOURS)
        on_i = df["NGP_A"].reindex(pd.DatetimeIndex(inc), method="nearest") > RUN_THR if inc else pd.Series([], dtype=bool)
        inc = [t for t, o in zip(inc, on_i.values) if o]
        inc_s = np.array([t.timestamp() for t in inc])
        nh = sum(1 for ti in inc_s if als.size and np.any((als >= ti - hs) & (als <= ti)))
        nfp = sum(1 for s0, s1 in eps if not (np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())) if inc_s.size else False))
        days = (mae.index[-1] - mae.index[0]).total_seconds() / 86400.0
        rec = nh / len(inc) if inc else float("nan")
        fa = nfp / max(days, 1.0)
        rc = "  -" if np.isnan(rec) else f"{rec*100:4.0f}%"
        print(f"{s:11s} {q:5.3f} {hl:4.1f} {athr:8.5f} | {rc:>6} {fa:6.3f} {duty*100:5.1f}% {len(inc):3d}")
        rows.append({"sensor": s, "q": q, "hl": hl, "recall": rec, "fa_per_day": fa, "duty_cycle": duty, "n_inc": len(inc)})
    res = pd.DataFrame(rows); m = res[res.n_inc > 0]
    print(f"\nmacro: recall {m.recall.mean()*100:.1f}%  duty {m.duty_cycle.mean()*100:.1f}%  ({int(m.n_inc.sum())} incidentes, {len(m)} sensores)")
    res.to_csv("eval_predictive_out/validate_deployed_2024.csv", index=False)
    print("csv: eval_predictive_out/validate_deployed_2024.csv")


if __name__ == "__main__":
    main()
