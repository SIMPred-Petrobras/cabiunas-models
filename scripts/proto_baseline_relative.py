"""PROTÓTIPO — detecção por desvio relativo ao baseline (vs threshold absoluto).

Hipótese (do diagnóstico): o EWMA-MAE de sensores como TC382_03/05 tem baseline
cronicamente alto e ondulante, então threshold horizontal não separa evento de
patamar. Aqui medimos um detector que responde à MUDANÇA, não ao nível:

  baseline = mediana móvel longa do EWMA (causal, trailing)
  escala   = IQR móvel / 1.349  (robusto)
  z        = (EWMA - baseline) / escala
  alerta   = z >= z_thr   (OFF excluído)

Compara recall × duty OOS 2024 contra o melhor threshold ABSOLUTO (duty_cycle_2024.csv),
sob a mesma regra: menor duty mantendo recall >= alvo.

Uso: PYTHONPATH=. python scripts/proto_baseline_relative.py \
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
HL = {"T5_AVG_A": 0.5, "TC382_03_A": 4.0, "TC382_05_A": 1.0, "TC382_06_A": 0.5}
RECALL_TARGET = 0.85
W0, W1 = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")


def recall_duty(alert_bool: pd.Series, on: np.ndarray, incidents, days):
    duty = (alert_bool.values & on).sum() / max(on.sum(), 1)
    al = alert_bool.copy()
    if STICKY > 0 and al.any():
        idx = al.index; res = al.values.copy(); td = pd.Timedelta(hours=STICKY)
        for p in np.where(res)[0]:
            res[p:idx.searchsorted(idx[p] + td, side="right")] = True
        al = pd.Series(res, index=idx)
    eps = E.detect_episodes_gap(al)
    als = np.array([t.timestamp() for t in al.index[al]]); hs = HORIZON * 3600
    inc_s = np.array([t.timestamp() for t in incidents])
    nh = sum(1 for ti in inc_s if als.size and np.any((als >= ti - hs) & (als <= ti)))
    rec = nh / len(incidents) if incidents else float("nan")
    return rec, duty


def pick(curve):
    """menor duty mantendo recall >= alvo; senão, maior recall."""
    ok = [c for c in curve if c[1] >= RECALL_TARGET]
    return min(ok, key=lambda c: c[2]) if ok else max(curve, key=lambda c: c[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_csv", required=True)
    ap.add_argument("--col_prefix", default="")
    ap.add_argument("--baseline_days", type=float, default=7.0)
    args = ap.parse_args()
    BASELINE_DAYS = args.baseline_days
    task = Task.get_task(task_id=TASK)
    alarms = E.load_alarms_gap(ALARM)
    try:
        absc = pd.read_csv("eval_predictive_out/duty_cycle_2024.csv")
    except FileNotFoundError:
        absc = None

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

    for s, hl in HL.items():
        bundle = load_bundle(f"production_bundles/{s}_inference_bundle.json")
        model = keras.models.load_model(task.artifacts[f"{s}_model_keras"].get_local_copy(), compile=False)
        sc = score_dataframe(model, bundle, df[[s, "NGP_A"]])
        mae = pd.Series(sc["mae_seq"].to_numpy(), index=pd.DatetimeIndex(sc["seq_end_time"]))
        dt_s = pd.Series(mae.index).diff().dt.total_seconds().median() or 300.0
        ew = mae.ewm(halflife=max(1, int(round(hl * 3600 / dt_s)))).mean()
        on = (df["NGP_A"].reindex(mae.index, method="nearest") > RUN_THR).to_numpy()
        days = (mae.index[-1] - mae.index[0]).total_seconds() / 86400.0
        inc = E.cluster_incidents([t for t in alarms.get(s, []) if W0 <= t <= W1], gap_hours=E.GAP_HOURS)
        on_i = df["NGP_A"].reindex(pd.DatetimeIndex(inc), method="nearest") > RUN_THR if inc else pd.Series([], dtype=bool)
        inc = [t for t, o in zip(inc, on_i.values) if o]

        # baseline relativo: mediana + IQR móvel causal (em pontos)
        W = max(50, int(round(BASELINE_DAYS * 24 * 3600 / dt_s)))
        med = ew.rolling(W, min_periods=W // 7).median()
        iqr = (ew.rolling(W, min_periods=W // 7).quantile(0.75) - ew.rolling(W, min_periods=W // 7).quantile(0.25))
        scale = (iqr / 1.349).replace(0, np.nan)
        z = ((ew - med) / scale).fillna(0.0)

        curve = []
        for zt in np.arange(0.25, 8.01, 0.25):
            rec, duty = recall_duty(pd.Series(z.to_numpy() >= zt, index=z.index), on, inc, days)
            curve.append((zt, rec, duty))
        zt, rec, duty = pick(curve)

        # melhor absoluto sob mesma regra
        abs_str = "n/d"
        if absc is not None:
            sub = absc[absc.sensor == s]
            cabs = [(r.q, r.recall, r.duty_cycle) for _, r in sub.iterrows()]
            if cabs:
                _, arec, aduty = pick(cabs)
                abs_str = f"recall {arec*100:.0f}% / duty {aduty*100:.0f}%"

        print(f"=== {s} (N={len(inc)}, hl={hl}h) ===")
        print(f"  RELATIVO (z-baseline): z*={zt:.1f}  recall {rec*100:.0f}% / duty {duty*100:.0f}%")
        print(f"  ABSOLUTO (melhor):                 {abs_str}")
        print()


if __name__ == "__main__":
    main()
