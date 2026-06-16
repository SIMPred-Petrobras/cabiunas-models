"""Teste OUT-OF-SAMPLE: modelo treinado/calibrado em 2025 aplicado por inferência
em 2024-H2 (jun-dez, janela com NGP). Mede recall/FA com o ponto de operação FIXO
de 2025 (half-life por sensor + threshold ABSOLUTO de EWMA), OFF excluído — NÃO
re-otimiza em 2024. É a validação de generalização que faltava (N grande: TC382_03
tem 83 alarmes só em 2024-H2).

Uso:
  # 2024-H2 (default, arquivo do projeto):
  PYTHONPATH=. python scripts/eval_oos_2024.py
  # 2024 inteiro (arquivo com prefixo bapiha02- e NGP):
  PYTHONPATH=. python scripts/eval_oos_2024.py \
    --sens_csv /home/thallys/Downloads/2024.csv --col_prefix bapiha02- \
    --w0 2024-01-01 --w1 2024-12-31
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
SENS2024 = "../dados/sensores_filtrados_Interpolados_2024.csv"
SENS2025 = f"{DS}/sensores_filtrados_Interpolados_2025.csv"
TASK = "58bc393c1d7a4e42815236e8897abc88"
RUN_THR = 50.0
HORIZON, STICKY, FA_BUDGET, GAP = 8.0, 12.0, 1.0, 4.0
# half-life por sensor (do fechamento da frota 2025)
HL = {"T5_AVG_A": 0.5, "TC382_01_A": 0.5, "TC382_02_A": 2.0, "TC382_03_A": 4.0,
      "TC382_04_A": 0.5, "TC382_05_A": 1.0, "TC382_06_A": 0.5}
Q = 0.5  # threshold_q saturado (sensibilidade máxima sob FA budget)


def abs_thr_2025(mae2025: pd.Series, ngp2025: pd.Series, hl: float, dt_s: float) -> float:
    on = ngp2025.reindex(mae2025.index, method="nearest") > RUN_THR
    m = mae2025[on.values]
    hl_pts = max(1, int(round(hl * 3600.0 / dt_s)))
    return float(m.ewm(halflife=hl_pts).mean().quantile(Q))


def recall_fa(alert: pd.Series, incidents, days: float) -> tuple:
    al = E.apply_sticky(alert.astype(bool), 0.5, STICKY) if False else alert.astype(bool)
    # sticky simples: mantém alerta por STICKY horas após disparo
    if STICKY > 0 and al.any():
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
    return rec, nfp / max(days, 1.0), nhit, len(incidents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_csv", default=SENS2024)
    ap.add_argument("--col_prefix", default="", help="prefixo a remover das colunas (ex.: bapiha02-)")
    ap.add_argument("--w0", default="2024-06-01")
    ap.add_argument("--w1", default="2024-12-31")
    args = ap.parse_args()
    W0, W1 = pd.Timestamp(args.w0, tz="UTC"), pd.Timestamp(args.w1, tz="UTC")

    task = Task.get_task(task_id=TASK)
    alarms = E.load_alarms_gap(ALARM)
    mae2025_all = E.load_mae_series(task, list(HL))

    df24 = pd.read_csv(args.sens_csv, low_memory=False)
    if args.col_prefix:
        df24.columns = [c[len(args.col_prefix):] if c.startswith(args.col_prefix) else c for c in df24.columns]
    tcol = next((c for c in df24.columns if "datetime" in c.lower() or c.lower() in ("data", "time", "timestamp")), df24.columns[0])
    df24[tcol] = pd.to_datetime(df24[tcol], utc=True, errors="coerce")
    df24 = df24.dropna(subset=[tcol]).set_index(tcol).sort_index()
    df24 = df24[(df24.index >= W0) & (df24.index <= W1)]
    # tipos mistos (sentinelas/strings) → numérico nas colunas usadas
    for c in list(HL) + ["NGP_A"]:
        if c in df24.columns:
            df24[c] = pd.to_numeric(df24[c], errors="coerce")
    ngp25 = pd.read_csv(SENS2025, usecols=["data_datetime", "NGP_A"])
    ngp25["data_datetime"] = pd.to_datetime(ngp25["data_datetime"], utc=True, errors="coerce")
    ngp25 = ngp25.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()["NGP_A"]

    print(f"OOS ({W0.date()}..{W1.date()})  ponto de operação FIXO de 2025 (hl por sensor, q={Q}, OFF excl)\n")
    print(f"{'sensor':11s} {'hl':>4s} {'absThr':>8s} | {'recall':>6s} {'FA/dia':>6s} {'hit/N':>7s}")
    rows = []
    for s, hl in HL.items():
        bundle = load_bundle(f"production_bundles/{s}_inference_bundle.json") if s in ("T5_AVG_A", "TC382_04_A") \
            else load_bundle(task.artifacts[f"{s}_inference_bundle_json"].get_local_copy())
        model = keras.models.load_model(task.artifacts[f"{s}_model_keras"].get_local_copy(), compile=False)
        sub = df24[[s, "NGP_A"]].rename(columns={s: s})  # feature_columns=[s], running_col=NGP_A
        scored = score_dataframe(model, bundle, sub)
        mae24 = pd.Series(scored["mae_seq"].to_numpy(), index=pd.DatetimeIndex(scored["seq_end_time"]))
        dt_s = pd.Series(mae24.index).diff().dt.total_seconds().median() or 300.0
        athr = abs_thr_2025(mae2025_all[s], ngp25, hl, dt_s)
        # alerta OOS 2024: EWMA absoluta >= threshold de 2025, OFF excluído
        on24 = df24["NGP_A"].reindex(mae24.index, method="nearest") > RUN_THR
        hl_pts = max(1, int(round(hl * 3600.0 / dt_s)))
        ew = mae24.ewm(halflife=hl_pts).mean()
        alert = (ew >= athr) & on24.values
        # incidentes 2024-H2 ON
        inc = E.cluster_incidents([t for t in alarms.get(s, []) if W0 <= t <= W1], gap_hours=GAP)
        on_inc = df24["NGP_A"].reindex(pd.DatetimeIndex(inc), method="nearest") > RUN_THR if inc else pd.Series([], dtype=bool)
        inc_on = [t for t, o in zip(inc, on_inc.values) if o]
        days = (mae24.index[-1] - mae24.index[0]).total_seconds() / 86400.0
        rec, fa, nh, n = recall_fa(alert, inc_on, days)
        print(f"{s:11s} {hl:4.1f} {athr:8.5f} | {rec*100:5.1f}% {fa:6.3f} {nh:3d}/{n:<3d}")
        rows.append({"sensor": s, "hl": hl, "abs_thr": athr, "recall": rec, "fa_per_day": fa, "n_inc": n})
    df = pd.DataFrame(rows); m = df[df.n_inc > 0]
    print(f"\nrecall macro OOS: {m.recall.mean()*100:.1f}%  (n sensores={len(m)}, incidentes={int(m.n_inc.sum())})")
    df.to_csv("eval_predictive_out/oos_2024h2.csv", index=False)
    print("csv: eval_predictive_out/oos_2024h2.csv")


if __name__ == "__main__":
    main()
