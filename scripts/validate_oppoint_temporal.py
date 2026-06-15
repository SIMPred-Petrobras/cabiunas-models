"""Validação temporal do ponto de operação (half-life + threshold) por sensor.

Calibra (half-life, threshold_q) maximizando recall sob orçamento de FA nos
incidentes ANTES de --split_date e mede recall/FA nos incidentes DEPOIS (não vistos
na calibração). Resposta honesta para "o ganho de half-life generaliza ou é overfit
de ponto de operação?". OFF (NGP <= RUN_THR) é sempre excluído (fora de escopo).

Uso:
  PYTHONPATH=. python scripts/validate_oppoint_temporal.py \
    --task_id 58bc393c1d7a4e42815236e8897abc88 --split_date 2025-05-01 \
    --sensors T5_AVG_A TC382_03_A TC382_04_A
"""
import argparse
import numpy as np
import pandas as pd
from clearml import Task

import scripts.eval_per_sensor_level as E

DS = "/home/thallys/.clearml/cache/storage_manager/datasets/ds_424e5b589e13402d9d95371a317e85c9"
ALARM = f"{DS}/alarmes_record_2025_tags_modelo.csv"
RAWCSV = f"{DS}/sensores_filtrados_Interpolados_2025.csv"
RUN_THR = 50
HORIZON, STICKY = 8.0, 12.0


def ngp_series() -> pd.Series:
    raw = pd.read_csv(RAWCSV, usecols=["data_datetime", "NGP_A"])
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    return raw["NGP_A"]


def metrics_at(health: pd.Series, inc, q: float):
    """recall/FA com half-life já aplicado em `health`, threshold_q fixo, sticky+debounce."""
    if len(health) == 0:
        return float("nan"), float("nan")
    al = E.apply_sticky(health, q, STICKY)
    eps = E.detect_episodes_gap(al)
    als = np.array([t.timestamp() for t in health.index[al]])
    hs = HORIZON * 3600
    inc_s = np.array([t.timestamp() for t in inc])
    nhit = sum(1 for ti in inc_s if als.size and np.any((als >= ti - hs) & (als <= ti)))
    nfp = sum(1 for s0, s1 in eps
              if not (np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())) if inc_s.size else False))
    days = (health.index[-1] - health.index[0]).total_seconds() / 86400.0
    rec = nhit / len(inc) if len(inc) else float("nan")
    return rec, nfp / max(days, 1.0)


def mae_of(task: Task, sensor: str):
    key = next((k for k in task.artifacts if "sequence_scores_all" in k and sensor in k), None)
    if key is None:
        return None
    d = pd.read_csv(task.artifacts[key].get_local_copy())
    d["seq_start_time"] = pd.to_datetime(d["seq_start_time"], utc=True, errors="coerce")
    return d.dropna(subset=["seq_start_time"]).sort_values("seq_start_time").set_index("seq_start_time")["mae_seq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="58bc393c1d7a4e42815236e8897abc88")
    ap.add_argument("--split_date", default="2025-05-01")
    ap.add_argument("--sensors", nargs="*", default=["T5_AVG_A", "TC382_03_A", "TC382_04_A"])
    ap.add_argument("--hl_grid", nargs="*", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--fa_budget", type=float, default=1.0)
    args = ap.parse_args()

    split = pd.Timestamp(args.split_date, tz="UTC")
    ngp = ngp_series()
    alarms = E.load_alarms_gap(ALARM)
    task = Task.get_task(task_id=args.task_id)

    print(f"split={split.date()}  modelo={args.task_id[:8]}  (calibra ANTES, testa DEPOIS; OFF excluido)\n")
    print(f"{'sensor':11s} {'hl*':>4s} {'q*':>5s} | {'tr_rec':>6s} {'tr_FA':>6s} {'tr_n':>4s} | {'TE_rec':>6s} {'TE_FA':>6s} {'TE_n':>4s}")
    for s in args.sensors:
        mae = mae_of(task, s)
        if mae is None:
            print(f"{s:11s}  sem artefato"); continue
        inc = E.cluster_incidents(alarms.get(s, []), gap_hours=E.GAP_HOURS)
        on = ngp.reindex(pd.DatetimeIndex(inc), method="nearest") > RUN_THR
        inc_on = [x for x, o in zip(inc, on.values) if o]
        tr_inc = [t for t in inc_on if t < split]
        te_inc = [t for t in inc_on if t >= split]

        best = None
        for hl in args.hl_grid:
            h = E.ewma_quantile(mae, hl)
            on_h = ngp.reindex(h.index, method="nearest") > RUN_THR
            h_tr = h.where(on_h).dropna()
            h_tr = h_tr[h_tr.index < split]
            for q in np.linspace(0.50, 0.999, 120):
                rec, fa = metrics_at(h_tr, tr_inc, q)
                if np.isnan(rec) or fa > args.fa_budget:
                    continue
                key = (rec, -fa)
                if best is None or key > best[0]:
                    best = (key, hl, q, rec, fa)
        if best is None:
            print(f"{s:11s}  sem ponto valido no treino"); continue
        _, hl, q, trrec, trfa = best
        h = E.ewma_quantile(mae, hl)
        on_h = ngp.reindex(h.index, method="nearest") > RUN_THR
        h_te = h.where(on_h).dropna()
        h_te = h_te[h_te.index >= split]
        terec, tefa = metrics_at(h_te, te_inc, q)
        print(f"{s:11s} {hl:4.1f} {q:5.3f} | {trrec*100:5.1f}% {trfa:6.3f} {len(tr_inc):4d} | "
              f"{terec*100:5.1f}% {tefa:6.3f} {len(te_inc):4d}")


if __name__ == "__main__":
    main()
