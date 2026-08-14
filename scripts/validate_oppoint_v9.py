"""Validação temporal do ponto de operação — task 2c9ccb1d (v9 SENTINEL 500).

Sem vazamento: (half-life, threshold) calibrados SÓ com dados/incidentes ANTES de
--split_date. O threshold é o quantile-q do EWMA em ON no período de calibração,
convertido para valor ABSOLUTO (como o finalize de produção: thr_q → ewma_abs) e
aplicado intacto ao período de teste. Incidentes = HI/HIHI agrupados (gap 4h),
onset em ON (RUNNING_A > 0.5); OFF excluído do denominador.

Uso:
  PYTHONPATH=. python scripts/validate_oppoint_v9.py [--split_date 2025-07-01]
"""
import argparse
import numpy as np
import pandas as pd
from clearml import Task

import scripts.eval_per_sensor_level as E

TASK_ID = "2c9ccb1d309c4537a2f02ae1239663d9"
ALARM = "../dados/alarmes_selecionados_turbina_a.csv"
RAWCSV = "../dados/sensores_brutos_2025_2026_30s.csv"
EXCLUDE_CONDS = ["UNDER", "CFN", "LOLO", "OVER"]
HORIZON, STICKY = 8.0, 12.0
SENSORS7 = ["T5_AVG_A", "TC382_01_A", "TC382_02_A", "TC382_03_A",
            "TC382_04_A", "TC382_05_A", "TC382_06_A"]


def running_series() -> pd.Series:
    # low_memory=False + seleção posterior: usecols quebra com chunks de dtype misto
    raw = pd.read_csv(RAWCSV, low_memory=False)[["data_datetime", "RUNNING_A"]]
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    return pd.to_numeric(raw["RUNNING_A"], errors="coerce").fillna(0.0)


def metrics_at_abs(ew: pd.Series, inc, thr: float):
    """recall/FA com threshold ABSOLUTO no EWMA (sticky+debounce como produção)."""
    if len(ew) == 0:
        return float("nan"), float("nan")
    al = E.apply_sticky(ew, thr, STICKY)
    eps = E.detect_episodes_gap(al)
    als = np.array([t.timestamp() for t in ew.index[al]])
    hs = HORIZON * 3600
    inc_s = np.array([t.timestamp() for t in inc])
    nhit = sum(1 for ti in inc_s if als.size and np.any((als >= ti - hs) & (als <= ti)))
    nfp = sum(1 for s0, s1 in eps
              if not (np.any((inc_s - hs <= s1.timestamp()) & (inc_s >= s0.timestamp())) if inc_s.size else False))
    days = (ew.index[-1] - ew.index[0]).total_seconds() / 86400.0
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
    ap.add_argument("--task_id", default=TASK_ID)
    ap.add_argument("--split_date", default="2025-07-01")
    ap.add_argument("--sensors", nargs="*", default=SENSORS7)
    ap.add_argument("--hl_grid", nargs="*", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--fa_budget", type=float, default=1.0)
    ap.add_argument("--max_duty", type=float, default=0.35)
    args = ap.parse_args()

    split = pd.Timestamp(args.split_date, tz="UTC")
    running = running_series()
    alarms = E.load_alarms_gap(ALARM, exclude_conditions=EXCLUDE_CONDS)
    task = Task.get_task(task_id=args.task_id)

    print(f"split={split.date()}  modelo={args.task_id[:8]}  HI/HIHI-only, threshold absoluto "
          f"(calibra ANTES, testa DEPOIS; OFF excluido)\n")
    print(f"{'sensor':11s} {'hl*':>4s} {'q*':>5s} | {'tr_rec':>6s} {'tr_FA':>6s} {'tr_n':>4s} | "
          f"{'TE_rec':>6s} {'TE_FA':>6s} {'TE_n':>4s}")
    rows = []
    for s in args.sensors:
        mae = mae_of(task, s)
        if mae is None:
            print(f"{s:11s}  sem artefato"); continue
        raw_al = [t for t in alarms.get(s, []) if mae.index.min() <= t <= mae.index.max()]
        inc = E.cluster_incidents(raw_al, gap_hours=E.GAP_HOURS)
        if inc:
            on = running.reindex(pd.DatetimeIndex(inc), method="nearest") > 0.5
            inc_on = [x for x, o in zip(inc, on.values) if o]
        else:
            inc_on = []
        tr_inc = [t for t in inc_on if t < split]
        te_inc = [t for t in inc_on if t >= split]
        if not tr_inc:
            print(f"{s:11s}    -     - |     -      - {len(tr_inc):4d} |     -      - {len(te_inc):4d}  (sem incidente p/ calibrar)")
            rows.append(dict(sensor=s, n_tr=len(tr_inc), n_te=len(te_inc)))
            continue

        ew_by_hl = {}
        for hl in args.hl_grid:
            hl_pts = max(1, int(round(pd.Timedelta(hours=hl) / pd.Timedelta(E.SAMPLING_INTERVAL))))
            ew = mae.ewm(halflife=hl_pts).mean()
            on_h = running.reindex(ew.index, method="nearest") > 0.5
            ew_by_hl[hl] = ew.where(on_h).dropna()

        best = None
        for hl, ew_on in ew_by_hl.items():
            ew_tr = ew_on[ew_on.index < split]
            if ew_tr.empty:
                continue
            for q in np.linspace(0.50, 0.999, 120):
                thr = float(ew_tr.quantile(q))
                if (ew_tr >= thr).mean() > args.max_duty:
                    continue
                rec, fa = metrics_at_abs(ew_tr, tr_inc, thr)
                if np.isnan(rec) or fa > args.fa_budget:
                    continue
                key = (rec, -fa)
                if best is None or key > best[0]:
                    best = (key, hl, q, thr, rec, fa)
        if best is None:
            print(f"{s:11s}  sem ponto valido na calibracao"); continue
        _, hl, q, thr, trrec, trfa = best
        ew_te = ew_by_hl[hl]
        ew_te = ew_te[ew_te.index >= split]
        terec, tefa = metrics_at_abs(ew_te, te_inc, thr)
        print(f"{s:11s} {hl:4.1f} {q:5.3f} | {trrec*100:5.1f}% {trfa:6.3f} {len(tr_inc):4d} | "
              f"{terec*100:5.1f}% {tefa:6.3f} {len(te_inc):4d}")
        rows.append(dict(sensor=s, hl=hl, threshold_q=q, threshold_abs=thr,
                         tr_recall=trrec, tr_fa=trfa, n_tr=len(tr_inc),
                         te_recall=terec, te_fa=tefa, n_te=len(te_inc)))

    df = pd.DataFrame(rows)
    out = f"eval_predictive_out/validate_oppoint_v9_split{split.date()}.csv"
    df.to_csv(out, index=False)
    print(f"\ncsv: {out}")


if __name__ == "__main__":
    main()
