"""Fechamento dos 17 sensores na métrica gap-based, COM e SEM exclusão de OFF.

OFF = equipamento desligado (NGP <= RUN_THR). Alarmes em OFF não são anomalia de
processo (fora do escopo do modelo, máscara operacional em produção) e os "hits"
nesses períodos são artefato do estado, não skill. A coluna SEM-OFF é o número
honesto de frota.

Uso:
  PYTHONPATH=. python scripts/close_fleet_off_excl.py --task_id <prod_task_id>
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
HORIZON, HL, STICKY, FA_BUDGET = 8.0, 4.0, 12.0, 1.0


def ngp_series() -> pd.Series:
    raw = pd.read_csv(RAWCSV, usecols=["data_datetime", "NGP_A"])
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    return raw["NGP_A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="58bc393c1d7a4e42815236e8897abc88")
    args = ap.parse_args()

    ngp = ngp_series()
    task = Task.get_task(task_id=args.task_id)
    mae_all = E.load_mae_series(task, E.SENSORS)
    alarms_gap = E.load_alarms_gap(ALARM)

    rows = []
    for sensor in E.SENSORS:
        mae = mae_all.get(sensor)
        if mae is None or mae.empty:
            continue
        health = E.ewma_quantile(mae, HL)
        on_h = ngp.reindex(health.index, method="nearest") > RUN_THR  # ON na cadência da health

        inc_raw = E.cluster_incidents(alarms_gap.get(sensor, []), gap_hours=E.GAP_HOURS)
        # ON por incidente (NGP no onset)
        on_inc = ngp.reindex(pd.DatetimeIndex(inc_raw), method="nearest") > RUN_THR if inc_raw else pd.Series([], dtype=bool)
        inc_on = [t for t, o in zip(inc_raw, on_inc.values)] if not len(inc_raw) else [t for t, o in zip(inc_raw, on_inc.values) if o]

        def pt(health_s, inc):
            if not inc:
                return dict(recall=float("nan"), fa_per_day=float("nan"), n_incidents=0, n_hit=0)
            return E.best_point_for_sensor(health_s, inc, horizon_hours=HORIZON,
                                           sticky_hours=STICKY, fa_budget=FA_BUDGET, n_thresholds=120)

        full = pt(health, inc_raw)                       # COM OFF (denominador inflado)
        # SEM OFF: health só em ON (sem hits/FP de estado), incidentes só ON
        excl = pt(health.where(on_h).dropna(), inc_on)

        rows.append({
            "sensor": sensor,
            "inc_all": len(inc_raw), "inc_on": len(inc_on), "inc_off": len(inc_raw) - len(inc_on),
            "recall_COM_off": full.get("recall"), "fa_COM_off": full.get("fa_per_day"),
            "recall_SEM_off": excl.get("recall"), "fa_SEM_off": excl.get("fa_per_day"),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    def fmt(x): return f"{x*100:.1f}%" if pd.notna(x) else "  -"
    out = df.copy()
    for c in ["recall_COM_off", "recall_SEM_off"]:
        out[c] = df[c].map(fmt)
    for c in ["fa_COM_off", "fa_SEM_off"]:
        out[c] = df[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "  -")
    print(out.to_string(index=False))

    # média macro de recall (só sensores com incidentes ON)
    m = df[df.inc_on > 0]
    print(f"\nincidentes OFF removidos no total: {int(df.inc_off.sum())} de {int(df.inc_all.sum())}")
    print(f"recall macro COM off: {m.recall_COM_off.mean()*100:.1f}%   SEM off: {m.recall_SEM_off.mean()*100:.1f}%")
    df.to_csv("eval_predictive_out/fleet_off_excl.csv", index=False)
    print("csv: eval_predictive_out/fleet_off_excl.csv")


if __name__ == "__main__":
    main()
