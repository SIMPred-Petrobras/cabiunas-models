"""Fechamento dos 17 sensores na métrica gap-based, COM e SEM exclusão de OFF.

OFF = equipamento desligado (NGP <= RUN_THR). Alarmes em OFF não são anomalia de
processo (fora do escopo do modelo, máscara operacional em produção) e os "hits"
nesses períodos são artefato do estado, não skill. A coluna SEM-OFF é o número
honesto de frota.

IMPORTANTE: usar o registro de alarme COMPLETO (alarmes_selecionados_turbina_a.csv,
todos os 17 sensores), NÃO o _tags_modelo (filtrado a 3). half-life é escolhida por
sensor (grade), sob orçamento de FA. Recall/FA são in-sample (não validados
temporalmente — usar validate_oppoint_temporal.py para isso).

Uso:
  PYTHONPATH=. python scripts/close_fleet_off_excl.py \
    --alarm_csv ../dados/alarmes_selecionados_turbina_a.csv
"""
import argparse
import numpy as np
import pandas as pd
from clearml import Task

import scripts.eval_per_sensor_level as E

DS = "/home/thallys/.clearml/cache/storage_manager/datasets/ds_424e5b589e13402d9d95371a317e85c9"
ALARM_COMPLETO = "../dados/alarmes_selecionados_turbina_a.csv"
RAWCSV = f"{DS}/sensores_filtrados_Interpolados_2025.csv"
RUN_THR = 50
HORIZON, STICKY, FA_BUDGET = 8.0, 12.0, 1.0
HL_GRID = [0.5, 1.0, 2.0, 4.0]


def best_over_hl(mae: pd.Series, inc, ngp: pd.Series, max_duty_cycle: float = 0.35) -> dict:
    """Escolhe a half-life (grade) que maximiza recall sob FA budget, OFF excluído.
    `max_duty_cycle` (default 0.35) garante ponto deployável (não o piso q=0.5)."""
    if not inc:
        return dict(recall=float("nan"), fa_per_day=float("nan"),
                    n_incidents=0, n_hit=0, hl=float("nan"), threshold_q=float("nan"))
    best = None
    for hl in HL_GRID:
        # EWMA + rank na população ON (não na série inteira): com OFF incluído, o MAE
        # alto do estado desligado rouba os ranks altos e o duty fica irreal. Ranqueando
        # só em ON, o duty = (health>=q).mean() = 1-q, consistente com produção.
        hl_pts = max(1, int(round(pd.Timedelta(hours=hl) / pd.Timedelta(E.SAMPLING_INTERVAL))))
        ew = mae.ewm(halflife=hl_pts).mean()
        on_h = ngp.reindex(ew.index, method="nearest") > RUN_THR
        h = ew.where(on_h).dropna().rank(pct=True)
        r = E.best_point_for_sensor(h, inc, horizon_hours=HORIZON,
                                    sticky_hours=STICKY, fa_budget=FA_BUDGET, n_thresholds=120,
                                    max_duty_cycle=max_duty_cycle)
        r["hl"] = hl
        if best is None or (r["recall"] > best["recall"]) or \
           (r["recall"] == best["recall"] and r["fa_per_day"] < best["fa_per_day"]):
            best = r
    return best


def ngp_series() -> pd.Series:
    raw = pd.read_csv(RAWCSV, usecols=["data_datetime", "NGP_A"])
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    return raw["NGP_A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="58bc393c1d7a4e42815236e8897abc88")
    ap.add_argument("--alarm_csv", default=ALARM_COMPLETO,
                    help="registro COMPLETO de alarmes (todos os 17 sensores)")
    args = ap.parse_args()

    ngp = ngp_series()
    task = Task.get_task(task_id=args.task_id)
    mae_all = E.load_mae_series(task, E.SENSORS)
    alarms_gap = E.load_alarms_gap(args.alarm_csv)

    rows = []
    for sensor in E.SENSORS:
        mae = mae_all.get(sensor)
        if mae is None or mae.empty:
            continue
        # janela do MAE (=2025): alarmes fora dela não têm sinal e deflacionariam o recall
        t_lo, t_hi = mae.index.min(), mae.index.max()
        raw_al = [t for t in alarms_gap.get(sensor, []) if t_lo <= t <= t_hi]
        inc_raw = E.cluster_incidents(raw_al, gap_hours=E.GAP_HOURS)
        # ON por incidente (NGP no onset)
        on_inc = ngp.reindex(pd.DatetimeIndex(inc_raw), method="nearest") > RUN_THR if inc_raw else pd.Series([], dtype=bool)
        inc_on = [t for t, o in zip(inc_raw, on_inc.values) if o]

        excl = best_over_hl(mae, inc_on, ngp)  # SEM OFF, half-life por sensor (in-sample)

        rows.append({
            "sensor": sensor,
            "inc_all": len(inc_raw), "inc_on": len(inc_on), "inc_off": len(inc_raw) - len(inc_on),
            "recall": excl.get("recall"), "fa_per_day": excl.get("fa_per_day"),
            "hl": excl.get("hl"), "threshold_q": excl.get("threshold_q"),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    out = df.copy()
    out["recall"] = df["recall"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "  -")
    out["fa_per_day"] = df["fa_per_day"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "  -")
    out["hl"] = df["hl"].map(lambda x: f"{x:.1f}" if pd.notna(x) else " -")
    out["threshold_q"] = df["threshold_q"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "  -")
    print(f"Alarme: {args.alarm_csv}  (SEM OFF, half-life por sensor, in-sample)\n")
    print(out.to_string(index=False))

    m = df[df.inc_on > 0]
    print(f"\nsensores com incidente ON: {len(m)}/17   "
          f"incidentes OFF removidos: {int(df.inc_off.sum())} de {int(df.inc_all.sum())}")
    print(f"recall macro (SEM off): {m.recall.mean()*100:.1f}%")
    df.to_csv("eval_predictive_out/fleet_off_excl_completo.csv", index=False)
    print("csv: eval_predictive_out/fleet_off_excl_completo.csv")


if __name__ == "__main__":
    main()
