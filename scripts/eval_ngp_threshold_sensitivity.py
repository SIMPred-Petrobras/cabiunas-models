"""Testa a sensibilidade do resultado ao corte NGP_A usado como filtro ON/OFF —
recomputa a máscara operacional a partir do NGP_A bruto (em vez de usar o
`operational_state` já embutido na task, calibrado em NGP_A>50) e reavalia,
sem retreinar nada.

Task usada: `3643b0b37cf440b2bd498d48df1f83d3` (v10_test_ngp_a_2025 — a única
variante da v10 treinada com dataset que tem NGP_A bruto disponível; a v10
principal, 2e92c618, usa RUNNING_A e não tem NGP_A na sua fonte de dado).

Uso:
    PYTHONPATH=. python scripts/eval_ngp_threshold_sensitivity.py --thresholds 50 60
"""
from __future__ import annotations

import argparse

import pandas as pd
from clearml import Task

from scripts.eval_per_sensor_level import (
    ALARM_CSV_DEFAULT, load_mae_series, ewma_quantile,
    load_alarms_gap, cluster_incidents, best_point_for_sensor,
)

TASK_ID = "3643b0b37cf440b2bd498d48df1f83d3"
DATASET_ID = "2bacfaf24011461d9d6217df704a7fea"
SENSORS = ["T5_AVG_A", "TC382_03_A"]
HALF_LIFE_OVERRIDES = {"T5_AVG_A": 0.5}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--thresholds", type=float, nargs="+", default=[50.0, 60.0])
    p.add_argument("--half_life", type=float, default=4.0)
    p.add_argument("--sticky_hours", type=float, default=12.0)
    p.add_argument("--horizon", type=float, default=8.0)
    p.add_argument("--max_duty_cycle", type=float, default=0.35)
    p.add_argument("--exclude_conditions", nargs="*", default=["UNDER", "OVER", "LOLO", "CFN"])
    args = p.parse_args()

    print(f"[1/4] Carregando task {TASK_ID[:8]}...")
    task = Task.get_task(task_id=TASK_ID)
    mae_dict = load_mae_series(task, SENSORS)

    print("[2/4] Carregando NGP_A bruto...")
    from clearml import Dataset
    local_dir = Dataset.get(dataset_id=DATASET_ID).get_local_copy()
    ngp = pd.read_csv(f"{local_dir}/sensores_filtrados_Interpolados_2025.csv",
                       usecols=["data_datetime", "NGP_A"])
    ngp["data_datetime"] = pd.to_datetime(ngp["data_datetime"], utc=True, errors="coerce")
    ngp = ngp.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()["NGP_A"]
    ngp = pd.to_numeric(ngp, errors="coerce")

    print("[3/4] Carregando incidentes (HI/HIHI-only)...")
    raw_alarms = load_alarms_gap(ALARM_CSV_DEFAULT, args.exclude_conditions)

    print(f"[4/4] Avaliando com NGP_A > {args.thresholds}...\n")
    for thr in args.thresholds:
        on_mask = (ngp > thr)
        print(f"=== NGP_A > {thr:.0f}% (ON = {on_mask.mean():.1%} do tempo) ===")
        for sensor in SENSORS:
            mae = mae_dict.get(sensor)
            if mae is None:
                continue
            hl = HALF_LIFE_OVERRIDES.get(sensor, args.half_life)
            health = ewma_quantile(mae, hl)

            mask_h = on_mask.reindex(health.index, method="nearest",
                                      tolerance=pd.Timedelta("6min")).fillna(False)
            health_masked = health.where(mask_h, other=0.0)

            alarms_s = raw_alarms.get(sensor, [])
            on_at = on_mask.reindex(pd.DatetimeIndex(alarms_s), method="nearest",
                                     tolerance=pd.Timedelta("30min")).fillna(True) if alarms_s else pd.Series(dtype=bool)
            kept = [a for a, ok in zip(alarms_s, on_at.tolist()) if ok]
            n_off = len(alarms_s) - len(kept)
            incidents = cluster_incidents(kept)

            result = best_point_for_sensor(
                health_masked, incidents, args.horizon,
                sticky_hours=args.sticky_hours, max_duty_cycle=args.max_duty_cycle,
            )
            print(f"  {sensor}: {len(incidents)} inc (excluiu {n_off} em OFF) | "
                  f"recall={result['recall']:.1%} FA/dia={result['fa_per_day']:.3f} "
                  f"duty={result.get('duty_cycle', float('nan')):.2f}")
        print()


if __name__ == "__main__":
    main()
