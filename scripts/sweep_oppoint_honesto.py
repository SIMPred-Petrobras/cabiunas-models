#!/usr/bin/env python3
"""
sweep_oppoint_honesto.py
Varre half_life x sticky_hours pra um ou mais sensores, usando a métrica HONESTA atual
(best_point_for_sensor com max_sticky_duty + recall_raw) — não a busca antiga permissiva
de scripts/sweep_halflife.py, que escolhe o piso q=0.5 sem teto de duty pós-sticky.

Motivo de existir: o ponto de operação em produção (half_life=0.5h p/ T5_AVG_A, sticky=12h)
foi calibrado ANTES do max_sticky_duty existir (commit 9f00d1a) — pode estar subótimo pra
essa restrição mais rigorosa. Sem retreinar nada, só reavalia os mesmos modelos já
treinados variando o pós-processamento.

Uso:
    PYTHONPATH=. python scripts/sweep_oppoint_honesto.py \
        --task_id 58bc393c1d7a4e42815236e8897abc88 \
        --sensors T5_AVG_A TC382_03_A \
        --max_sticky_duty 0.25 --horizon 8 --fa_budget 1.0
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from clearml import Task

from scripts.eval_per_sensor_level import (
    ALARM_CSV_DEFAULT,
    load_mae_series,
    ewma_quantile,
    load_alarms_gap,
    cluster_incidents,
    best_point_for_sensor,
)

HALF_LIVES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]
STICKY_HOURS = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
MIN_DURATION_GRID = [0.0, 1.0, 2.0, 4.0, 8.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--sensors", nargs="+", required=True)
    parser.add_argument("--alarm_csv", default=ALARM_CSV_DEFAULT)
    parser.add_argument("--half_lives", type=float, nargs="+", default=HALF_LIVES)
    parser.add_argument("--sticky_hours_grid", type=float, nargs="+", default=STICKY_HOURS)
    parser.add_argument("--min_duration_grid", type=float, nargs="+", default=MIN_DURATION_GRID)
    parser.add_argument("--horizon", type=float, default=8.0)
    parser.add_argument("--fa_budget", type=float, default=1.0)
    parser.add_argument("--max_sticky_duty", type=float, default=0.25)
    parser.add_argument("--max_duty_cycle", type=float, default=1.0)
    parser.add_argument("--exclude_conditions", nargs="*", default=[])
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    print(f"[1/3] Carregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"      {task.name}  |  status={task.get_status()}")

    print(f"[2/3] Baixando sequence_scores ({', '.join(args.sensors)})...")
    mae_dict = load_mae_series(task, args.sensors)

    print(f"[3/3] Carregando alarmes ({args.alarm_csv})...")
    raw_alarms = load_alarms_gap(args.alarm_csv, args.exclude_conditions)

    for sensor in args.sensors:
        if sensor not in mae_dict:
            print(f"\n[SKIP] {sensor}: sem sequence_scores nesta task.")
            continue
        mae = mae_dict[sensor]
        incidents = cluster_incidents(raw_alarms.get(sensor, []))
        print(f"\n=== {sensor} — {len(incidents)} incidentes ===")
        if not incidents:
            print("  Sem incidentes — pulando.")
            continue

        rows = []
        for hl in args.half_lives:
            health = ewma_quantile(mae, hl)
            for sticky in args.sticky_hours_grid:
                r = best_point_for_sensor(
                    health, incidents, args.horizon,
                    sticky_hours=sticky, fa_budget=args.fa_budget,
                    min_duration_grid=args.min_duration_grid,
                    max_duty_cycle=args.max_duty_cycle,
                    max_sticky_duty=args.max_sticky_duty,
                )
                rows.append({
                    "half_life_h": hl, "sticky_h": sticky,
                    "recall": r.get("recall", 0.0),
                    "recall_raw": r.get("recall_raw", float("nan")),
                    "fa_per_day": r.get("fa_per_day", float("nan")),
                    "threshold_q": r.get("threshold_q", float("nan")),
                    "duty_cycle": r.get("duty_cycle", float("nan")),
                    "duty_sticky": r.get("duty_sticky", float("nan")),
                    "min_duration_h": r.get("min_duration_hours", 0.0),
                })

        df = pd.DataFrame(rows).sort_values(
            ["recall", "fa_per_day"], ascending=[False, True]
        ).reset_index(drop=True)

        print(f"  Top-{args.top_n} combinações (max recall, desempate por menor FA):")
        print(f"  {'hl(h)':>6} {'sticky(h)':>10} {'mindur(h)':>10} {'recall':>8} "
              f"{'rec_raw':>8} {'FA/dia':>8} {'thr_q':>6} {'duty':>6} {'duty_st':>8}")
        for _, row in df.head(args.top_n).iterrows():
            print(f"  {row['half_life_h']:>6.2f} {row['sticky_h']:>10.1f} "
                  f"{row['min_duration_h']:>10.1f} {row['recall']:>7.1%} "
                  f"{row['recall_raw']:>7.1%} {row['fa_per_day']:>8.3f} "
                  f"{row['threshold_q']:>6.3f} {row['duty_cycle']:>6.2f} "
                  f"{row['duty_sticky']:>8.2f}")

        out_csv = f"eval_predictive_out/oppoint_sweep_{sensor}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Salvo: {out_csv}")


if __name__ == "__main__":
    main()
