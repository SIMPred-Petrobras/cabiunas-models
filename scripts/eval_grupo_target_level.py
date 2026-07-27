#!/usr/bin/env python3
"""
eval_grupo_target_level.py
Reavalia uma task de grupo multivariado (target_sensor) com a mesma métrica honesta
de eval_per_sensor_level.py (sticky duty ceiling + recall_raw), sem retreinar.

Motivo de existir: tasks de grupo (pipeline.run_one_group, ex. MULTI_T5_AVG_TC382__*)
salvam artefatos com o nome do GRUPO, não do sensor (ex.
"MULTI_T5_AVG_TC382__csv_sequence_scores_all.csv" em vez de
"T5_AVG_A_csv_sequence_scores_all.csv") e não gravam a coluna operational_state em
point_anomalies_all.csv — então eval_per_sensor_level.py não consegue carregar essas
tasks diretamente (load_mae_series faz match por substring "sensor in key").

Uso:
    PYTHONPATH=. python scripts/eval_grupo_target_level.py \
        --task_id 617ee77695d548c58175d8d7cb574c6e --target_sensor T5_AVG_A \
        --label grupo_cnn_t5 --half_life 0.5 --sticky_hours 12 \
        --max_duty_cycle 0.35 --horizon 8
"""
from __future__ import annotations

import argparse

import pandas as pd
from clearml import Task

from scripts.eval_per_sensor_level import (
    ALARM_CSV_DEFAULT,
    ewma_quantile,
    load_alarms_gap,
    load_alarms_ok_aware,
    cluster_incidents,
    best_point_for_sensor,
)


def load_group_mae(task: Task, target_sensor: str) -> pd.Series:
    arts = task.artifacts
    key = next((k for k in arts if "sequence_scores_all" in k), None)
    if key is None:
        raise RuntimeError("Nenhum artifact *sequence_scores_all* encontrado na task.")
    path = arts[key].get_local_copy()
    df = pd.read_csv(path)
    df["seq_start_time"] = pd.to_datetime(df["seq_start_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["seq_start_time"]).sort_values("seq_start_time")
    col = f"mae_{target_sensor}" if f"mae_{target_sensor}" in df.columns else "mae_seq"
    print(f"      Usando coluna '{col}' de {key}")
    return df.set_index("seq_start_time")[col]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--target_sensor", required=True)
    parser.add_argument("--label", default="grupo_eval")
    parser.add_argument("--alarm_csv", default=ALARM_CSV_DEFAULT)
    parser.add_argument("--half_life", type=float, default=4.0)
    parser.add_argument("--sticky_hours", type=float, default=0.0)
    parser.add_argument("--horizon", type=float, default=8.0)
    parser.add_argument("--fa_budget", type=float, default=1.0)
    parser.add_argument("--max_duty_cycle", type=float, default=1.0)
    parser.add_argument("--max_sticky_duty", type=float, default=0.25)
    parser.add_argument("--ok_aware", action="store_true")
    parser.add_argument("--exclude_conditions", nargs="*", default=[])
    args = parser.parse_args()

    print(f"[1/3] Carregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"      {task.name}  |  status={task.get_status()}")

    print(f"[2/3] Baixando sequence_scores (canal {args.target_sensor})...")
    mae = load_group_mae(task, args.target_sensor)
    print(f"      EWMA hl={args.half_life}h + quantile normalization...")
    health = ewma_quantile(mae, args.half_life)

    print(f"[3/3] Carregando alarmes ({args.alarm_csv})...")
    if args.ok_aware:
        raw_alarms = load_alarms_ok_aware(args.alarm_csv, args.exclude_conditions)
        incidents = raw_alarms.get(args.target_sensor, [])
    else:
        raw_alarms = load_alarms_gap(args.alarm_csv, args.exclude_conditions)
        incidents = cluster_incidents(raw_alarms.get(args.target_sensor, []))

    result = best_point_for_sensor(
        health, incidents, args.horizon,
        sticky_hours=args.sticky_hours, fa_budget=args.fa_budget,
        max_duty_cycle=args.max_duty_cycle, max_sticky_duty=args.max_sticky_duty,
    )
    print(f"\n=== {args.label} ({args.target_sensor}, H={args.horizon}h) ===")
    print(f"  n_incidents:  {result.get('n_incidents')}")
    print(f"  recall:       {result.get('recall'):.1%}")
    print(f"  recall_raw:   {result.get('recall_raw', float('nan')):.1%}")
    print(f"  fa_per_day:   {result.get('fa_per_day'):.3f}")
    print(f"  threshold_q:  {result.get('threshold_q'):.3f}")
    print(f"  duty_cycle:   {result.get('duty_cycle'):.2f}")
    print(f"  duty_sticky:  {result.get('duty_sticky', float('nan')):.2f}")
    print(f"  median_lead_hours: {result.get('median_lead_hours', float('nan')):.2f}")


if __name__ == "__main__":
    main()
