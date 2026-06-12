#!/usr/bin/env python3
"""Injeta o ponto de operação calibrado (threshold_q + debounce) no bundle de
inferência, convertido para um threshold ABSOLUTO de EWMA-MAE (streaming-safe).

Motivação: o `threshold_q` do eval é um quantil de RANK (`ewm(mae).rank(pct)`),
que não existe em streaming. Aqui convertemos:
    ewma_abs_threshold = quantil( ewm(mae, half_life).mean(), threshold_q )
sobre a janela de calibração. Em produção: `ewm(mae_novo, half_life) >= ewma_abs_threshold`
(causal, online). O bloco `production_alerting` resultante fecha o ciclo: o bundle
passa a carregar exatamente o ponto de operação que minimiza FP sem perder recall.

Uso:
  PYTHONPATH=. python scripts/finalize_bundle.py \
    --task_id <regen_task> --eval_csv eval_predictive_out/min_fp/per_sensor_eval_depois*.csv \
    --half_life 4.0 --half_life_overrides TC382_04_A=0.5 --sticky_hours 12 \
    --eval_start 2025-01-01 --eval_end 2025-11-01 \
    --out_dir production_bundles [--upload]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd
from clearml import Task

import scripts.eval_per_sensor_level as E


def smoothed_mae(mae: pd.Series, half_life_hours: float) -> pd.Series:
    hl_pts = int(round(pd.Timedelta(hours=half_life_hours) / pd.Timedelta(E.SAMPLING_INTERVAL)))
    return mae.ewm(halflife=max(1, hl_pts)).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", required=True, help="Task ClearML com modelos+bundles regenerados")
    p.add_argument("--eval_csv", required=True, help="CSV per_sensor_eval com threshold_q e min_duration_hours")
    p.add_argument("--half_life", type=float, default=4.0)
    p.add_argument("--half_life_overrides", nargs="*", default=[])
    p.add_argument("--sticky_hours", type=float, default=12.0)
    p.add_argument("--eval_start", default=None)
    p.add_argument("--eval_end", default=None)
    p.add_argument("--out_dir", default="production_bundles")
    p.add_argument("--upload", action="store_true", help="Re-upload do bundle finalizado como artefato")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hl_over = {}
    for item in args.half_life_overrides:
        if "=" in item:
            k, v = item.split("=", 1)
            hl_over[k.strip()] = float(v)

    t0 = pd.Timestamp(args.eval_start, tz="UTC") if args.eval_start else None
    t1 = pd.Timestamp(args.eval_end, tz="UTC") if args.eval_end else None

    csv_path = sorted(glob.glob(args.eval_csv))[-1] if any(c in args.eval_csv for c in "*?[") else args.eval_csv
    ops = pd.read_csv(csv_path).set_index("sensor")
    print(f"[1/3] Pontos de operação de {csv_path} ({len(ops)} sensores)")

    task = Task.get_task(task_id=args.task_id)
    print(f"[2/3] Task {task.name} | status={task.get_status()}")
    mae_dict = E.load_mae_series(task, list(ops.index))

    print(f"[3/3] Convertendo threshold_q → ewma_abs_threshold e finalizando bundles...")
    arts = task.artifacts
    n_done = 0
    for sensor, row in ops.iterrows():
        thr_q = float(row["threshold_q"])
        debounce = float(row.get("min_duration_hours", 0.0) or 0.0)
        hl = hl_over.get(sensor, args.half_life)

        mae = mae_dict.get(sensor)
        if mae is None:
            print(f"  [skip] {sensor}: sem mae_seq")
            continue
        if t0 is not None:
            mae = mae[mae.index >= t0]
        if t1 is not None:
            mae = mae[mae.index <= t1]
        abs_thr = float(smoothed_mae(mae, hl).quantile(thr_q))

        bkey = next((k for k in arts if "inference_bundle" in k and sensor in k), None)
        if bkey is None:
            print(f"  [skip] {sensor}: sem inference_bundle no task (modelo regenerado?)")
            continue
        bundle = json.load(open(arts[bkey].get_local_copy()))
        bundle["production_alerting"] = {
            "half_life_hours": hl,
            "threshold_q": thr_q,
            "ewma_abs_threshold": abs_thr,
            "sticky_hours": float(args.sticky_hours),
            "debounce_hours": debounce,
            "recall_at_op": float(row.get("recall", float("nan"))),
            "fa_per_day_at_op": float(row.get("fa_per_day", float("nan"))),
        }
        out_path = os.path.join(args.out_dir, f"{sensor}_inference_bundle.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False)
        print(f"  {sensor}: hl={hl}h thr_q={thr_q:.3f} → ewma_abs={abs_thr:.5f} debounce={debounce:.1f}h")
        if args.upload:
            task.upload_artifact(name=f"{sensor}_inference_bundle_final_json", artifact_object=out_path)
        n_done += 1

    print(f"\nFinalizados {n_done} bundles em {args.out_dir}/")


if __name__ == "__main__":
    main()
