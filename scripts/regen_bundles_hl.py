"""Regenera o bloco `production_alerting` dos bundles cujo half-life mudou,
recalculando o threshold ABSOLUTO de EWMA-MAE com OFF excluído (NGP<=RUN_THR).

Mudança validada (validate_oppoint_temporal): T5_AVG_A e TC382_04_A passam a
half-life 0.5h (eventos UNDER breves) — T5 sobe 81.8%→100% out-of-sample. Só esses
bundles são tocados; os demais (hl=4h) ficam inalterados. threshold_q=0.5 (saturou
no piso na calibração = sensibilidade máxima dentro do orçamento de FA).

Uso:
  PYTHONPATH=. python scripts/regen_bundles_hl.py --task_id 58bc393c... \
    --out_dir production_bundles --eval_start 2025-01-01 --eval_end 2025-11-01
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from clearml import Task

import scripts.eval_per_sensor_level as E
from scripts.finalize_bundle import smoothed_mae

DS = "/home/thallys/.clearml/cache/storage_manager/datasets/ds_424e5b589e13402d9d95371a317e85c9"
RAWCSV = f"{DS}/sensores_filtrados_Interpolados_2025.csv"
RUN_THR = 50
# sensor -> (half_life_hours, threshold_q)  (ponto de operação DEPLOYÁVEL:
# q escolhido pelo trade-off recall × duty-cycle OOS 2024, scripts/analyze_duty_cycle.py.
# q=0.9 na maioria (recall ~90-100%, duty 13-21%); TC382_03 q=0.92 (meio-termo
# 76% recall / 33% duty — sensor genuinamente anômalo grande parte do tempo).
RECIPE = {
    "T5_AVG_A": (0.5, 0.90),
    "TC382_01_A": (0.5, 0.90),
    "TC382_02_A": (2.0, 0.90),
    "TC382_03_A": (4.0, 0.92),
    "TC382_04_A": (0.5, 0.90),
    "TC382_05_A": (1.0, 0.90),
    "TC382_06_A": (0.5, 0.90),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default="58bc393c1d7a4e42815236e8897abc88")
    ap.add_argument("--out_dir", default="production_bundles")
    ap.add_argument("--eval_start", default="2025-01-01")
    ap.add_argument("--eval_end", default="2025-11-01")
    ap.add_argument("--sticky_hours", type=float, default=12.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ngp = pd.read_csv(RAWCSV, usecols=["data_datetime", "NGP_A"])
    ngp["data_datetime"] = pd.to_datetime(ngp["data_datetime"], utc=True, errors="coerce")
    ngp = ngp.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()["NGP_A"]

    t0 = pd.Timestamp(args.eval_start, tz="UTC")
    t1 = pd.Timestamp(args.eval_end, tz="UTC")

    task = Task.get_task(task_id=args.task_id)
    arts = task.artifacts
    mae_dict = E.load_mae_series(task, list(RECIPE))

    for sensor, (hl, thr_q) in RECIPE.items():
        mae = mae_dict.get(sensor)
        if mae is None:
            print(f"  [skip] {sensor}: sem mae_seq"); continue
        mae = mae[(mae.index >= t0) & (mae.index <= t1)]
        # OFF excluido: o threshold de produção é calibrado só em operação real
        on = ngp.reindex(mae.index, method="nearest") > RUN_THR
        mae_on = mae[on.values]
        abs_thr = float(smoothed_mae(mae_on, hl).quantile(thr_q))

        bkey = next((k for k in arts if "inference_bundle" in k and sensor in k), None)
        if bkey is None:
            print(f"  [skip] {sensor}: sem inference_bundle"); continue
        bundle = json.load(open(arts[bkey].get_local_copy()))
        old = bundle.get("production_alerting", {})
        bundle["production_alerting"] = {
            "half_life_hours": hl,
            "threshold_q": thr_q,
            "ewma_abs_threshold": abs_thr,
            "sticky_hours": float(args.sticky_hours),
            "debounce_hours": float(old.get("debounce_hours", 0.0)),
            "off_excluded_in_calibration": True,
            "note": "hl revisado p/ eventos UNDER breves (validado temporalmente OOS)",
        }
        out_path = os.path.join(args.out_dir, f"{sensor}_inference_bundle.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False)
        print(f"  {sensor}: hl {old.get('half_life_hours','?')}→{hl}h  "
              f"ewma_abs {old.get('ewma_abs_threshold','?')}→{abs_thr:.5f}  (n_on={len(mae_on)})")
    print(f"\nbundles regenerados em {args.out_dir}/")


if __name__ == "__main__":
    main()
