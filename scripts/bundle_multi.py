#!/usr/bin/env python3
"""Cria inference_bundle.json para o Dense AE multivariado OOS.

pipeline_multi.py não salva center/scale. Este script reproduz o mesmo
preprocessing para obter o scaler e monta o bundle pronto para inference.py.

Uso:
    PYTHONPATH=. python scripts/bundle_multi.py \
        --task_id 8de61587255943d88b42a868d7acdc25 \
        --out_dir production_bundles/dense_multi_2026
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from clearml import Task

from src.cnn1d_ae.config import PipelineConfig
from src.cnn1d_ae.io import load_data
from src.cnn1d_ae.preprocess import (
    apply_hampel_filter,
    build_exclusion_mask,
    build_sensor_dataframe,
    build_startup_exclusion_mask,
)


def _load_running_col(cfg, df_raw, index):
    if not cfg.RUNNING_COL:
        return None
    src = df_raw if cfg.TRAIN_SOURCE.lower() == "raw" else None
    if src is None or cfg.RUNNING_COL not in src.columns:
        return None
    running = (
        src.drop_duplicates(subset=[cfg.TIME_COL])
        .set_index(cfg.TIME_COL)[cfg.RUNNING_COL]
    )
    return pd.to_numeric(running, errors="coerce").reindex(index).fillna(0.0)


def compute_scaler(cfg: PipelineConfig, df_alarm: pd.DataFrame,
                   df_feat: pd.DataFrame, df_raw: pd.DataFrame,
                   sensors: list[str]) -> tuple[pd.Series, pd.Series]:
    """Reproduz exatamente os passos 1-4 de pipeline_multi.py para obter center/scale."""

    # Passo 1: preprocessar cada sensor
    sensor_dfs = {}
    for sensor in sensors:
        try:
            df_s, _ = build_sensor_dataframe(cfg, df_feat, df_raw, sensor)
            df_s = apply_hampel_filter(df_s, sensor, cfg)
            sensor_dfs[sensor] = df_s[[sensor]]
        except Exception as exc:
            print(f"  [WARN] {sensor}: {exc} — ignorado")

    sensors_ok = list(sensor_dfs.keys())
    print(f"  Sensores preprocessados: {len(sensors_ok)}/{len(sensors)}")

    # Passo 2: inner-join
    common_index = None
    for df_s in sensor_dfs.values():
        common_index = df_s.index if common_index is None else common_index.intersection(df_s.index)
    common_index = common_index.sort_values()

    df_all = pd.DataFrame(index=common_index)
    for sensor, df_s in sensor_dfs.items():
        df_all[sensor] = df_s.loc[common_index, sensor]
    if df_all.isna().any().any():
        df_all = df_all.ffill().bfill().fillna(0.0)

    print(f"  Index comum: {len(df_all):,} pts ({df_all.index.min()} → {df_all.index.max()})")

    # Passo 3: exclusion mask
    all_alarm_times = pd.to_datetime(
        df_alarm["Data da Ocorrencia"], errors="coerce"
    ).dropna().drop_duplicates()

    alarm_exclude = build_exclusion_mask(common_index, all_alarm_times, cfg.EXCLUDE_MINUTES_AROUND_ALARM)
    exclude = np.array(alarm_exclude, dtype=bool)

    running = _load_running_col(cfg, df_raw, common_index)
    if running is not None:
        n_off = int((running <= cfg.RUNNING_THRESHOLD).sum())
        print(f"  OFF excluídos: {n_off:,} (RUNNING_COL={cfg.RUNNING_COL} ≤ {cfg.RUNNING_THRESHOLD})")
        exclude = exclude | (running <= cfg.RUNNING_THRESHOLD).values

    temp_sensors = [s for s in sensors_ok if s.startswith("TC") or s.startswith("T5")]
    ref_sensor = temp_sensors[0] if temp_sensors else sensors_ok[0]
    if cfg.EXCLUDE_STARTUP_MINUTES > 0:
        off_thr = float(df_all[ref_sensor].quantile(cfg.OFF_VALUE_QUANTILE))
        startup_excl = build_startup_exclusion_mask(
            common_index, df_all[ref_sensor], off_thr, cfg.EXCLUDE_STARTUP_MINUTES
        )
        exclude = exclude | np.array(startup_excl, dtype=bool)

    # Passo 4: df_normal → df_fit → center/scale
    df_normal = df_all.loc[~exclude].copy()
    print(f"  Dados normais: {len(df_normal):,} pts ({100*len(df_normal)/max(len(df_all),1):.1f}%)")

    n_norm = len(df_normal)
    n_val_rows = int(np.floor(cfg.VAL_FRAC * n_norm))
    df_fit = df_normal.iloc[: n_norm - n_val_rows] if n_val_rows > 0 else df_normal
    print(f"  df_fit (treino): {len(df_fit):,} pts (VAL_FRAC={cfg.VAL_FRAC})")

    center = df_fit.mean(axis=0)
    scale = df_fit.std(axis=0).replace(0, 1.0)
    return center, scale, sensors_ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", default="8de61587255943d88b42a868d7acdc25")
    p.add_argument("--out_dir", default="production_bundles/dense_multi_2026")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Carrega task e run_config
    print(f"\n[1/5] Carregando task {args.task_id[:8]}...")
    task = Task.get_task(task_id=args.task_id)
    print(f"  Nome: {task.name} | Status: {task.status}")

    arts = task.artifacts
    run_config_key = next(k for k in arts if "run_config" in k)
    run_config_path = arts[run_config_key].get_local_copy()
    with open(run_config_path) as f:
        run_config = json.load(f)

    report_key = next(k for k in arts if "multivariado_report" in k)
    report_path = arts[report_key].get_local_copy()
    with open(report_path) as f:
        report = json.load(f)

    cfg = PipelineConfig(**{k: v for k, v in run_config.items() if hasattr(PipelineConfig, k)})
    sensors = report["sensors"]
    threshold = report["threshold"]
    op_point = report["predictive_operating_points"]["H72h"]
    ewma_abs_threshold = op_point["threshold"]

    print(f"  Sensores: {sensors}")
    print(f"  threshold (raw MAE): {threshold:.6f}")
    print(f"  ewma_abs_threshold (H72h): {ewma_abs_threshold:.6f}")

    # 2. Carrega dados (load_data baixa o dataset via CLEARML_DATASET_ID automaticamente)
    print(f"\n[2/4] Carregando dados do dataset {cfg.CLEARML_DATASET_ID[:8]}...")
    df_alarm, df_feat, df_raw, _ = load_data(cfg)
    print(f"  df_raw: {df_raw.shape} | df_alarm: {df_alarm.shape}")

    # 3. Computa center/scale reproduzindo pipeline_multi.py
    print("\n[3/4] Computando scaler (center/scale)...")
    center, scale, sensors_ok = compute_scaler(cfg, df_alarm, df_feat, df_raw, sensors)

    print("\n  center:")
    for s in sensors_ok:
        print(f"    {s}: {center[s]:.6f}")
    print("  scale:")
    for s in sensors_ok:
        print(f"    {s}: {scale[s]:.6f}")

    # 4. Baixa model.keras
    print("\n[4/4] Baixando model.keras...")
    model_key = next(k for k in arts if "model_keras" in k)
    model_src = arts[model_key].get_local_copy()
    model_dst = os.path.join(args.out_dir, "model.keras")
    shutil.copy2(model_src, model_dst)
    print(f"  Salvo em: {model_dst}")

    # 6. Monta e salva o bundle
    bundle = {
        "sensors": sensors_ok,
        "feature_columns": sensors_ok,
        "model_file": "model.keras",
        "model_arch": "dense",
        "task_id": args.task_id,
        "time_steps": int(cfg.TIME_STEPS),
        "stride": int(cfg.STRIDE),
        "normalize_mode": cfg.NORMALIZE_MODE,
        "center": {str(k): float(v) for k, v in center.to_dict().items()},
        "scale":  {str(k): float(v) for k, v in scale.to_dict().items()},
        "clip_bounds": {},
        "threshold": float(threshold),
        "thresh_mode": cfg.THRESH_MODE,
        "running_col": cfg.RUNNING_COL,
        "running_threshold": float(cfg.RUNNING_THRESHOLD),
        "point_rule": cfg.POINT_RULE,
        "point_window": int(cfg.POINT_WINDOW),
        "point_min_count": int(cfg.POINT_MIN_COUNT),
        "production_alerting": {
            "half_life_hours": float(cfg.PREDICTIVE_EWMA_HALF_LIFE_HOURS),
            "ewma_abs_threshold": float(ewma_abs_threshold),
            "debounce_hours": float(cfg.PREDICTIVE_ALERT_DEBOUNCE_HOURS),
        },
    }

    bundle_path = os.path.join(args.out_dir, "inference_bundle.json")
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    print(f"\n  Bundle salvo em: {bundle_path}")

    # Verificação rápida
    assert len(bundle["center"]) == len(sensors_ok), "center incompleto"
    assert all(v > 0 for v in bundle["scale"].values()), "scale tem zero(s)"
    assert bundle["threshold"] > 0, "threshold inválido"
    assert bundle["production_alerting"]["ewma_abs_threshold"] > 0, "ewma_abs_threshold inválido"
    print("  Verificação OK: center/scale/threshold válidos")
    print("\nBundle pronto. Use com inference.load_bundle() + score_production().")


if __name__ == "__main__":
    main()
