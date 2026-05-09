from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import PipelineConfig, cfg_to_dict


def _parse_datetime_smart(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip()
    slash_ratio = s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}", na=False).mean()
    iso_ratio = s.str.match(r"^\d{4}-\d{2}-\d{2}", na=False).mean()

    if slash_ratio >= 0.8:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    if iso_ratio >= 0.8:
        dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S")
        if dt.isna().mean() > 0.1:
            dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
        return dt
    return pd.to_datetime(s, errors="coerce", dayfirst=False)


def _to_utc_indexed_series(
    dt: pd.Series,
    source_tz: str,
    target_tz: str,
    apply_hour_shift: bool,
    shift_hours: int,
) -> pd.Series:
    out = dt.copy()
    if getattr(out.dt, "tz", None) is None:
        out = out.dt.tz_localize(source_tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        out = out.dt.tz_convert(source_tz)

    out = out.dt.tz_convert(target_tz)

    if apply_hour_shift and shift_hours != 0:
        out = out + pd.Timedelta(hours=shift_hours)

    return out


def _log_time_audit(name: str, before: pd.Series, after: pd.Series, n: int) -> None:
    n = max(0, int(n))
    if n == 0:
        return
    print(f"[TIME-AUDIT] {name}: amostras before->after (n={n})")
    pairs = pd.DataFrame({"before": before.head(n).astype(str), "after": after.head(n).astype(str)})
    for _, row in pairs.iterrows():
        print(f"  {row['before']} -> {row['after']}")


def build_time_integrity_report(df: pd.DataFrame, time_col: str, name: str) -> Dict[str, object]:
    s = df[time_col]
    is_monotonic = bool(s.is_monotonic_increasing)
    n_dupes = int(s.duplicated().sum())
    coverage = {
        "start": str(s.min()) if len(s) else None,
        "end": str(s.max()) if len(s) else None,
        "rows": int(len(s)),
    }
    return {
        "dataset": name,
        "is_monotonic_increasing": is_monotonic,
        "duplicate_timestamps": n_dupes,
        "coverage": coverage,
    }


def resolve_output_dir(cfg: PipelineConfig, sensor: str) -> str:
    per_sensor = cfg.OUTPUT_DIR_TEMPLATE.format(sensor=sensor)
    if cfg.OUTPUT_ROOT:
        return os.path.join(cfg.OUTPUT_ROOT, per_sensor)
    return per_sensor


def ensure_sensor_dirs(cfg: PipelineConfig, sensor: str) -> Dict[str, str]:
    base = resolve_output_dir(cfg, sensor)
    paths = {
        "root": base,
        "tuner": os.path.join(base, "tuner"),
        "best_model": os.path.join(base, "best_model"),
        "figs": os.path.join(base, "figs"),
        "csv": os.path.join(base, "csv"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def save_run_config(cfg: PipelineConfig, out_dirs: Dict[str, str]) -> str:
    path = os.path.join(out_dirs["csv"], "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg_to_dict(cfg), f, indent=2, ensure_ascii=False)
    return path


def _resolve_dataset_file(dataset_root: Path, configured_path: str) -> Path:
    raw_path = Path(configured_path)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(dataset_root / raw_path.relative_to(raw_path.anchor))
    else:
        candidates.append(dataset_root / raw_path)

    candidates.append(dataset_root / raw_path.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = [p for p in dataset_root.rglob(raw_path.name) if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        parts = raw_path.parts[-3:]
        for match in matches:
            if match.parts[-len(parts):] == parts:
                return match
        return matches[0]

    raise FileNotFoundError(
        f"Arquivo '{configured_path}' nao encontrado no ClearML Dataset em: {dataset_root}"
    )


def _resolve_input_paths(cfg: PipelineConfig) -> Tuple[str, str, str]:
    dataset_id = (cfg.CLEARML_DATASET_ID or os.getenv("CLEARML_DATASET_ID", "")).strip()
    if not cfg.USE_CLEARML_DATASET:
        return cfg.ALARM_CSV, cfg.FEATURES_CSV, cfg.RAW_CSV

    from clearml import Dataset

    if dataset_id:
        dataset = Dataset.get(dataset_id=dataset_id)
    else:
        dataset = Dataset.get(
            dataset_name=cfg.CLEARML_DATASET_NAME,
            dataset_project=cfg.CLEARML_PROJECT_NAME,
        )
    dataset_root = Path(dataset.get_local_copy())
    print(f"[CLEARML-DATASET] Usando dataset_id={dataset.id}")
    print(f"[CLEARML-DATASET] Dataset: {cfg.CLEARML_PROJECT_NAME}/{cfg.CLEARML_DATASET_NAME}")
    print(f"[CLEARML-DATASET] Local copy: {dataset_root}")

    alarm_csv = _resolve_dataset_file(dataset_root, cfg.ALARM_CSV)
    raw_csv = _resolve_dataset_file(dataset_root, cfg.RAW_CSV)
    # FEATURES_CSV é opcional quando TRAIN_SOURCE=raw; evita exigir arquivo não utilizado.
    if cfg.TRAIN_SOURCE.lower() == "raw":
        features_csv = ""
    else:
        features_csv = _resolve_dataset_file(dataset_root, cfg.FEATURES_CSV)

    print(f"[CLEARML-DATASET] ALARM_CSV -> {alarm_csv}")
    print(f"[CLEARML-DATASET] FEATURES_CSV -> {features_csv or '(não utilizado)'}")
    print(f"[CLEARML-DATASET] RAW_CSV -> {raw_csv}")
    return str(alarm_csv), str(features_csv), str(raw_csv)


def _process_time_column(df: pd.DataFrame, col: str, cfg: PipelineConfig, name: str) -> pd.DataFrame:
    raw = df[col].copy()
    parsed = _parse_datetime_smart(raw)
    parsed = parsed.dropna()

    valid_idx = parsed.index
    df = df.loc[valid_idx].copy()

    converted = _to_utc_indexed_series(
        parsed,
        source_tz=cfg.SOURCE_TZ,
        target_tz=cfg.TARGET_TZ,
        apply_hour_shift=cfg.APPLY_HOUR_SHIFT,
        shift_hours=cfg.SHIFT_HOURS,
    )

    _log_time_audit(name, parsed.reset_index(drop=True), converted.reset_index(drop=True), cfg.LOG_TIME_AUDIT_SAMPLES)

    # Guardamos como naive em UTC para compatibilidade com CSV e comparacoes.
    df[col] = converted.dt.tz_localize(None)
    df = df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    return df


def load_data(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, object]]]:
    alarm_csv, features_csv, raw_csv = _resolve_input_paths(cfg)

    df_alarm = pd.read_csv(alarm_csv)
    if "Data da Ocorrência" in df_alarm.columns and "Data da Ocorrencia" not in df_alarm.columns:
        df_alarm["Data da Ocorrencia"] = df_alarm["Data da Ocorrência"]
    if "Tag Alarme" in df_alarm.columns and "Tag" not in df_alarm.columns:
        df_alarm["Tag"] = df_alarm["Tag Alarme"]

    df_raw = pd.read_csv(raw_csv)

    if features_csv:
        df_feat = pd.read_csv(features_csv)
        df_feat = _process_time_column(df_feat, cfg.TIME_COL, cfg, "features")
    else:
        df_feat = pd.DataFrame(columns=[cfg.TIME_COL])

    df_alarm = _process_time_column(df_alarm, "Data da Ocorrencia", cfg, "alarms")
    df_raw = _process_time_column(df_raw, cfg.TIME_COL, cfg, "raw")

    report = {
        "alarm": build_time_integrity_report(df_alarm, "Data da Ocorrencia", "alarm"),
        "feat": build_time_integrity_report(df_feat, cfg.TIME_COL, "feat") if not df_feat.empty else {},
        "raw": build_time_integrity_report(df_raw, cfg.TIME_COL, "raw"),
    }
    print("[TIME-INTEGRITY]", json.dumps(report, ensure_ascii=False))

    return df_alarm, df_feat, df_raw, report
