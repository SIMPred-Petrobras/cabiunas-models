from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def _resolve_input_paths(cfg: PipelineConfig) -> Tuple[str, str, str, Optional[str]]:
    dataset_id = (cfg.CLEARML_DATASET_ID or os.getenv("CLEARML_DATASET_ID", "")).strip()
    if not cfg.USE_CLEARML_DATASET:
        return cfg.ALARM_CSV, cfg.FEATURES_CSV, cfg.RAW_CSV, cfg.EXTRA_RAW_CSV or None

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
    features_csv = _resolve_dataset_file(dataset_root, cfg.FEATURES_CSV)
    raw_csv = _resolve_dataset_file(dataset_root, cfg.RAW_CSV)

    print(f"[CLEARML-DATASET] ALARM_CSV -> {alarm_csv}")
    print(f"[CLEARML-DATASET] FEATURES_CSV -> {features_csv}")
    print(f"[CLEARML-DATASET] RAW_CSV -> {raw_csv}")

    extra_raw_csv: Optional[str] = None
    if cfg.EXTRA_RAW_CSV:
        extra_raw_csv = str(_resolve_dataset_file(dataset_root, cfg.EXTRA_RAW_CSV))
        print(f"[CLEARML-DATASET] EXTRA_RAW_CSV -> {extra_raw_csv}")

    return str(alarm_csv), str(features_csv), str(raw_csv), extra_raw_csv


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
    alarm_csv, features_csv, raw_csv, extra_raw_csv = _resolve_input_paths(cfg)

    df_alarm = pd.read_csv(alarm_csv)
    if "Data da Ocorrência" in df_alarm.columns and "Data da Ocorrencia" not in df_alarm.columns:
        df_alarm["Data da Ocorrencia"] = df_alarm["Data da Ocorrência"]
    if "Tag Alarme" in df_alarm.columns and "Tag" not in df_alarm.columns:
        df_alarm["Tag"] = df_alarm["Tag Alarme"]
    if "Status" in df_alarm.columns:
        # Alguns arquivos de alarme (ex: alarmes_selecionados_turbina_a.csv)
        # trazem pares onset/clear (ACT/UNACK + INACT/UNACK) para o mesmo
        # evento. So o onset (ACT) e o que faz sentido prever com antecedencia
        # -- o clear nao e "detectavel antes", e so a baixa do alarme.
        df_alarm = df_alarm[df_alarm["Status"].astype(str).str.startswith("ACT")].copy()

    df_feat = pd.read_csv(features_csv)
    df_raw = pd.read_csv(raw_csv)

    df_alarm = _process_time_column(df_alarm, "Data da Ocorrencia", cfg, "alarms")
    df_feat = _process_time_column(df_feat, cfg.TIME_COL, cfg, "features")
    df_raw = _process_time_column(df_raw, cfg.TIME_COL, cfg, "raw")

    if cfg.DATA_START_DATE:
        start = pd.Timestamp(cfg.DATA_START_DATE)
        df_feat = df_feat.loc[df_feat[cfg.TIME_COL] >= start].reset_index(drop=True)
        df_raw = df_raw.loc[df_raw[cfg.TIME_COL] >= start].reset_index(drop=True)
    if cfg.DATA_END_DATE:
        end = pd.Timestamp(cfg.DATA_END_DATE)
        df_feat = df_feat.loc[df_feat[cfg.TIME_COL] <= end].reset_index(drop=True)
        df_raw = df_raw.loc[df_raw[cfg.TIME_COL] <= end].reset_index(drop=True)

    report = {
        "alarm": build_time_integrity_report(df_alarm, "Data da Ocorrencia", "alarm"),
        "feat": build_time_integrity_report(df_feat, cfg.TIME_COL, "feat"),
        "raw": build_time_integrity_report(df_raw, cfg.TIME_COL, "raw"),
    }

    # Mescla colunas do arquivo extra (ex: NGP_A do arquivo antigo).
    # Apenas colunas ausentes no df_raw principal são adicionadas —
    # o arquivo principal tem precedência em caso de conflito.
    if extra_raw_csv:
        df_extra = pd.read_csv(extra_raw_csv)
        df_extra = _process_time_column(df_extra, cfg.TIME_COL, cfg, "extra_raw")
        new_cols = [c for c in df_extra.columns if c != cfg.TIME_COL and c not in df_raw.columns]
        if new_cols:
            print(f"[EXTRA_RAW_CSV] Mesclando {len(new_cols)} coluna(s) nova(s): {new_cols[:8]}")
            df_raw = df_raw.merge(df_extra[[cfg.TIME_COL] + new_cols], on=cfg.TIME_COL, how="outer")
            df_raw = df_raw.sort_values(cfg.TIME_COL).reset_index(drop=True)
            report["extra_raw"] = build_time_integrity_report(df_extra, cfg.TIME_COL, "extra_raw")
        else:
            print("[EXTRA_RAW_CSV] Nenhuma coluna nova encontrada; arquivo ignorado.")

    print("[TIME-INTEGRITY]", json.dumps(report, ensure_ascii=False))

    return df_alarm, df_feat, df_raw, report


def load_alert_context_catalog(cfg: PipelineConfig) -> pd.DataFrame:
    """Carrega o catalogo AMPLO de alarmes usado so pra anotacao de
    contexto (ENABLE_ALERT_CATALOG_CONTEXT) -- arquivo independente do
    ALARM_CSV de avaliacao (ex: alarmes_selecionados_turbina_a.csv, 47
    tags, vs a lista curada usada no hit_rate oficial). Resolvido do
    mesmo jeito que os outros CSVs do dataset (local ou ClearML
    Dataset). Retorna vazio (sem lancar erro) se ALERT_CONTEXT_CATALOG_CSV
    nao estiver configurado."""
    if not cfg.ALERT_CONTEXT_CATALOG_CSV:
        return pd.DataFrame(columns=["Tag", "Data da Ocorrencia"])

    if cfg.USE_CLEARML_DATASET:
        from clearml import Dataset

        dataset_id = (cfg.CLEARML_DATASET_ID or os.getenv("CLEARML_DATASET_ID", "")).strip()
        if dataset_id:
            dataset = Dataset.get(dataset_id=dataset_id)
        else:
            dataset = Dataset.get(dataset_name=cfg.CLEARML_DATASET_NAME, dataset_project=cfg.CLEARML_PROJECT_NAME)
        dataset_root = Path(dataset.get_local_copy())
        catalog_path = _resolve_dataset_file(dataset_root, cfg.ALERT_CONTEXT_CATALOG_CSV)
    else:
        catalog_path = cfg.ALERT_CONTEXT_CATALOG_CSV

    df_catalog = pd.read_csv(catalog_path)
    if "Data da Ocorrência" in df_catalog.columns and "Data da Ocorrencia" not in df_catalog.columns:
        df_catalog["Data da Ocorrencia"] = df_catalog["Data da Ocorrência"]
    if "Tag Alarme" in df_catalog.columns and "Tag" not in df_catalog.columns:
        df_catalog["Tag"] = df_catalog["Tag Alarme"]
    if "Status" in df_catalog.columns:
        df_catalog = df_catalog[df_catalog["Status"].astype(str).str.startswith("ACT")].copy()

    df_catalog = _process_time_column(df_catalog, "Data da Ocorrencia", cfg, "alert_context_catalog")
    return df_catalog
