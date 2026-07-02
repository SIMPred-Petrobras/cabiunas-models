"""
io.py — Carregamento de dados Transpetro a partir de arquivos .feather.

Suporta dois formatos de feather:
  - Largo  (wide): cada coluna é um sensor; linha por timestamp.
                   O índice pode ser DatetimeIndex ou existe coluna 'Data Hora'/'Timestamp'.
  - Longo  (long): colunas Timestamp, Value, Quality, Tagname, Descrição (formato HISTORIAN/IFIX).
                   Pivotado por Descrição → wide; duplicatas resolvidas pela média.

Interface pública:
  load_data_transpetro(cfg) → (df_alarm, df_feat, df_raw, report)
  Mesma assinatura de src.cnn1d_ae.io.load_data().
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..cnn1d_ae.config import PipelineConfig
from ..cnn1d_ae.io import (
    _to_utc_indexed_series,
    _log_time_audit,
    build_time_integrity_report,
)

_LONG_FORMAT_COLS = {"Timestamp", "Value", "Tagname", "Descrição"}


# ---------------------------------------------------------------------------
# Detecção e pivô do formato longo
# ---------------------------------------------------------------------------

def _is_long_format(df: pd.DataFrame) -> bool:
    return _LONG_FORMAT_COLS.issubset(set(df.columns))


def _pivot_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivota feather longo (Timestamp/Value/Tagname/Descrição) para formato largo.
    Descrição se torna nome de coluna; duplicatas (mesmo timestamp + descrição)
    são resolvidas pela média.
    Retorna DataFrame com coluna 'Timestamp' (datetime) + colunas de sensores.
    """
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["Descrição"] = df["Descrição"].astype(str).str.strip()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    grouped = (
        df.groupby(["Timestamp", "Descrição"], sort=False)["Value"]
        .mean()
        .reset_index()
    )

    wide = grouped.pivot(index="Timestamp", columns="Descrição", values="Value")
    wide.columns.name = None
    wide = wide.reset_index()
    return wide


# ---------------------------------------------------------------------------
# Normalização da coluna de tempo
# ---------------------------------------------------------------------------

def _prepare_time_column(
    df: pd.DataFrame,
    time_col_out: str,
    source_tz: str,
    target_tz: str,
    apply_hour_shift: bool,
    shift_hours: int,
    log_samples: int,
    name: str,
) -> pd.DataFrame:
    """
    Encontra a coluna/índice de tempo, converte timezone e renomeia para time_col_out.
    Funciona para feathers com DatetimeIndex, coluna 'Timestamp' ou coluna 'Data Hora'.
    """
    df = df.copy()

    # Se o índice já é datetime, traz para coluna
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Localiza a coluna de tempo
    time_col_src = None
    for candidate in (time_col_out, "Timestamp", "Data Hora", "datetime", "time"):
        if candidate in df.columns:
            time_col_src = candidate
            break
    if time_col_src is None:
        # Procura qualquer coluna datetime
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                time_col_src = c
                break
    if time_col_src is None:
        raise ValueError(
            f"[TRANSPETRO] Coluna de tempo não encontrada em '{name}'. "
            f"Colunas: {list(df.columns)}"
        )

    raw = pd.to_datetime(df[time_col_src], errors="coerce")

    if source_tz != target_tz or apply_hour_shift:
        converted = _to_utc_indexed_series(
            raw, source_tz, target_tz, apply_hour_shift, shift_hours
        )
        _log_time_audit(name, raw, converted, log_samples)
        converted = converted.dt.tz_localize(None)
    else:
        # Sem conversão — garante apenas naive UTC
        converted = raw.dt.tz_localize(None) if raw.dt.tz is not None else raw

    df[time_col_out] = converted
    if time_col_src != time_col_out:
        df = df.drop(columns=[time_col_src])

    df = df.dropna(subset=[time_col_out]).sort_values(time_col_out).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Tabela de alarmes a partir da data de falha documentada
# ---------------------------------------------------------------------------

def _build_alarm_df(failure_date_str: str) -> pd.DataFrame:
    """
    Constrói df_alarm com coluna 'Data da Ocorrencia' a partir de uma string.
    Múltiplas datas separadas por ";".
    Retorna DataFrame vazio se failure_date_str estiver em branco.
    """
    if not failure_date_str or not failure_date_str.strip():
        return pd.DataFrame(columns=["Data da Ocorrencia"])

    parsed = []
    for raw in failure_date_str.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        # ISO (YYYY-...) → dayfirst=False; BR (DD/MM/...) → dayfirst=True
        is_iso = bool(re.match(r"^\d{4}[-/]", raw))
        ts = pd.to_datetime(raw, dayfirst=not is_iso, errors="coerce")
        if pd.notna(ts):
            parsed.append(ts)
        else:
            warnings.warn(f"[TRANSPETRO] Não foi possível parsear FAILURE_DATE: '{raw}'")

    if not parsed:
        return pd.DataFrame(columns=["Data da Ocorrencia"])

    return pd.DataFrame({"Data da Ocorrencia": parsed})


# ---------------------------------------------------------------------------
# Resolução do caminho do feather (local ou ClearML Dataset)
# ---------------------------------------------------------------------------

def _resolve_feather_from_clearml(cfg: PipelineConfig, feather_path: str) -> Path:
    from clearml import Dataset
    dataset_id = (cfg.CLEARML_DATASET_ID or "").strip()
    if dataset_id:
        ds = Dataset.get(dataset_id=dataset_id)
    else:
        ds = Dataset.get(
            dataset_name="Transpetro 2025",
            dataset_project=(cfg.CLEARML_PROJECT_NAME or "TranspetroML"),
        )
    dataset_root = Path(ds.get_local_copy())
    filename = Path(feather_path).name
    candidates = list(dataset_root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(
            f"[TRANSPETRO] Feather '{filename}' não encontrado no dataset ClearML "
            f"(root: {dataset_root})"
        )
    return candidates[0]


def _resolve_feather_path(cfg: PipelineConfig) -> Path:
    path = (cfg.FEATHER_PATH or "").strip()
    if not path:
        raise ValueError(
            "FEATHER_PATH não definido na config. "
            "Defina o caminho do arquivo .feather do equipamento."
        )
    if cfg.USE_CLEARML_DATASET:
        return _resolve_feather_from_clearml(cfg, path)

    base = (cfg.FEATHER_BASE_DIR or "").strip()
    full = Path(base) / path if base else Path(path)
    if not full.is_file():
        raise FileNotFoundError(f"[TRANSPETRO] Feather não encontrado: {full}")
    return full


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def load_data_transpetro(
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Lê feather de equipamento Transpetro; devolve (df_alarm, df_feat, df_raw, report).

    - df_alarm : datas de falha documentadas (campo FAILURE_DATE do config)
    - df_feat  : igual a df_raw (sem features pré-calculadas separadas)
    - df_raw   : DataFrame largo com cfg.TIME_COL + colunas de sensores
    - report   : dict com integridade temporal
    """
    feather_path = _resolve_feather_path(cfg)
    equip = cfg.EQUIPMENT_ID or feather_path.stem

    print(f"[TRANSPETRO] Equipamento : {equip}")
    print(f"[TRANSPETRO] Feather     : {feather_path}")

    df_src = pd.read_feather(feather_path)

    if _is_long_format(df_src):
        fmt = "long"
        print(f"[TRANSPETRO] Formato     : longo → pivotando por Descrição")
        df_wide = _pivot_long(df_src)
    else:
        fmt = "wide"
        print(f"[TRANSPETRO] Formato     : largo")
        df_wide = df_src

    print(f"[TRANSPETRO] Shape bruto : {df_wide.shape}")

    df_raw = _prepare_time_column(
        df=df_wide,
        time_col_out=cfg.TIME_COL,
        source_tz=cfg.SOURCE_TZ,
        target_tz=cfg.TARGET_TZ,
        apply_hour_shift=cfg.APPLY_HOUR_SHIFT,
        shift_hours=cfg.SHIFT_HOURS,
        log_samples=cfg.LOG_TIME_AUDIT_SAMPLES,
        name=equip,
    )

    sensor_cols = [c for c in df_raw.columns if c != cfg.TIME_COL]
    print(
        f"[TRANSPETRO] Sensores    : {len(sensor_cols)} | "
        f"Linhas: {len(df_raw):,} | "
        f"{df_raw[cfg.TIME_COL].min()} → {df_raw[cfg.TIME_COL].max()}"
    )
    print(f"[TRANSPETRO] Colunas     : {sensor_cols[:6]}{'...' if len(sensor_cols) > 6 else ''}")

    df_alarm = _build_alarm_df(cfg.FAILURE_DATE)
    if len(df_alarm):
        print(
            f"[TRANSPETRO] Falha(s)    : {df_alarm['Data da Ocorrencia'].tolist()}"
        )
        if cfg.FAILURE_DESCRIPTION:
            print(f"[TRANSPETRO] Descrição   : {cfg.FAILURE_DESCRIPTION}")
    else:
        print("[TRANSPETRO] FAILURE_DATE não configurado — sem exclusão de alarme.")

    report = {
        "raw": build_time_integrity_report(df_raw, cfg.TIME_COL, equip),
        "alarm": {
            "n_failures": len(df_alarm),
            "failure_dates": [
                str(d) for d in df_alarm.get("Data da Ocorrencia", pd.Series([]))
            ],
            "description": cfg.FAILURE_DESCRIPTION,
        },
        "feather_path": str(feather_path),
        "feather_format": fmt,
        "n_sensors": len(sensor_cols),
    }
    print("[TIME-INTEGRITY]", json.dumps(report, ensure_ascii=False, default=str))

    return df_alarm, df_raw.copy(), df_raw, report
