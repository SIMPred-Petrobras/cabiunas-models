"""
Converte os arquivos mensais xlsx de sensores (2025 - 30s) para um único CSV consolidado.

Saídas:
  ../dados/sensores_brutos_2025_30s.csv  — série completa com colunas limpas
  ../dados/alarmes_selecionados_turbina_a.csv — alarmes do xlsx para o ano

Uso:
    PYTHONPATH=. python scripts/convert_xlsx_to_csv.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import openpyxl


XLSX_DIR = Path("../dados/2025 - 30s")
ALARM_XLSX = Path("../dados/Alarmes Selecionados Turbina A.xlsx")
OUT_SENSORS = Path("../dados/sensores_brutos_2025_30s.csv")
OUT_ALARMS = Path("../dados/alarmes_selecionados_turbina_a.csv")

PREFIX = "bapiha02-"


def _clean_col(name: str) -> str:
    if name.startswith(PREFIX):
        return name[len(PREFIX):]
    return name


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_xlsx_month(path: Path) -> pd.DataFrame:
    print(f"  Lendo {path.name} ...", end=" ", flush=True)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header = [_clean_col(str(c)) for c in rows[0]]
    records = []
    for row in rows[1:]:
        rec = {}
        for col, val in zip(header, row):
            if col == "data_datetime":
                rec[col] = val  # datetime object
            else:
                rec[col] = _safe_float(val)
        records.append(rec)

    df = pd.DataFrame(records)
    df["data_datetime"] = pd.to_datetime(df["data_datetime"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["data_datetime"])
    print(f"{len(df):,} linhas ({n_before - len(df)} descartadas sem timestamp)")
    return df


def _mark_constant_runs(series: pd.Series, min_length: int = 3) -> pd.Series:
    """
    Marca True onde há runs de ≥ min_length valores idênticos consecutivos.
    Esses segmentos são artefatos de forward-fill upstream (não dados reais).
    """
    s = series.copy()
    mask = pd.Series(False, index=s.index)
    is_equal = (s == s.shift(1)) & s.notna() & s.shift(1).notna()

    # identificar grupos
    grp = (~is_equal).cumsum()
    run_len = is_equal.groupby(grp).transform("sum")
    # pontos com run_len >= min_length - 1 (o primeiro ponto do run tem is_equal=False)
    # propagar backward para incluir o ponto inicial
    suspect = run_len >= (min_length - 1)
    # inclui o ponto que iniciou o run (o anterior)
    mask = suspect | suspect.shift(-1).fillna(False)
    return mask


def convert_sensors() -> None:
    xlsx_files = sorted(XLSX_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"Nenhum xlsx encontrado em {XLSX_DIR}")

    print(f"Encontrados {len(xlsx_files)} arquivos mensais.")
    dfs = []
    for p in xlsx_files:
        df = _read_xlsx_month(p)
        dfs.append(df)

    print("Concatenando ...")
    full = pd.concat(dfs, ignore_index=True)
    full = full.sort_values("data_datetime").drop_duplicates(subset=["data_datetime"]).reset_index(drop=True)

    # Normalizar RUNNING_A: 0/1 limpo
    if "RUNNING_A" in full.columns:
        full["RUNNING_A"] = (full["RUNNING_A"].fillna(0.0) > 0.5).astype(int)

    # Marcar pontos de forward-fill suspeitos por sensor
    sensor_cols = [c for c in full.columns if c not in ("data_datetime", "RUNNING_A", "HSX_6240001A")]
    interp_flags = pd.DataFrame(False, index=full.index, columns=sensor_cols)
    print("Detectando runs de forward-fill por sensor ...")
    for col in sensor_cols:
        interp_flags[col] = _mark_constant_runs(full[col], min_length=3)

    # Coluna booleana consolidada: True se QUALQUER sensor tem forward-fill neste ponto
    full["any_sensor_constant_run"] = interp_flags.any(axis=1)

    # Salvar
    OUT_SENSORS.parent.mkdir(exist_ok=True)
    full.to_csv(OUT_SENSORS, index=False)
    print(f"\nSalvo: {OUT_SENSORS}  ({len(full):,} linhas, {len(full.columns)} colunas)")

    # Estatísticas rápidas
    if "RUNNING_A" in full.columns:
        on = full["RUNNING_A"].sum()
        total = len(full)
        print(f"RUNNING_A — ON: {on:,} ({100*on/total:.1f}%)  OFF: {total-on:,} ({100*(total-on)/total:.1f}%)")

    n_const = full["any_sensor_constant_run"].sum()
    on_mask = full.get("RUNNING_A", pd.Series(1, index=full.index)) == 1
    n_const_on = (full["any_sensor_constant_run"] & on_mask).sum()
    print(f"Forward-fill detectado — total: {n_const:,} | durante ON: {n_const_on:,}")


def convert_alarms() -> None:
    print(f"\nLendo alarmes de {ALARM_XLSX.name} ...")
    df = pd.read_excel(ALARM_XLSX, sheet_name="Alarmes Selecionados 2022-2026")
    df.columns = [c.strip() for c in df.columns]
    col_dt = "Data da Ocorrência"
    if col_dt in df.columns:
        df[col_dt] = pd.to_datetime(df[col_dt], dayfirst=True, errors="coerce")

    df.to_csv(OUT_ALARMS, index=False)
    print(f"Salvo: {OUT_ALARMS}  ({len(df):,} alarmes, {df['Tag Alarme'].nunique()} tags únicas)")


if __name__ == "__main__":
    convert_sensors()
    convert_alarms()
