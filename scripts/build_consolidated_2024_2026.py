#!/usr/bin/env python3
"""
build_consolidated_2024_2026.py
Consolida a remessa de 2024 (12 XLSX mensais) com o CSV bruto de 2025–2026 num único
arquivo de pipeline cobrindo jan/2024 → abr/2026 com o CONJUNTO COMPLETO de tags —
inclusive os 18 instrumentos `954005_624_*` (pressão/temperatura), que até agora só
existiam de 2025 em diante.

⚠️ FUSO — a parte crítica. A remessa de 2024 vem em HORA LOCAL (UTC−3) e precisa de
`+3h`; os CSVs de pipeline estão em UTC. Provado por sonda (TC382_03_A), não assumido:

    2024 (remessa) vs sensores_2024h2_2025_2026_30s   → |erro| p50 = 0,000 °C em +180 min
                                                         (607.860 pontos; vizinhos ≥ 0,463)
    sensores_2024h2 vs sensores_brutos_2025_2026      → |erro| p50 = 0,000 °C em 0 min
                                                         (1.367.943 pontos)

Logo a cadeia fecha: remessa+3h ≡ 2024h2 ≡ brutos. O sinal é OPOSTO ao dos exports
`record_*` (que pedem −3h) — aplicar por analogia erraria em 6 horas.

Corte: 2024 vem da remessa, 2025+ vem do brutos (fonte já validada no pipeline). O
`any_sensor_constant_run` é recalculado só para o trecho de 2024 com o mesmo algoritmo
de `convert_xlsx_to_csv.py`; as linhas de 2025+ preservam o valor original.

Uso:
    PYTHONPATH=. python scripts/build_consolidated_2024_2026.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
from pathlib import Path

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "convert_xlsx_to_csv", os.path.join(_HERE, "convert_xlsx_to_csv.py"))
cx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cx)

XLSX_2024 = "../dados/2024"
BRUTOS = "../dados/sensores_brutos_2025_2026_30s.csv"
OUT = "../dados/sensores_full_2024_2026_30s.csv"
TZ_SHIFT = pd.Timedelta(hours=3)          # local (UTC−3) → UTC
CUT = pd.Timestamp("2025-01-01", tz="UTC")
CHUNK = 200_000


def build_2024() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(XLSX_2024, "*_2024_interpolated_TagsSelecionadas.xlsx")))
    if len(files) != 12:
        raise SystemExit(f"esperava 12 arquivos mensais em {XLSX_2024}, achei {len(files)}")
    print(f"[1/4] Lendo {len(files)} arquivos de 2024")
    df = pd.concat([cx._read_xlsx_month(Path(f)) for f in files], ignore_index=True)
    df["data_datetime"] = (pd.to_datetime(df["data_datetime"], errors="coerce")
                           .dt.tz_localize("UTC") + TZ_SHIFT)
    df = (df.dropna(subset=["data_datetime"])
            .sort_values("data_datetime")
            .drop_duplicates(subset=["data_datetime"])
            .reset_index(drop=True))
    df = df[df["data_datetime"] < CUT]
    if "RUNNING_A" in df.columns:
        df["RUNNING_A"] = (df["RUNNING_A"].fillna(0.0) > 0.5).astype(int)

    print("[2/4] Marcando forward-fill por sensor (mesmo algoritmo do conversor)")
    cols = [c for c in df.columns if c not in ("data_datetime", "RUNNING_A", "HSX_6240001A")]
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    for c in cols:
        flags[c] = cx._mark_constant_runs(df[c], min_length=3)
    df["any_sensor_constant_run"] = flags.any(axis=1)
    print(f"      2024: {len(df):,} linhas  {df['data_datetime'].min()} → {df['data_datetime'].max()}")
    # O brutos grava carimbo NAIVE ("2025-01-01 00:00:00"). Se a parte de 2024 sair
    # com sufixo "+00:00", o arquivo fica com DOIS formatos e o pandas 2.x coage a
    # metade não-inferida para NaT em silêncio — 1,4 M linhas somem sem erro.
    df["data_datetime"] = df["data_datetime"].dt.tz_localize(None)
    return df


def main() -> None:
    df24 = build_2024()

    head = pd.read_csv(BRUTOS, nrows=1)
    order = list(head.columns)
    faltando = [c for c in order if c not in df24.columns]
    if faltando:
        raise SystemExit(f"remessa 2024 não tem colunas do brutos: {faltando}")
    df24 = df24[order]

    print(f"[3/4] Escrevendo 2024 em {OUT}")
    df24.to_csv(OUT, index=False)
    n24 = len(df24)
    del df24

    print(f"[4/4] Anexando 2025+ de {os.path.basename(BRUTOS)} em blocos de {CHUNK:,}")
    n25 = 0
    with open(OUT, "a", encoding="utf-8") as fh:
        for chunk in pd.read_csv(BRUTOS, chunksize=CHUNK, low_memory=False):
            t = pd.to_datetime(chunk["data_datetime"], utc=True, errors="coerce")
            chunk = chunk[t >= CUT]
            if chunk.empty:
                continue
            chunk[order].to_csv(fh, index=False, header=False)
            n25 += len(chunk)
            print(f"      +{len(chunk):>7,}  (total 2025+: {n25:,})", end="\r", flush=True)

    mb = os.path.getsize(OUT) / 1e6
    print(f"\n\nGravado: {OUT}")
    print(f"  {n24:,} linhas de 2024  +  {n25:,} de 2025–2026  =  {n24 + n25:,}  ({mb:.0f} MB)")
    print(f"  {len(order)} colunas")


if __name__ == "__main__":
    main()
