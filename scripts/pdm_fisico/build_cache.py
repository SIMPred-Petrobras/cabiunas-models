#!/usr/bin/env python3
"""Cache 2-min a partir do CSV de 30 s (1,05 GB, 2,45 M linhas).

Faz o mínimo indispensável e nada mais:
  - objetos de status do PI ("No Data", "Out of Serv") viram NaN
  - faixa física por tipo de sensor (corta sentinela de termopar em -40,5 degC)
  - reamostra para 2 min por mediana (robusto a spike isolado, dispensa Hampel)
  - RUNNING_A vira fração da janela com maquina ligada
"""
import os, sys
import numpy as np, pandas as pd

SRC = "../../dados/sensores_full_2024_2026_30s.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grade2min.parquet")

cols = pd.read_csv(SRC, nrows=0).columns.tolist()
sig = [c for c in cols if c not in ("data_datetime", "any_sensor_constant_run")]

def faixa(c):
    b = c.split("_")[-1] if not c.startswith("954005") else c.split("_", 2)[2]
    if c.startswith("TC382") or c.startswith("T5"):   return (-15.0, 900.0)
    if "_TI_" in c or c.startswith("TI_"):            return (-15.0, 900.0)
    if c.startswith("TV_"):                           return (0.0, 200.0)
    if "_PDI" in c or c.startswith("PDI"):            return (-5.0, 200.0)
    if "_PI_" in c or c.startswith("PI_"):            return (-1.5, 200.0)
    return (-1e9, 1e9)

parts = []
rd = pd.read_csv(SRC, chunksize=250_000, low_memory=False)
for i, ch in enumerate(rd):
    ch["data_datetime"] = pd.to_datetime(ch["data_datetime"], utc=True, errors="coerce")
    ch = ch.dropna(subset=["data_datetime"]).set_index("data_datetime")
    d = {}
    for c in sig:
        v = pd.to_numeric(ch[c], errors="coerce")
        lo, hi = faixa(c)
        d[c] = v.where((v >= lo) & (v <= hi)).astype("float32")
    df = pd.DataFrame(d, index=ch.index)
    agg = df.resample("2min").median()
    agg["RUNNING_A"] = df["RUNNING_A"].resample("2min").mean()
    parts.append(agg.astype("float32"))
    print(f"  chunk {i} -> {ch.index[0]}  ({len(agg)} janelas)", flush=True)

g = pd.concat(parts)
g = g[~g.index.duplicated(keep="first")].sort_index()
g = g.reindex(pd.date_range(g.index[0], g.index[-1], freq="2min", tz="UTC"))
g.index.name = "ts"
g.to_parquet(OUT)
print(f"\n{OUT}: {g.shape}, {g.index[0]} .. {g.index[-1]}")
print("cobertura nao-nula por coluna (%):")
print((g.notna().mean() * 100).round(1).sort_values().to_string())
