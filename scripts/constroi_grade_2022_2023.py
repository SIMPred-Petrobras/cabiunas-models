#!/usr/bin/env python3
"""Grade de 2 min de fev/2022 a dez/2023 a partir dos xlsx de 30 s (`tagsselecionadas_30s`), com o MESMO tratamento validado em jan/2024 (`valida_fuso_xlsx_2024.py`):
  hora local do xlsx + 3 h = UTC; valores de texto do historiador viram NaN; média de 2 min com rótulo à esquerda; prefixo `bapiha02-` das colunas removido.
Cache por mês em data_local/grade_2022_2023/ (parquet) e o consolidado em data_local/grade_2022_2023.parquet (fora do git).
"""
import sys
from pathlib import Path
import pandas as pd

RAIZ = Path("/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabv2/cabiunas-models/estrutura_pra_prod/Cabiunas/dados/tagsselecionadas_30s")
SAIDA = Path(__file__).resolve().parents[1] / "data_local"; MES = SAIDA / "grade_2022_2023"; MES.mkdir(parents=True, exist_ok=True)
partes = []
for ano in (2022, 2023):
    for f in sorted((RAIZ / str(ano)).glob("data_*interpolated.xlsx")):
        out = MES / (f.stem.replace("_interpolated", "") + ".parquet")
        if not out.exists():
            d = pd.read_excel(f); d = d.rename(columns={c: c.replace("bapiha02-", "") for c in d.columns}); t = pd.to_datetime(d.pop("data_datetime")) + pd.Timedelta(hours=3)
            d = d.apply(pd.to_numeric, errors="coerce"); d.index = t.dt.tz_localize("UTC"); d.resample("2min").mean().to_parquet(out)
            print(f"   {f.name} -> {out.name}", flush=True)
        partes.append(pd.read_parquet(out))
g = pd.concat(partes).sort_index(); g = g[~g.index.duplicated()]; g.to_parquet(SAIDA / "grade_2022_2023.parquet")
print(f"grade 2022-2023: {g.shape}, {g.index[0]} a {g.index[-1]}", flush=True)
