#!/usr/bin/env python3
"""Dados de 30 s de out-dez/2024 (e jan-fev/2025, para validação) a partir dos xlsx interpolados (`tagsselecionadas_30s`), no formato do CSV bruto de produção:
hora local + 3 h = UTC (validado em jan/2024), texto do historiador -> NaN, prefixo `bapiha02-` removido. Saída: data_local/raw30s_2024q4_2025q1.parquet (fora do git)."""
from pathlib import Path
import pandas as pd

RAIZ = Path("/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabv2/cabiunas-models/estrutura_pra_prod/Cabiunas/dados/tagsselecionadas_30s")
SAIDA = Path(__file__).resolve().parents[1] / "data_local"; partes = []
for ano, meses in ((2024, (9, 10, 11, 12)), (2025, (1, 2))):
    for m in meses:
        f = next((RAIZ / str(ano)).glob(f"data_{ano}-{m:02d}-*interpolated.xlsx"))
        d = pd.read_excel(f); d = d.rename(columns={c: c.replace("bapiha02-", "") for c in d.columns}); t = pd.to_datetime(d.pop("data_datetime")) + pd.Timedelta(hours=3)
        d = d.apply(pd.to_numeric, errors="coerce"); d.index = t.dt.tz_localize("UTC"); d.index.name = "ts"; partes.append(d); print(f"   {f.name}: {d.shape}", flush=True)
g = pd.concat(partes).sort_index(); g = g[~g.index.duplicated()]; g.to_parquet(SAIDA / "raw30s_2024q4_2025q1.parquet"); print(g.shape, g.index[0], g.index[-1])
