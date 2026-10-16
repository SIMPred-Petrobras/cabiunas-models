#!/usr/bin/env python3
"""Valida o alinhamento de fuso dos xlsx de 30 s (tagsselecionadas_30s) contra a grade da pesquisa em jan/2024 (que já existe): o xlsx reamostrado a 2 min (média, rótulo à esquerda) deve coincidir com a grade em algum deslocamento (0, +-3 h)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nv import io as nvio

RAIZ = Path("/home/thallys/Documents/projeto-petrobras/Analise-exploratoria-dos-dados/analise_cabiunas/cabv2/cabiunas-models/estrutura_pra_prod/Cabiunas/dados/tagsselecionadas_30s")
g = nvio.carrega_grade(); f = sorted((RAIZ / "2024").glob("data_2024-01-*interpolated.xlsx"))[0]
d = pd.read_excel(f); d = d.rename(columns={c: c.replace("bapiha02-", "") for c in d.columns}); t = pd.to_datetime(d["data_datetime"])
print(f.name, d.shape, "| datetime", t.iloc[0], "->", t.iloc[-1], "| tz:", t.dt.tz)
d = d.drop(columns="data_datetime"); d.index = t
nao_num = {c: int(pd.to_numeric(d[c], errors="coerce").isna().sum() - d[c].isna().sum()) for c in d.columns if d[c].dtype == object}
print("colunas com texto (valores não numéricos):", {k: v for k, v in nao_num.items() if v})
d = d.apply(pd.to_numeric, errors="coerce")
comuns = [c for c in d.columns if c in g.columns]; print(f"colunas em comum com a grade: {len(comuns)} de {d.shape[1]}")
for deslocamento in (-3, 0, 3):
    idx = (t + pd.Timedelta(hours=deslocamento))
    idx = idx.dt.tz_localize("UTC") if idx.dt.tz is None else idx.dt.tz_convert("UTC")
    x = d.copy(); x.index = idx; x2 = x[comuns].resample("2min").mean()
    j = x2.index.intersection(g.index); a = x2.loc[j]; b = g.loc[j, comuns]
    ok = a.notna() & b.notna(); err = (a.where(ok) - b.where(ok)).abs(); rel = (err / (b.where(ok).abs() + 1e-9)).median().median()
    print(f"deslocamento {deslocamento:+d} h: {len(j)} instantes em comum | mediana (entre tags) do erro relativo mediano: {rel:.2e}")
