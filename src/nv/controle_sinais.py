"""Os quatro sinais do v2 (t, p, sp, vb, já suavizados) e a máscara, extraídos pelo caminho de PRODUÇÃO de forma CAUSAL.

O controle é uma caixa-preta: aqui só se LÊ o pacote de produção (`cabiunas_inference`) do worktree da pesquisa, nunca se edita. A extração repete o replay do modo sombra:
a cada corte semanal o `preprocessar` recebe a janela de 75 d e o registro de trips só com os trips já ocorridos; o intervalo novo é fatiado nos instantes dos trips do registro
(cada trecho só conhece os trips anteriores a ele), e só as linhas depois do corte anterior são guardadas. Cada linha tem, portanto, o valor que um detector ao vivo teria visto.
"""
from __future__ import annotations
import contextlib, io, sys
from pathlib import Path
import pandas as pd

from .io import RAIZ

PKG = RAIZ / "estrutura_pra_prod" / "Cabiunas"
JANELA = pd.Timedelta(days=75)


def _ci():
    sys.path.insert(0, str(PKG / "scripts"))
    import cabiunas_inference as ci
    return ci


def sinais_causais(inicio: pd.Timestamp, cache: Path | None = None, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """`extra`: dados de 30 s ANTERIORES ao CSV bruto (mesmas colunas, índice UTC), concatenados antes de `preparar_grade` para estender o replay para trás."""
    if cache is not None and cache.exists():
        return pd.read_parquet(cache)
    ci = _ci()
    csv = next((PKG / "dados" / "2025_2026").glob("data_*_raw.csv"))
    with contextlib.redirect_stdout(io.StringIO()):
        raw = ci.carregar_dados(csv)
        if extra is not None:
            raw = pd.concat([extra.loc[extra.index < raw.index[0], raw.columns], raw]).sort_index()
        df = ci.preparar_grade(raw)
        modelos = ci.carregar_modelos(PKG / "modelos")
    trips = sorted(ci.carregar_trips(PKG / "registro_trips.csv"))
    real = df["RUNNING_A"].notna() & df[[c for c in df.columns if c.startswith("TV_")]].notna().any(axis=1)
    df = df.loc[:df.index[real][-1]]; fim = df.index[-1]
    cortes = [inicio]
    while cortes[-1] + pd.Timedelta(days=7) < fim:
        cortes.append(cortes[-1] + pd.Timedelta(days=7))
    cortes.append(fim)
    pedacos, anterior = [], None
    for c in cortes:
        sub = [f for f in trips if anterior is not None and anterior < f < c] + [c]
        for s in sub:
            conhecidos = trips if s == c else [t for t in trips if t < s]
            w = df.loc[max(df.index[0], s - JANELA):s]
            with contextlib.redirect_stdout(io.StringIO()):
                out = ci.preprocessar(modelos, w, trips=conhecidos)
            novo = out[out.index > anterior] if anterior is not None else out
            pedacos.append(novo[["t", "p", "sp", "vb", "mask", "reset"]]); anterior = s
    res = pd.concat(pedacos)
    res = res[~res.index.duplicated(keep="first")]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True); res.to_parquet(cache)
    return res
