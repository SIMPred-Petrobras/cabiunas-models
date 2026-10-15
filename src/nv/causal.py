"""Causalidade: o que um detector ao vivo pode saber em cada instante.

Duas ferramentas:
  1. `trips_conhecidos`: o registro de trips só vale a partir do instante do trip (um trip em f só é conhecido em t > f).
     O `vb` da pesquisa apagava a referência em torno de trips FUTUROS no backtest, e isso foi otimista (M32 da pesquisa).
  2. `auditoria_do_futuro`: para qualquer função de atributos f(df) -> DataFrame, altera o dado DEPOIS de cada corte e exige que
     os valores até o corte fiquem idênticos. É o teste que pega vazamento em normalização, janelas centradas, EWMA mal feito etc.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

from .io import REGISTRO


def carrega_registro(caminho: Path = REGISTRO) -> list[pd.Timestamp]:
    d = pd.read_csv(caminho)
    col = "evento" if "evento" in d.columns else d.columns[0]
    return sorted(pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t).tz_localize("UTC") for t in d[col])


def trips_conhecidos(registro: list[pd.Timestamp], t: pd.Timestamp) -> list[pd.Timestamp]:
    """Trips já ocorridos ESTRITAMENTE antes de `t`."""
    return [f for f in registro if f < t]


def auditoria_do_futuro(f: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame, cortes: list[pd.Timestamp], semente: int = 20261015, tol: float = 0.0) -> dict:
    """Vazamento do futuro: embaralha o dado depois de cada corte e compara f(df) com f(df_alterado) até o corte.

    Passa se a diferença máxima nos instantes <= corte é <= tol. Devolve a maior diferença por corte.
    """
    rng = np.random.default_rng(semente)
    base = f(df)
    saida = {}
    for c in cortes:
        alt = df.copy()
        depois = alt.index > c
        num = alt.select_dtypes("number").columns
        alt[num] = alt[num].astype("float64")                    # a grade vem em float32 e o pandas 3 não aceita gravar float64 nela
        if depois.any():
            vals = alt.loc[depois, num].to_numpy(float)
            alt.loc[depois, num] = vals[rng.permutation(len(vals))] * (1.0 + 0.5 * rng.standard_normal(vals.shape))
        r = f(alt)
        a, b = base.loc[:c], r.loc[:c]
        comuns = a.columns.intersection(b.columns)
        x, y = a[comuns].to_numpy(float), b[comuns].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        dif = np.abs(x[m] - y[m]).max() if m.any() else 0.0
        iguais_nan = bool((np.isnan(x) == np.isnan(y)).all())
        saida[str(c)] = dict(dif_max=float(dif), nan_iguais=iguais_nan)
    ok = all(v["dif_max"] <= tol and v["nan_iguais"] for v in saida.values())
    return dict(passa=ok, por_corte=saida)
