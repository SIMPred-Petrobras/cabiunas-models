"""Limpeza dos dados PI: status-objects, sentinelas físicas, sensores congelados."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def strip_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Remove o prefixo 'bapiha02-' dos nomes de colunas."""
    return df.rename(columns=lambda c: c.removeprefix(config.PREFIX))


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Células com dicts de status do PI ('Out of Serv', 'Comm Fail', ...) → NaN."""
    return df.apply(pd.to_numeric, errors="coerce")


def apply_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Valores fora da faixa física da família do sensor → NaN.

    Remove as sentinelas de interpolação (-40.51 em termopares, -19.06/-11.02
    em temperaturas/pressões etc.) sem enumerá-las uma a uma.
    """
    out = df.copy()
    for col in out.columns:
        fam = config.family(col)
        if fam == "discrete":
            continue
        lo, hi = config.PHYSICAL_RANGE[fam]
        s = out[col]
        out[col] = s.where((s >= lo) & (s <= hi))
    return out


def freeze_flags(df: pd.DataFrame, running: pd.Series) -> pd.DataFrame:
    """Flag por sensor: janela FREEZE_WINDOW com std==0 durante operação.

    Retorna DataFrame booleano alinhado a df (True = congelado).
    """
    win = config.FREEZE_WINDOW
    stds = df[config.SENSOR_TAGS].rolling(win).std()
    frozen = stds.eq(0.0) & running.eq(1).to_numpy()[:, None]
    return frozen


def stable_mask(operability: pd.Series, threshold: float | None = None) -> pd.Series:
    """Máscara de operação estável e fora do transiente pós-partida.

    Para um tag discreto, use ``threshold=None`` (valor 1 = operação). Para
    NGP_A, use o limiar validado da rotação do gerador de gás. A função é
    deliberadamente agnóstica ao tag para evitar que RUNNING_A volte a ser a
    fonte de verdade por acidente.
    """
    run = operability.eq(1) if threshold is None else operability.ge(threshold)
    starts = run & ~run.shift(fill_value=False)
    # exclui STARTUP_EXCLUDE após cada partida
    n = int(pd.Timedelta(config.STARTUP_EXCLUDE) / pd.Timedelta(config.GRID))
    recent_start = starts.rolling(n, min_periods=1).max().astype(bool)
    return run & ~recent_start
