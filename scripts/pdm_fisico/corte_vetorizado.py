"""Versao vetorizada de corta_por_decaimento, para caber nas 2.187 configuracoes
da busca conjunta. Verificada contra a escalar em decaimento_pico.py."""
import numpy as np, pandas as pd


def corta_vet(voto: np.ndarray, forca: np.ndarray, frac: float) -> np.ndarray:
    """Dentro de cada corrida de `voto`, mata do primeiro ponto em que a forca
    cai abaixo de frac x o pico ja atingido naquela corrida, em diante."""
    v = np.asarray(voto, dtype=bool)
    if frac <= 0 or not v.any():
        return v
    grupo = np.cumsum(~v)                       # id da corrida (constante dentro dela)
    f = pd.Series(np.where(v, forca, -np.inf))
    f = f.where(np.isfinite(f), np.nan)
    pico = f.groupby(grupo).cummax()
    morto = (f < frac * pico).fillna(False).to_numpy()
    morto = pd.Series(morto).groupby(grupo).cummax().to_numpy().astype(bool)
    return v & ~morto
