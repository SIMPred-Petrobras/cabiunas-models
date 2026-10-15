import numpy as np
import pandas as pd
import pytest

from nv.causal import auditoria_do_futuro, trips_conhecidos


def _df(n=500):
    idx = pd.date_range("2025-01-01", periods=n, freq="2min", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.standard_normal(n).cumsum(), "y": rng.standard_normal(n)}, index=idx)


def causal_ewma(d):                       # só olha o passado
    return d.ewm(halflife=20).mean()


def vazador_centrado(d):                  # janela CENTRADA: usa o futuro
    return d.rolling(21, center=True, min_periods=1).mean()


def vazador_normalizacao_global(d):       # z com média e desvio do dado INTEIRO
    return (d - d.mean()) / d.std()


def test_ewma_causal_passa():
    d = _df(); cortes = [d.index[100], d.index[300]]
    assert auditoria_do_futuro(causal_ewma, d, cortes)["passa"]


def test_janela_centrada_e_pega():
    d = _df(); cortes = [d.index[100], d.index[300]]
    assert not auditoria_do_futuro(vazador_centrado, d, cortes)["passa"]


def test_normalizacao_global_e_pega():
    d = _df(); cortes = [d.index[100], d.index[300]]
    assert not auditoria_do_futuro(vazador_normalizacao_global, d, cortes)["passa"]


def test_trips_conhecidos_estritamente_antes():
    f = [pd.Timestamp("2025-03-17 18:16", tz="UTC"), pd.Timestamp("2025-04-07 21:18", tz="UTC")]
    assert trips_conhecidos(f, pd.Timestamp("2025-03-17 18:16", tz="UTC")) == []
    assert trips_conhecidos(f, pd.Timestamp("2025-03-17 18:18", tz="UTC")) == [f[0]]
