import numpy as np
import pandas as pd

from nv.plugin import mapeia_quantis, substitui


def test_mapa_de_quantis_e_monotono_e_iguala_a_distribuicao():
    rng = np.random.default_rng(0); idx = pd.date_range("2025-01-01", periods=20000, freq="2min", tz="UTC")
    cand = pd.Series(np.exp(rng.standard_normal(20000)), index=idx)                      # lognormal
    ref = pd.Series(rng.gamma(2.0, 1.5, 20000), index=idx)
    vig = pd.Series(True, index=idx)
    m = mapeia_quantis(cand, ref, vig)
    o = np.argsort(cand.to_numpy())
    assert (np.diff(m.to_numpy()[o]) >= -1e-9).all()                                      # monótona
    assert abs(np.quantile(m.dropna(), 0.99) - np.quantile(ref, 0.99)) / np.quantile(ref, 0.99) < 0.02
    assert abs(np.quantile(m.dropna(), 0.5) - np.quantile(ref, 0.5)) / np.quantile(ref, 0.5) < 0.02


def test_mapa_respeita_a_mascara():
    rng = np.random.default_rng(1); idx = pd.date_range("2025-01-01", periods=1000, freq="2min", tz="UTC")
    cand = pd.Series(rng.random(1000), index=idx); ref = pd.Series(rng.random(1000) * 10, index=idx)
    vig = pd.Series(True, index=idx); vig.iloc[100:200] = False
    assert mapeia_quantis(cand, ref, vig).iloc[100:200].isna().all()


def test_substitui_so_troca_um_canal():
    idx = pd.date_range("2025-01-01", periods=10, freq="2min", tz="UTC")
    proc = pd.DataFrame({"t": 1.0, "p": 2.0, "sp": 3.0, "vb": 4.0, "mask": True, "reset": False}, index=idx)
    out = substitui(proc, "t", pd.Series(9.0, index=idx))
    assert (out["t"] == 9.0).all() and (out["p"] == 2.0).all() and (proc["t"] == 1.0).all()
