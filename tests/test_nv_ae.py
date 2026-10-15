import numpy as np
import pandas as pd
import torch

from nv.causal import auditoria_do_futuro
from nv.contexto import tau_horas, nivel_lento, features_tau
from nv.modelos.ae_torch import ModeloAE


def _g(n=6000, semente=0):
    idx = pd.date_range("2025-01-01", periods=n, freq="2min", tz="UTC"); rng = np.random.default_rng(semente)
    run = np.ones(n); run[1000:1100] = 0; run[3000:3003] = 0            # uma parada longa e uma lacuna de 6 min
    return pd.DataFrame({"RUNNING_A": run, "a": rng.standard_normal(n).cumsum() * 0.01 + 10, "b": rng.standard_normal(n) + 5}, index=idx)


def test_tau_zera_na_partida_e_funde_lacuna_curta():
    g = _g(); tau = tau_horas(g)
    assert np.isnan(tau.iloc[1050]) and tau.iloc[1100] == 0.0                           # parada longa: reinicia
    assert tau.iloc[3020] > tau.iloc[2990]                                              # lacuna de 6 min (< 10 min): NÃO reinicia


def test_contexto_passa_a_auditoria_do_futuro():
    g = _g(); est = np.ones(len(g), bool)
    f = lambda d: pd.concat([tau_horas(d).rename("tau"), nivel_lento(d, est, ["a", "b"], janela_h=6, atraso_h=3)], axis=1)
    assert auditoria_do_futuro(f, g, [g.index[2500], g.index[4500]])["passa"]


def test_ae_reconstroi_e_denuncia_estrutura_quebrada():
    torch.set_num_threads(2); rng = np.random.default_rng(1); L = rng.standard_normal((4000, 2)); W = rng.standard_normal((2, 6))
    X = L @ W + 0.05 * rng.standard_normal((4000, 6)) + 10; C = np.zeros((4000, 0))
    m = ModeloAE(6, 0, semente=0).fit(X, C, epocas=12)
    out = X[:100].copy(); out[:, 3] += 6.0                                               # quebra a relação com as outras tags
    s_n = m.score(X[2000:3000], C[2000:3000]); s_o = m.score(out, C[:100])
    assert np.median(s_o) > np.quantile(s_n, 0.95)
