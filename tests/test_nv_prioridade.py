import numpy as np
import pandas as pd

from nv.prioridade import aplica, carga_nascimento, prioriza, promocao


def _t5(n_dias=40, salto_em=None, nivel=500.0):
    idx = pd.date_range("2025-01-01", periods=n_dias * 720, freq="2min", tz="UTC")
    rng = np.random.default_rng(0); s = pd.Series(nivel + rng.standard_normal(len(idx)), index=idx)
    if salto_em is not None:
        s.loc[salto_em:] += 10.0
    return s


def test_carga_baixa_e_promocao_quando_sobe():
    s = _t5(); s.loc["2025-02-01":] -= 10.0                       # nasce em carga baixa
    a, b = pd.Timestamp("2025-02-01", tz="UTC"), pd.Timestamp("2025-02-03", tz="UTC")
    s.loc["2025-02-02":] += 20.0                                  # a carga sobe no dia seguinte
    assert carga_nascimento(s, a) < 0.1
    t = promocao(s, a, b)
    assert t is not None and pd.Timestamp("2025-02-02", tz="UTC") <= t <= pd.Timestamp("2025-02-02 02:00", tz="UTC")


def test_promocao_e_causal():
    s = _t5(); s.loc["2025-02-01":] -= 10.0; s.loc["2025-02-02 05:00":] += 20.0
    a, b = pd.Timestamp("2025-02-01", tz="UTC"), pd.Timestamp("2025-02-03", tz="UTC")
    t = promocao(s, a, b)
    assert promocao(s.loc[:t], a, b) == t                          # cortar o dado no instante não muda o instante


def test_sem_subida_fica_baixa_e_aplica_tira_o_episodio():
    s = _t5(); s.loc["2025-02-01":] -= 10.0
    a, b = pd.Timestamp("2025-02-01", tz="UTC"), pd.Timestamp("2025-02-01 12:00", tz="UTC")
    tab = prioriza([(a, b)], s)
    assert tab.loc[0, "estatica"] == "baixa" and tab.loc[0, "dinamica"] == "baixa"
    al = pd.Series(False, index=s.index); al.loc[a:b] = True
    assert not aplica(al, tab, "estatica").any() and not aplica(al, tab, "dinamica").any()


def test_pendente_sem_dado_da_notificacao():
    s = _t5(); a = s.index[-1] - pd.Timedelta(hours=1)
    assert prioriza([(a, s.index[-1])], s).loc[0, "estatica"] == "pendente"
