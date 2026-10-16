import numpy as np
import pandas as pd

from nv.consenso import CHAVES, configura, consenso, sorteio


def test_sorteio_e_deterministico_e_dentro_da_faixa():
    a, b = sorteio(), sorteio()
    assert a == b and len(a) == 60 and list(a[0].keys()) == CHAVES
    assert all(0.9 <= f <= 1.1 for m in a for f in m.values())


def test_configura_multiplica_so_os_k():
    det = {"k_nivel_a": {"t": 1.0, "p": 2.0, "sp": 3.0, "vb": 4.0}, "k_nivel_b": {"t": 1.0, "p": 2.0, "sp": 3.0, "vb": 4.0}, "outro": 5}
    d = configura(det, {("a", "t"): 1.1, ("b", "vb"): 0.5})
    assert d["k_nivel_a"]["t"] == 1.1 and d["k_nivel_b"]["vb"] == 2.0 and d["k_nivel_a"]["p"] == 2.0 and d["outro"] == 5 and det["k_nivel_a"]["t"] == 1.0


def test_consenso_e_a_maioria():
    idx = pd.date_range("2025-01-01", periods=4, freq="2min", tz="UTC")
    proc = pd.DataFrame(index=idx)
    det = {"k_nivel_a": {c: 1.0 for c in ("t", "p", "sp", "vb")}, "k_nivel_b": {c: 1.0 for c in ("t", "p", "sp", "vb")}}

    def decide(p, d):  # alarma no instante 0 sempre, e no instante 1 só se k_a[t] > 1
        a = np.array([True, d["k_nivel_a"]["t"] > 1.0, False, False])
        return pd.DataFrame({"is_anomaly": a}, index=p.index)
    c = consenso(proc, det, decide, q=0.7)
    assert c.iloc[0] and not c.iloc[2] and not c.iloc[3]
    assert consenso(proc, det, decide, q=0.0).all()
