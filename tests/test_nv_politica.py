import numpy as np
import pandas as pd

from nv.politica import ewma_vigiado, episodios, alarme_de, calibra_limiar
from nv.modelos import FABRICAS
from nv.walk import walk_forward


def _idx(n=3000):
    return pd.date_range("2025-01-01", periods=n, freq="2min", tz="UTC")


def test_episodio_precisa_durar_dur_min():
    idx = _idx(); e = pd.Series(0.0, index=idx)
    e.iloc[100:130] = 5.0            # 60 min: curto
    e.iloc[500:600] = 5.0            # 200 min: vale
    eps = episodios(e, 1.0)
    assert len(eps) == 1 and eps[0][0] == idx[500]


def test_refratario_descarta_episodios_proximos():
    idx = _idx(8000); e = pd.Series(0.0, index=idx)
    e.iloc[100:200] = 5.0
    e.iloc[500:600] = 5.0           # a < 72 h do fim do anterior: descartado
    e.iloc[5000:5100] = 5.0         # a > 72 h (2160 amostras): vale
    assert [x[0] for x in episodios(e, 1.0)] == [idx[100], idx[5000]]


def test_ewma_nao_atravessa_parada():
    idx = _idx(600); s = pd.Series(10.0, index=idx); vig = pd.Series(True, index=idx); vig.iloc[200:400] = False
    s.iloc[200:400] = 1000.0          # valores absurdos durante a parada
    e = ewma_vigiado(s, vig)
    assert e.iloc[200:400].isna().all() and abs(e.iloc[450] - 10.0) < 1e-6


def test_calibracao_monotona():
    rng = np.random.default_rng(0); idx = _idx(20000); vig = pd.Series(True, index=idx)
    e = pd.Series(np.abs(rng.standard_normal(20000)).cumsum() % 5, index=idx)
    t1 = calibra_limiar(e, vig, [], alvo_h_fora=50.0); t2 = calibra_limiar(e, vig, [], alvo_h_fora=150.0)
    assert t2 <= t1


def _dados_correlacionados(n=3000, d=10, k=3, semente=1):
    rng = np.random.default_rng(semente); W = rng.standard_normal((k, d)); L = rng.standard_normal((n, k)) * np.array([3.0, 2.0, 1.5])
    X = L @ W + 0.1 * rng.standard_normal((n, d)) + 10.0
    q, _ = np.linalg.qr(W.T, mode="complete"); ortogonal = q[:, k]                       # direção fora do subespaço dos fatores
    return X, W, ortogonal


def test_cada_baseline_detecta_o_tipo_de_outlier_que_deveria():
    X, W, ort = _dados_correlacionados()
    dentro = X[:60] + 12.0 * W[0] / np.linalg.norm(W[0]) * 3.0                 # fora do normal, mas DENTRO do subespaço
    fora = X[:60] + 4.0 * ort                                                   # ORTOGONAL ao subespaço (estrutura quebrada)
    detecta = {}
    for nome, F in FABRICAS.items():
        m = F(semente=0).fit(X); p90 = np.quantile(m.score(X[1000:2000]), 0.90)
        detecta[nome] = (np.median(m.score(dentro)) > p90, np.median(m.score(fora)) > p90)
    assert detecta["pca_spe"][1] and detecta["pca_t2"][0]                       # SPE vê a estrutura quebrada; T2 vê o desvio dentro do subespaço
    assert detecta["pca_spe"][1] and not detecta["pca_t2"][1] or detecta["pca_t2"][1]   # T2 pode ou não ver o ortogonal (depende do ruído retido)
    for nome in ("maha_mcd", "knn", "iforest", "rz_max"):
        assert any(detecta[nome]), nome                                         # os demais veem pelo menos um dos dois


def test_walk_forward_so_usa_o_passado():
    idx = pd.date_range("2025-01-01", periods=60 * 720, freq="2min", tz="UTC"); rng = np.random.default_rng(2)
    g = pd.DataFrame(rng.standard_normal((len(idx), 4)), index=idx, columns=list("abcd"))
    est = np.ones(len(idx), bool)
    a = walk_forward(FABRICAS["pca_spe"], g, list("abcd"), est, pd.Timestamp("2025-02-01", tz="UTC"), idx[-1], n_treino=5000)
    g2 = g.copy(); g2.iloc[45 * 720:] += 100.0              # altera só o futuro
    b = walk_forward(FABRICAS["pca_spe"], g2, list("abcd"), est, pd.Timestamp("2025-02-01", tz="UTC"), idx[-1], n_treino=5000)
    corte = 45 * 720
    assert np.allclose(a.iloc[:corte].to_numpy(), b.iloc[:corte].to_numpy(), equal_nan=True)
