import numpy as np
import pandas as pd
import pytest

from nv.avalia import avalia, episodios_de_serie, ic_blocos_mensais, cobertura_ao_acaso, p_binomial_de_deteccao


def _grade(dias=40):
    return pd.date_range("2025-01-01", periods=dias * 720, freq="2min", tz="UTC")


def _serie(idx, intervalos):
    s = pd.Series(False, index=idx)
    for a, b in intervalos:
        s.loc[pd.Timestamp(a, tz="UTC"):pd.Timestamp(b, tz="UTC")] = True
    return s


def test_episodios_corridas_contiguas():
    idx = _grade(3)
    s = _serie(idx, [("2025-01-01 02:00", "2025-01-01 04:00"), ("2025-01-02 00:00", "2025-01-02 00:10")])
    e = episodios_de_serie(s)
    assert len(e) == 2 and e[0][0] == pd.Timestamp("2025-01-01 02:00", tz="UTC") and e[1][1] == pd.Timestamp("2025-01-02 00:10", tz="UTC")


def test_detecao_lead_banda_e_fp():
    idx = _grade(40); vig = pd.Series(True, index=idx)
    T = pd.Timestamp("2025-01-20 12:00", tz="UTC")
    # alarme nasce 10 h antes do trip e dura até o trip; um FP longe
    al = _serie(idx, [("2025-01-20 02:00", "2025-01-20 12:00"), ("2025-01-05 00:00", "2025-01-05 03:00")])
    r = avalia(al, vig, [T])
    assert (r.det, r.inicio, r.banda, r.fp) == (1, 1, 1, 1)
    assert abs(r.leads_h[0] - 10.0) < 0.1
    assert r.horas_alarme_fora == pytest.approx(3 + 2 / 60, abs=0.05)
    assert r.fp_mes == pytest.approx(1 / (len(idx) * 2 / 60 / 730), rel=1e-6)


def test_alarme_que_nasce_antes_da_janela_detecta_mas_nao_e_inicio():
    idx = _grade(40); vig = pd.Series(True, index=idx)
    T = pd.Timestamp("2025-01-20 12:00", tz="UTC")
    al = _serie(idx, [("2025-01-17 00:00", "2025-01-20 06:00")])           # nasce 84 h antes
    r = avalia(al, vig, [T])
    assert (r.det, r.inicio, r.banda, r.fp) == (1, 0, 0, 0)


def test_sem_alarme_nao_detecta():
    idx = _grade(10); vig = pd.Series(True, index=idx)
    r = avalia(pd.Series(False, index=idx), vig, [pd.Timestamp("2025-01-05", tz="UTC")])
    assert (r.det, r.fp) == (0, 0) and r.horas_alarme == 0


def test_ic_blocos_contem_o_ponto_e_cresce_com_poucos_meses():
    idx = pd.date_range("2025-01-01", periods=90 * 720, freq="2min", tz="UTC"); vig = pd.Series(True, index=idx)
    al = _serie(idx, [("2025-01-10 00:00", "2025-01-10 05:00"), ("2025-02-10 00:00", "2025-02-10 05:00"), ("2025-03-10 00:00", "2025-03-10 05:00")])
    r = avalia(al, vig, []); ic = ic_blocos_mensais(al, vig, [])
    lo, hi = ic["fp_mes"]
    assert lo <= r.fp_mes <= hi and ic["meses"] == 3 and hi > lo


def test_cobertura_ao_acaso_proporcional_ao_tempo_em_alarme():
    idx = _grade(200); vig = pd.Series(True, index=idx)
    # alarmes de 2 h a cada 10 dias: a janela de 48 h tomada ao acaso quase sempre os contém
    al = pd.Series(False, index=idx)
    for d in range(2, 200, 10):
        al.loc[idx[d * 720]: idx[d * 720 + 60]] = True
    p = cobertura_ao_acaso(al, vig, [])
    assert 0.1 < p < 0.4                                                     # 48 h de janela / 240 h de ciclo = 0,2 mais a largura do alarme


def test_binomial_unilateral():
    assert p_binomial_de_deteccao(8, 8, 0.25) < 1e-4
    assert p_binomial_de_deteccao(2, 8, 0.25) > 0.3
