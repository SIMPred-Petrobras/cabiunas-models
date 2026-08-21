"""Testes da política de baseline e do cálculo de limiar.

Série sintética de propósito: o que está sob teste é a seleção da janela e a
equivalência do limiar, não o comportamento do compressor.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]


def _carrega_automl():
    """Importa scripts/automl_clearml.py, que é autocontido por design."""
    caminho = RAIZ / "scripts" / "automl_clearml.py"
    spec = importlib.util.spec_from_file_location("cabiunas_automl_test", caminho)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo          # dataclasses precisam do módulo registrado
    spec.loader.exec_module(modulo)
    return modulo


automl = _carrega_automl()
BaselinePolicy = automl.BaselinePolicy
PASSO_S = 120.0                              # grade de 2 min
POR_HORA = int(3600 / PASSO_S)               # 30 amostras por hora


def elegiveis(horas: int, inicio: str = "2025-01-01") -> pd.DatetimeIndex:
    """Eixo de tempo elegível contínuo, com `horas` horas de operação."""
    return pd.date_range(inicio, periods=horas * POR_HORA, freq="2min")


# ------------------------------------------------------------------ janela
def test_janela_em_horas_pega_o_final_do_historico():
    idx = elegiveis(1000)
    pol = BaselinePolicy(window_hours=300)
    sel, diag = pol.select(idx, idx[-1] + pd.Timedelta("2min"), PASSO_S)
    assert len(sel) == 300 * POR_HORA
    assert diag["horas"] == pytest.approx(300.0)
    assert diag["truncado"] is False
    assert sel[-1] == idx[-1]                # é o trecho MAIS RECENTE


def test_janela_maior_que_o_historico_fica_truncada():
    idx = elegiveis(200)
    pol = BaselinePolicy(window_hours=3000)
    sel, diag = pol.select(idx, idx[-1] + pd.Timedelta("2min"), PASSO_S)
    assert len(sel) == len(idx)               # devolve tudo o que existe
    assert diag["truncado"] is True           # e avisa
    assert diag["horas"] == pytest.approx(200.0)


def test_acumulativo_e_o_caso_degenerado():
    idx = elegiveis(500)
    sel, diag = BaselinePolicy().select(idx, idx[-1] + pd.Timedelta("2min"), PASSO_S)
    assert len(sel) == len(idx)
    assert diag["truncado"] is False          # sem janela pedida, nada a truncar
    assert BaselinePolicy().label == "acum"


def test_janela_em_dias_corta_pelo_calendario():
    idx = elegiveis(24 * 30)                  # 30 dias corridos de operação
    fim = idx[-1] + pd.Timedelta("2min")
    sel, _ = BaselinePolicy(window_days=10).select(idx, fim, PASSO_S)
    assert sel[0] >= fim - pd.Timedelta(days=10)
    assert len(sel) == pytest.approx(10 * 24 * POR_HORA, rel=0.01)


def test_teto_de_idade_vence_a_janela_em_horas():
    """Com máquina parada no meio, 300 h alcançam mais de 30 dias — o teto corta."""
    antigo = pd.date_range("2025-01-01", periods=200 * POR_HORA, freq="2min")
    recente = pd.date_range("2025-03-01", periods=200 * POR_HORA, freq="2min")
    idx = antigo.append(recente)              # ~2 meses de parada entre os blocos
    fim = idx[-1] + pd.Timedelta("2min")

    so_horas, _ = BaselinePolicy(window_hours=300).select(idx, fim, PASSO_S)
    com_teto, diag = BaselinePolicy(window_hours=300, max_age_days=30).select(
        idx, fim, PASSO_S)

    assert len(so_horas) == 300 * POR_HORA    # alcança o bloco antigo
    assert so_horas[0] < pd.Timestamp("2025-02-01")
    assert len(com_teto) < len(so_horas)      # o teto descarta o bloco antigo
    assert com_teto[0] >= fim - pd.Timedelta(days=30)
    assert diag["truncado"] is True           # e por isso a janela não fecha


def test_piso_marca_o_retreino_como_invalido():
    idx = elegiveis(50)
    _, diag = BaselinePolicy(window_hours=300, min_hours=100).select(
        idx, idx[-1] + pd.Timedelta("2min"), PASSO_S)
    assert diag["valido"] is False
    _, diag_ok = BaselinePolicy(window_hours=300, min_hours=10).select(
        idx, idx[-1] + pd.Timedelta("2min"), PASSO_S)
    assert diag_ok["valido"] is True


def test_nao_usa_futuro():
    idx = elegiveis(1000)
    corte = idx[500 * POR_HORA]
    sel, _ = BaselinePolicy().select(idx, corte, PASSO_S)
    assert sel[-1] <= corte


# ------------------------------------------------------------------ limiar
@pytest.mark.parametrize("n", [9_000, 11_859, 47_439, 238_006])
@pytest.mark.parametrize("pct", [99.0, 99.5, 99.9, 99.97, 99.99, 99.995])
def test_limite_percentil_reproduz_nanpercentile(n: int, pct: float):
    """A cauda guardada tem que dar o MESMO número que o vetor inteiro.

    É isso que permite cachear só a cauda e ainda manter comparáveis os
    resultados das buscas anteriores.
    """
    rng = np.random.default_rng(n)
    valores = rng.lognormal(size=n)
    esperado = float(np.nanpercentile(valores, pct))

    m = min(n, max(8192, int(n * 0.012) + 16))
    cauda = np.sort(np.partition(valores, n - m)[n - m:])[::-1]
    obtido = automl.WalkForwardEvaluator._limite(cauda, n, pct, "percentil")
    assert obtido == pytest.approx(esperado, rel=1e-12)


def test_limite_k_maiores_e_o_k_esimo_maior():
    valores = np.arange(1000.0)
    cauda = valores[::-1].copy()
    for k in (1, 3, 10, 100):
        assert automl.WalkForwardEvaluator._limite(cauda, 1000, k, "k_maiores") == 1000 - k


def test_k_maiores_independe_do_tamanho_da_janela():
    """O motivo de existir: mesmo orçamento de excedências em qualquer janela."""
    rng = np.random.default_rng(0)
    grande = rng.lognormal(size=100_000)
    pequena = grande[-10_000:]
    lim = lambda v: automl.WalkForwardEvaluator._limite(
        np.sort(v)[::-1], v.size, 30, "k_maiores")
    excede_grande = (grande > lim(grande)).sum()
    excede_pequena = (pequena > lim(pequena)).sum()
    assert excede_grande == excede_pequena == 29
