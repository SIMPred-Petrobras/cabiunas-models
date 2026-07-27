"""Regressão para detect_fabricated_gap_mask: até agora só validado contra dado
real (gap de 8h10 em TC382_03_A, 02/10/2025 — ver memória
tc382-03a-interpolacao-gap-janela6), sem guarda de teste sintético em tests/."""
import unittest

import numpy as np
import pandas as pd

from src.cnn1d_ae.preprocess import detect_fabricated_gap_mask

DT_SECONDS = 30.0
MIN_RUN_MINUTES = 30.0
CURVATURE_FRAC = 0.05


def _noisy_series(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 20.0 + 0.5 * np.sin(2 * np.pi * t / 500.0)
    noise = rng.normal(0.0, 0.3, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="30s", tz="UTC")
    return pd.Series(base + noise, index=idx)


def _insert_linear_ramp(series: pd.Series, start: int, n_points: int) -> pd.Series:
    v = series.to_numpy(dtype=float).copy()
    ramp = np.linspace(v[start - 1], v[start - 1] - 1.0, n_points + 1)[1:]
    v[start:start + n_points] = ramp
    return pd.Series(v, index=series.index)


class TestFabricatedGapMask(unittest.TestCase):
    def test_rampa_linear_sustentada_e_flagada(self):
        # 80 pontos a 30s = 40 min, acima do min_run_minutes=30 — deve ser flagada
        series = _insert_linear_ramp(_noisy_series(), start=1500, n_points=80)
        mask = detect_fabricated_gap_mask(series, DT_SECONDS, MIN_RUN_MINUTES, CURVATURE_FRAC)
        # miolo do trecho linear flagado (ignora bordas por causa do alinhamento de diff2)
        self.assertTrue(mask.iloc[1510:1570].all())
        # fora do trecho, nada flagado
        self.assertFalse(mask.iloc[:1490].any())
        self.assertFalse(mask.iloc[1590:].any())

    def test_ruido_normal_sem_falso_positivo(self):
        series = _noisy_series()
        mask = detect_fabricated_gap_mask(series, DT_SECONDS, MIN_RUN_MINUTES, CURVATURE_FRAC)
        self.assertFalse(mask.any())

    def test_rampa_curta_abaixo_do_minimo_nao_e_flagada(self):
        # 40 pontos a 30s = 20 min, abaixo do min_run_minutes=30 — não deve ser flagada
        series = _insert_linear_ramp(_noisy_series(), start=1500, n_points=40)
        mask = detect_fabricated_gap_mask(series, DT_SECONDS, MIN_RUN_MINUTES, CURVATURE_FRAC)
        self.assertFalse(mask.any())

    def test_serie_curta_nao_quebra(self):
        idx = pd.date_range("2025-01-01", periods=2, freq="30s", tz="UTC")
        series = pd.Series([1.0, 2.0], index=idx)
        mask = detect_fabricated_gap_mask(series, DT_SECONDS, MIN_RUN_MINUTES, CURVATURE_FRAC)
        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
