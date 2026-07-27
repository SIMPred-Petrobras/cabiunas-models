"""Teste do teto de duty-cycle em best_point_for_sensor: a calibração não pode mais
escolher o piso permissivo (q=0.5) quando há um teto de tempo-em-alerta — foi o que
deixou o alarme ligado 72-93% do tempo em produção."""
import unittest

import numpy as np
import pandas as pd

from scripts.eval_per_sensor_level import best_point_for_sensor


def _health(n=2000, freq="5min"):
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    # health = rank uniforme em [0,1] (como o ewma_quantile de produção)
    vals = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(0)
    return pd.Series(vals[rng.permutation(n)], index=idx)


class TestDutyCycleCeiling(unittest.TestCase):
    def setUp(self):
        self.health = _health()
        # incidentes nos pontos de health mais alto (detectáveis em qualquer q razoável)
        top = self.health.sort_values().index[-5:]
        self.inc = sorted(top.tolist())

    def test_sem_teto_escolhe_piso(self):
        # max_sticky_duty=1.0 desliga o teto novo — este teste é sobre
        # max_duty_cycle especificamente, não sobre max_sticky_duty.
        r = best_point_for_sensor(self.health, self.inc, horizon_hours=8.0,
                                  sticky_hours=0.0, fa_budget=10.0, n_thresholds=50,
                                  max_sticky_duty=1.0)
        # sem teto, a busca tende ao piso (q baixo) com duty alto
        self.assertLessEqual(r["threshold_q"], 0.6)
        self.assertGreater(r["duty_cycle"], 0.35)

    def test_com_teto_respeita_duty(self):
        r = best_point_for_sensor(self.health, self.inc, horizon_hours=8.0,
                                  sticky_hours=0.0, fa_budget=10.0, n_thresholds=50,
                                  max_duty_cycle=0.2, max_sticky_duty=1.0)
        # com teto 0.2, o ponto escolhido tem duty <= 0.2 e q mais alto
        self.assertLessEqual(r["duty_cycle"], 0.2 + 1e-9)
        self.assertGreater(r["threshold_q"], 0.6)

    def test_default_backward_compativel(self):
        # default max_duty_cycle=1.0 não filtra nada (chave duty_cycle presente)
        r = best_point_for_sensor(self.health, self.inc, horizon_hours=8.0, n_thresholds=20,
                                  max_sticky_duty=1.0)
        self.assertIn("duty_cycle", r)


class TestStickyDutyCeiling(unittest.TestCase):
    """max_sticky_duty limita o duty PÓS-sticky, que pode ser bem maior que o
    duty bruto pré-sticky quando sticky_hours é grande — é essa lacuna que o
    teto de duty_cycle (pré-sticky) sozinho não cobre."""
    def setUp(self):
        self.health = _health()
        top = self.health.sort_values().index[-5:]
        self.inc = sorted(top.tolist())

    def test_teto_sticky_rejeita_threshold_com_duty_bruto_baixo(self):
        # sticky_hours alto infla duty_sticky bem acima do duty bruto; um teto
        # de max_duty_cycle frouxo (bruto) não pega isso, só max_sticky_duty pega.
        r = best_point_for_sensor(self.health, self.inc, horizon_hours=8.0,
                                  sticky_hours=12.0, fa_budget=10.0, n_thresholds=50,
                                  max_duty_cycle=1.0, max_sticky_duty=0.25)
        self.assertIn("duty_sticky", r)
        self.assertLessEqual(r["duty_sticky"], 0.25 + 1e-9)

    def test_recall_raw_nunca_maior_que_recall_com_sticky(self):
        # apply_sticky só ESTENDE o alerta bruto (nunca remove pontos), então o
        # alerta final é superset do cruzamento bruto — logo recall_raw <= recall
        # sempre. Se isso inverter é sinal de alerta "emprestando" cobertura da
        # cauda do sticky de um incidente anterior sem que o dado bruto sustente.
        r = best_point_for_sensor(self.health, self.inc, horizon_hours=8.0,
                                  sticky_hours=12.0, fa_budget=10.0, n_thresholds=50,
                                  max_sticky_duty=1.0)
        self.assertIn("recall_raw", r)
        self.assertLessEqual(r["recall_raw"], r["recall"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
