"""Graduação de confiança: o episódio que não se sustenta é rebaixado a `observação`.

Calibrado em `scripts/eval_confidence_grading_offline.py` sobre 71 FP e 29 TP do braço
b2024 (W=6h, dens≥0,70, incl≥−0,030 → 65% dos FP rebaixados, 83% dos TP mantidos).
Aqui testamos a LÓGICA em episódios sintéticos — a verificação contra os 46/71 · 24/29
reais é o próprio script offline, que agora chama esta mesma função.
"""
import unittest

import numpy as np
import pandas as pd

from src.cnn1d_ae.inference import grade_episodes, health_to_reference_rank

W, DENS_MIN, INCL_MIN = 6.0, 0.70, -0.030
THR = 0.886


def _ep(values, freq="5min", start="2025-03-01"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def _grade(h):
    return grade_episodes(h, [(h.index[0], h.index[-1])], threshold=THR,
                          window_hours=W, dens_min=DENS_MIN, incl_min=INCL_MIN)


class TestGradeEpisodes(unittest.TestCase):
    def test_episodio_sustentado_vira_acao(self):
        # 12h acima do limiar, sem decair: o perfil do evento real
        g = _grade(_ep(np.full(144, 0.95)))
        self.assertEqual(g.loc[0, "nivel"], "acao")
        self.assertAlmostEqual(g.loc[0, "densidade"], 1.0)

    def test_episodio_que_decai_vira_observacao(self):
        # pico alto e queda rápida: o perfil do falso positivo. O pico NÃO separa
        # (AUC 0,44) — quem separa é a inclinação.
        g = _grade(_ep(np.linspace(0.99, 0.40, 144)))
        self.assertEqual(g.loc[0, "nivel"], "observacao")
        self.assertLess(g.loc[0, "inclinacao"], INCL_MIN)

    def test_densidade_baixa_vira_observacao_mesmo_sem_decair(self):
        # oscila em torno do limiar sem tendência: densidade 0,5 < 0,70
        v = np.where(np.arange(144) % 2 == 0, 0.95, 0.50)
        g = _grade(_ep(v))
        self.assertEqual(g.loc[0, "nivel"], "observacao")
        # janela de 6h a 5min é inclusiva nas duas pontas: 73 pontos, 37 acima
        self.assertAlmostEqual(g.loc[0, "densidade"], 37 / 73)

    def test_episodio_curto_demais_fica_em_acao(self):
        # menos de 3 pontos: não dá para medir. Não se rebaixa o que não se avaliou —
        # o custo relevante é TP rebaixado, não FP mantido.
        g = _grade(_ep([0.99, 0.99]))
        self.assertEqual(g.loc[0, "nivel"], "acao")
        self.assertFalse(bool(g.loc[0, "medido"]))
        self.assertTrue(np.isnan(g.loc[0, "densidade"]))

    def test_so_a_janela_W_conta_nao_o_episodio_inteiro(self):
        # sustentado nas 6h de decisão, desaba depois: tem de virar `acao`, senão a
        # decisão dependeria do futuro e não seria tomável em onset+W.
        v = np.concatenate([np.full(72, 0.95), np.full(200, 0.10)])
        g = _grade(_ep(v))
        self.assertEqual(g.loc[0, "nivel"], "acao")

    def test_lead_preservado_o_onset_nao_se_move(self):
        h = _ep(np.linspace(0.99, 0.40, 144))
        g = _grade(h)
        self.assertEqual(g.loc[0, "onset"], h.index[0])


class TestReferenceRank(unittest.TestCase):
    def test_rank_reproduz_o_quantil_da_calibracao(self):
        cal = pd.Series(np.linspace(0.0, 10.0, 1001))
        probs = np.linspace(0.0, 1.0, 201)
        rq = [float(cal.quantile(p)) for p in probs]
        got = health_to_reference_rank([0.0, 2.5, 5.0, 10.0], rq)
        np.testing.assert_allclose(got, [0.0, 0.25, 0.5, 1.0], atol=1e-3)

    def test_valores_fora_da_faixa_de_calibracao_saturam(self):
        rq = [float(v) for v in np.linspace(0.0, 1.0, 201)]
        got = health_to_reference_rank([-5.0, 99.0], rq)
        np.testing.assert_allclose(got, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
