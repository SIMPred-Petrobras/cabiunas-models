"""Invariante visual: período com equipamento DESLIGADO (off) não deve aparecer
nos plots de série — nem linha, nem ponto de anomalia, nem marcador de alarme.
Codifica a decisão de esconder OFF (alarmes em OFF são fora de escopo do modelo)."""
import unittest

import numpy as np
import pandas as pd

from src.cnn1d_ae.plots import _on_mask, _filter_idx_on


def _idx(n, freq="5min"):
    return pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")


class TestOnMask(unittest.TestCase):
    def setUp(self):
        self.idx = _idx(30)
        # estados: 0-9 on, 10-19 off_longo, 20-29 on
        st = ["on"] * 10 + ["off_longo"] * 10 + ["on"] * 10
        self.state = pd.Series(st, index=self.idx)

    def test_on_mask_marca_off(self):
        is_on = _on_mask(self.idx, self.state)
        self.assertTrue(bool(is_on.iloc[5]))      # ON
        self.assertFalse(bool(is_on.iloc[15]))    # OFF
        self.assertTrue(bool(is_on.iloc[25]))     # ON
        self.assertEqual(int((~is_on).sum()), 10)

    def test_sem_state_assume_tudo_ligado(self):
        is_on = _on_mask(self.idx, None)
        self.assertTrue(bool(is_on.all()))

    def test_serie_no_off_vira_nan(self):
        s = pd.Series(np.arange(30, dtype=float), index=self.idx)
        is_on = _on_mask(self.idx, self.state)
        s_on = s.where(is_on)
        self.assertTrue(bool(np.isnan(s_on.iloc[15])))   # OFF apagado
        self.assertFalse(bool(np.isnan(s_on.iloc[5])))   # ON preservado

    def test_filtra_alarmes_em_off(self):
        # alarme em ON (idx 3) mantido; alarme em OFF (idx 15) removido
        alarms = pd.DatetimeIndex([self.idx[3], self.idx[15]])
        is_on = _on_mask(self.idx, self.state)
        kept = _filter_idx_on(alarms, is_on)
        self.assertIn(self.idx[3], kept)
        self.assertNotIn(self.idx[15], kept)


if __name__ == "__main__":
    unittest.main()
