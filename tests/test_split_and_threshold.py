import unittest
import numpy as np
import pandas as pd

from src.cnn1d_ae.sequences import train_val_split
from src.cnn1d_ae.scoring import compute_threshold, apply_load_gate


class TestSplitAndThreshold(unittest.TestCase):
    def test_temporal_split_keeps_order(self):
        x = np.arange(100).reshape(20, 5)
        x_train, x_val = train_val_split(x, val_frac=0.2, shuffle=True, seed=42, split_mode="temporal")
        self.assertEqual(x_train.shape[0], 16)
        self.assertEqual(x_val.shape[0], 4)
        self.assertTrue(np.array_equal(x_train[-1], x[15]))
        self.assertTrue(np.array_equal(x_val[0], x[16]))

    def test_random_split_works(self):
        x = np.arange(100).reshape(20, 5)
        x_train, x_val = train_val_split(x, val_frac=0.2, shuffle=True, seed=42, split_mode="random")
        self.assertEqual(x_train.shape[0], 16)
        self.assertEqual(x_val.shape[0], 4)

    def test_threshold_modes(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(compute_threshold(arr, "p95"), np.percentile(arr, 95))
        self.assertAlmostEqual(compute_threshold(arr, "p97"), np.percentile(arr, 97))
        self.assertAlmostEqual(compute_threshold(arr, "p99_5"), np.percentile(arr, 99.5))
        self.assertAlmostEqual(compute_threshold(arr, "target_rate", target_rate=0.2), np.quantile(arr, 0.8))

    def test_load_gate_blocks_only_during_ramp(self):
        idx = pd.date_range("2025-01-01", periods=60, freq="1min")
        # estavel (500) -> rampa forte (500->2000 em 20min) -> estavel (2000)
        values = np.concatenate([
            np.full(20, 500.0),
            np.linspace(500.0, 2000.0, 20),
            np.full(20, 2000.0),
        ])
        load_series = pd.Series(values, index=idx)

        df_point = pd.DataFrame({"is_anom_point": np.ones(60, dtype=int)}, index=idx)
        out = apply_load_gate(
            df_point, load_series,
            ramp_max=500.0,               # unidades/hora: rampa real ~= 75*60=4500/h
            ramp_halflife_minutes=0.1,     # quase sem suavizacao, pra separar os trechos
            window_minutes=1.0,
        )

        # trechos estaveis (inicio e fim) nao devem ser bloqueados
        self.assertFalse(out["load_gate_blocked"].iloc[:15].any())
        self.assertFalse(out["load_gate_blocked"].iloc[45:].any())
        # o trecho em rampa deve ter pontos bloqueados
        self.assertTrue(out["load_gate_blocked"].iloc[22:38].any())
        self.assertEqual(out.loc[out["load_gate_blocked"], "is_anom_point"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
