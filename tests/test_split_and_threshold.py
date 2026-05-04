import unittest
import numpy as np

from src.cnn1d_ae.sequences import train_val_split
from src.cnn1d_ae.scoring import compute_threshold


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


if __name__ == "__main__":
    unittest.main()
