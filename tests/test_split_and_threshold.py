import unittest
import numpy as np

from src.cnn1d_ae.sequences import train_val_split
import pandas as pd

from src.cnn1d_ae.scoring import compute_threshold, evaluate_alarm_detection, compute_monthly_mae_drift


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

    def test_mean_std_mode(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu, sigma = float(np.mean(arr)), float(np.std(arr))
        self.assertAlmostEqual(compute_threshold(arr, "mean_std", std_mult=3.0), mu + 3.0 * sigma)
        self.assertAlmostEqual(compute_threshold(arr, "mean_std", std_mult=0.5), mu + 0.5 * sigma)
        # y maior => threshold maior (é o botão de sensibilidade que o operador gira)
        self.assertGreater(compute_threshold(arr, "mean_std", std_mult=3.0),
                           compute_threshold(arr, "mean_std", std_mult=1.0))

    def test_mean_std_handles_zero_variance(self):
        arr = np.full(10, 7.0)
        self.assertAlmostEqual(compute_threshold(arr, "mean_std", std_mult=3.0), 7.0, places=6)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            compute_threshold(np.array([1.0, 2.0]), "nao_existe")

    def test_alarm_detection_metrics_include_precision_f1_and_lead_time(self):
        idx = pd.date_range("2025-04-01 00:00:00", periods=12, freq="h")
        df_point = pd.DataFrame({"is_anom_point": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]}, index=idx)
        df_alarm = pd.DataFrame({"Data da Ocorrencia": [idx[2], idx[5]]})

        metrics = evaluate_alarm_detection(df_alarm, df_point, window_minutes=90)

        self.assertEqual(metrics["n_alarms"], 2)
        self.assertEqual(metrics["alarms_with_detected_anomaly_in_window"], 2)
        self.assertEqual(metrics["n_predicted_anomaly_episodes"], 3)
        self.assertEqual(metrics["n_false_positive_anomaly_episodes"], 1)
        self.assertAlmostEqual(metrics["precision_event"], 2 / 3)
        self.assertAlmostEqual(metrics["f1_event"], 0.8)
        self.assertAlmostEqual(metrics["median_lead_time_minutes_before_alarm"], 60.0)

    def test_monthly_mae_drift_returns_monthly_threshold_ratios(self):
        df = pd.DataFrame(
            {
                "seq_start_time": pd.to_datetime(["2025-04-01", "2025-04-02", "2025-05-01"]),
                "mae_seq": [0.1, 0.2, 0.5],
                "is_anom_seq": [0, 0, 1],
            }
        )

        drift = compute_monthly_mae_drift(df, threshold=0.3)

        self.assertEqual(list(drift["month"]), ["2025-04", "2025-05"])
        self.assertTrue(bool(drift.loc[drift["month"] == "2025-05", "drift_flag_p99_above_threshold"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
