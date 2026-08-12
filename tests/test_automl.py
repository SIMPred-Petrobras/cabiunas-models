import unittest
import numpy as np
import pandas as pd

from src.cnn1d_ae.scoring import (
    compute_composite_score,
    compute_normal_alert_rate,
    map_seq_to_point_anomalies,
)
from src.cnn1d_ae.automl_models import (
    build_dense_autoencoder,
    dense_reconstruction_error,
    fit_ocsvm,
    ocsvm_error,
    fit_isolation_forest,
    isolation_forest_error,
)


class TestCompositeScore(unittest.TestCase):
    def test_perfect_detection_no_fp_scores_high(self):
        r = compute_composite_score(detection_rate=1.0, normal_alert_rate=0.0, fp_penalty=2.0)
        self.assertAlmostEqual(r["composite_score"], 1.0)

    def test_fp_penalizes_quadratically(self):
        low_fp = compute_composite_score(detection_rate=0.5, normal_alert_rate=0.05, fp_penalty=2.0)
        high_fp = compute_composite_score(detection_rate=0.5, normal_alert_rate=0.20, fp_penalty=2.0)
        self.assertGreater(low_fp["composite_score"], high_fp["composite_score"])

    def test_min_detection_rate_penalty_applies_below_floor(self):
        below = compute_composite_score(detection_rate=0.1, normal_alert_rate=0.0, min_detection_rate=0.3)
        above = compute_composite_score(detection_rate=0.3, normal_alert_rate=0.0, min_detection_rate=0.3)
        self.assertLess(below["balanced_score"], above["balanced_score"])


class TestNormalAlertRate(unittest.TestCase):
    def test_excludes_near_alarm_and_off_points(self):
        idx = pd.date_range("2025-01-01", periods=10, freq="1h")
        df_point = pd.DataFrame({
            "is_anom_point": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            "operational_state": ["on"] * 8 + ["off_longo"] * 2,
        }, index=idx)
        near_alarm = pd.Series(False, index=idx)
        near_alarm.iloc[0] = True  # o unico anomalo fora do "off" mas perto de alarme

        rate = compute_normal_alert_rate(df_point, near_alarm)
        # pontos elegiveis (on, longe de alarme): indices 1-7 -> so o indice 3 e 7 tem anomalia = 2/7
        self.assertAlmostEqual(rate, 2 / 7)


class TestDebounceViaPointMapping(unittest.TestCase):
    def test_time_steps_one_is_pure_pointwise_debounce(self):
        idx = pd.date_range("2025-01-01", periods=6, freq="1min")
        flags = np.array([1, 1, 1, 0, 1, 1])
        out = map_seq_to_point_anomalies(
            flags, idx, time_steps=1, point_rule="all_of_window", point_window=3, point_min_count=3,
        )
        # so a posicao 2 tem os 3 ultimos pontos (0,1,2) todos =1
        self.assertListEqual(out["is_anom_point"].tolist(), [0, 0, 1, 0, 0, 0])


class TestAutomlModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x_normal = rng.normal(size=(200, 3)).astype(np.float32)
        self.x_all = rng.normal(size=(50, 3)).astype(np.float32)

    def test_dense_autoencoder_roundtrip_shapes(self):
        model = build_dense_autoencoder(n_features=3, layer_sizes=[4, 2], dropout=0.0, lr=1e-3)
        model.fit(self.x_normal, self.x_normal, epochs=1, batch_size=32, verbose=0)
        err = dense_reconstruction_error(model, self.x_all, batch_size=32)
        self.assertEqual(err.shape, (50,))
        self.assertTrue(np.all(err >= 0))

    def test_ocsvm_scores_are_finite(self):
        clf = fit_ocsvm(self.x_normal, nu=0.05, gamma="scale")
        err = ocsvm_error(clf, self.x_all)
        self.assertEqual(err.shape, (50,))
        self.assertTrue(np.all(np.isfinite(err)))

    def test_isolation_forest_scores_are_finite(self):
        model = fit_isolation_forest(self.x_normal, contamination=0.05, n_estimators=20, random_state=42)
        err = isolation_forest_error(model, self.x_all)
        self.assertEqual(err.shape, (50,))
        self.assertTrue(np.all(np.isfinite(err)))


if __name__ == "__main__":
    unittest.main()
