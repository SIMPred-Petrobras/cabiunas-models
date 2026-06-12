import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.cnn1d_ae.inference import load_bundle, transform_features, score_dataframe, score_production


class _ZeroModel:
    """Stub: reconstrói tudo como zero → mae da sequência = média(|entrada normalizada|).
    Evita dependência de Keras/GPU mantendo o cálculo de mae determinístico."""

    def predict_on_batch(self, xb):
        return np.zeros_like(xb)


def _bundle(**over):
    b = {
        "sensor": "TC382_03_A",
        "feature_columns": ["TC382_03_A"],
        "n_features": 1,
        "time_steps": 3,
        "stride": 1,
        "normalize_mode": "zscore",
        "center": {"TC382_03_A": 10.0},
        "scale": {"TC382_03_A": 2.0},
        "outlier_mode": "mad",
        "clip_bounds": {"TC382_03_A": [0.0, 20.0]},
        "threshold": 1.0,
        "thresh_mode": "p99",
        "monthly_thresholds": {},
        "running_col": "NGP_A",
        "running_threshold": 50.0,
        "predictive_ewma_half_life_hours": 4.0,
        "alarm_policy": "threshold",
        "point_rule": "k_of_window",
        "point_window": 60,
        "point_min_count": 3,
    }
    b.update(over)
    return b


def _df(vals, ngp=None):
    idx = pd.date_range("2025-01-01", periods=len(vals), freq="5min")
    data = {"TC382_03_A": vals}
    if ngp is not None:
        data["NGP_A"] = ngp
    return pd.DataFrame(data, index=idx)


class TestTransformFeatures(unittest.TestCase):
    def test_clip_then_normalize_with_train_stats(self):
        # 100 -> clip 20 -> (20-10)/2 = 5 ; -50 -> clip 0 -> (0-10)/2 = -5 ; 12 -> (12-10)/2 = 1
        out = transform_features(_df([100.0, -50.0, 12.0]), _bundle())
        np.testing.assert_allclose(out.ravel(), [5.0, -5.0, 1.0], rtol=0, atol=1e-6)

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            transform_features(_df([1.0]).rename(columns={"TC382_03_A": "x"}), _bundle())

    def test_no_clip_when_bounds_absent(self):
        out = transform_features(_df([100.0]), _bundle(clip_bounds={}))
        self.assertAlmostEqual(float(out.ravel()[0]), (100.0 - 10.0) / 2.0)


class TestScoreDataframe(unittest.TestCase):
    def test_mae_matches_zero_model_and_threshold_flag(self):
        # valores já dentro do clip; normalizado = (v-10)/2
        df = _df([10.0, 10.0, 10.0, 30.0, 10.0])  # 30 clipa em 20 -> norm 5
        b = _bundle(threshold=1.0)
        out = score_dataframe(_ZeroModel(), b, df)
        # 3 janelas (T=3, stride=1): médias de |norm| por janela
        # norm = [0,0,0,5(clip),0]; janelas: [0,0,0]=0 ; [0,0,5]=1.667 ; [0,5,0]=1.667
        np.testing.assert_allclose(out["mae_seq"].to_numpy(), [0.0, 5.0 / 3, 5.0 / 3], atol=1e-6)
        self.assertEqual(out["is_anom_seq"].tolist(), [0, 1, 1])  # >1.0

    def test_operational_mask_suppresses_off_periods(self):
        df = _df([10.0, 10.0, 30.0, 30.0, 30.0], ngp=[90, 90, 90, 0, 0])
        out = score_dataframe(_ZeroModel(), _bundle(threshold=1.0), df)
        # janelas terminam em pos 2,3,4 -> NGP 90(on),0(off),0(off)
        self.assertEqual(out["operational_state"].tolist(), ["on", "off", "off"])
        # mesmo com mae alto, OFF é suprimido
        self.assertEqual(out.loc[out["operational_state"] == "off", "is_anom_seq"].sum(), 0)


class TestScoreProduction(unittest.TestCase):
    def _prod_bundle(self, debounce_hours=0.0):
        b = _bundle()
        b["production_alerting"] = {
            "half_life_hours": 0.5,
            "ewma_abs_threshold": 2.0,
            "sticky_hours": 12.0,
            "debounce_hours": debounce_hours,
        }
        return b

    def test_requires_production_block(self):
        with self.assertRaises(ValueError):
            score_production(_ZeroModel(), _bundle(), _df([10.0, 10.0, 10.0]))

    def test_ewma_abs_threshold_and_columns(self):
        df = _df([10.0] * 10 + [30.0] * 10, ngp=[90] * 20)  # 30 clipa 20 → norm 5
        out = score_production(_ZeroModel(), self._prod_bundle(), df)
        self.assertIn("health_ewma", out.columns)
        self.assertIn("alert", out.columns)
        # baseline 0 (norm 0) não alerta; após o degrau a EWMA cruza 2.0 e alerta
        self.assertEqual(int(out["alert"].iloc[:5].sum()), 0)
        self.assertGreater(int(out["alert"].iloc[-5:].sum()), 0)

    def test_debounce_zeroes_short_runs(self):
        # alerta de 1 ponto isolado deve ser apagado por debounce longo
        df = _df([10.0] * 5 + [30.0] + [10.0] * 14, ngp=[90] * 20)
        out = score_production(_ZeroModel(), self._prod_bundle(debounce_hours=2.0), df)
        self.assertEqual(int(out["alert"].sum()), 0)


class TestBundleRoundTrip(unittest.TestCase):
    def test_load_bundle(self):
        b = _bundle()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "inference_bundle.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(b, f)
            loaded = load_bundle(p)
        self.assertEqual(loaded["threshold"], b["threshold"])
        self.assertEqual(loaded["center"], b["center"])
        self.assertEqual(loaded["clip_bounds"], b["clip_bounds"])


if __name__ == "__main__":
    unittest.main()
