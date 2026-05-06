import unittest

import numpy as np
import pandas as pd

from src.cnn1d_ae.config import PipelineConfig
from src.cnn1d_ae.feature_engineering import generate_features


class TestFeatureEngineering(unittest.TestCase):
    def _base_df(self):
        return pd.DataFrame(
            {"T5_AVG_A": [0.0, 0.0, 1.0, 2.0, 4.0, 8.0]},
            index=pd.date_range("2025-04-01 00:00:00", periods=6, freq="30s"),
        )

    def test_rolling_features_keep_same_number_of_rows(self):
        cfg = PipelineConfig(ENABLE_ROLLING_FEATURES=True, ROLLING_WINDOW=3)
        df = self._base_df()

        out = generate_features(df, "T5_AVG_A", cfg)

        self.assertEqual(len(out), len(df))
        self.assertIn("T5_AVG_A_roll_rms", out.columns)
        self.assertIn("T5_AVG_A_delta_2", out.columns)

    def test_rolling_features_do_not_generate_nan(self):
        cfg = PipelineConfig(ENABLE_ROLLING_FEATURES=True, ROLLING_WINDOW=3)

        out = generate_features(self._base_df(), "T5_AVG_A", cfg)

        self.assertFalse(out.isna().any().any())

    def test_same_signal_produces_identical_features_for_normal_and_all(self):
        cfg = PipelineConfig(ENABLE_ROLLING_FEATURES=True, ROLLING_WINDOW=3)
        df_normal = self._base_df()
        df_all = self._base_df()

        out_normal = generate_features(df_normal, "T5_AVG_A", cfg)
        out_all = generate_features(df_all, "T5_AVG_A", cfg)

        pd.testing.assert_frame_equal(out_normal, out_all)

    def test_crest_factor_handles_zero_regions_without_division_by_zero(self):
        cfg = PipelineConfig(ENABLE_ROLLING_FEATURES=True, ROLLING_WINDOW=3)
        df = pd.DataFrame(
            {"T5_AVG_A": [0.0, 0.0, 0.0, 0.0]},
            index=pd.date_range("2025-04-01", periods=4, freq="30s"),
        )

        out = generate_features(df, "T5_AVG_A", cfg)
        crest = out["T5_AVG_A_roll_crest"]

        self.assertTrue(np.isfinite(crest).all())
        self.assertFalse(crest.isna().any())

    def test_all_feature_flags_disabled_returns_dataframe_without_changes(self):
        cfg = PipelineConfig(
            ENABLE_ROLLING_FEATURES=False,
            ENABLE_SPECTRAL_FEATURES=False,
            ENABLE_CONTEXT_FEATURES=False,
        )
        df = self._base_df()

        out = generate_features(df, "T5_AVG_A", cfg)

        pd.testing.assert_frame_equal(out, df)


if __name__ == "__main__":
    unittest.main()
