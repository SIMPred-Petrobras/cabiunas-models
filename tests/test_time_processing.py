import unittest
import pandas as pd

from src.cnn1d_ae.config import PipelineConfig
from src.cnn1d_ae.io import _parse_datetime_smart, _to_utc_indexed_series, build_time_integrity_report


class TestTimeProcessing(unittest.TestCase):
    def test_parse_datetime_smart_mixed_formats(self):
        s_iso = pd.Series(["2025-01-10 03:00:00", "2025-01-10 04:00:00"])
        s_br = pd.Series(["10/01/2025 03:00:00", "11/01/2025 04:00:00"])

        dt_iso = _parse_datetime_smart(s_iso)
        dt_br = _parse_datetime_smart(s_br)

        self.assertTrue(dt_iso.notna().all())
        self.assertTrue(dt_br.notna().all())
        self.assertEqual(dt_br.iloc[0].day, 10)
        self.assertEqual(dt_br.iloc[0].month, 1)

    def test_utc_conversion_with_minus3_shift(self):
        cfg = PipelineConfig(
            SOURCE_TZ="America/Sao_Paulo",
            TARGET_TZ="UTC",
            APPLY_HOUR_SHIFT=True,
            SHIFT_HOURS=-3,
        )
        raw = pd.Series(pd.to_datetime(["2025-01-10 00:00:00"]))
        out = _to_utc_indexed_series(
            raw,
            source_tz=cfg.SOURCE_TZ,
            target_tz=cfg.TARGET_TZ,
            apply_hour_shift=cfg.APPLY_HOUR_SHIFT,
            shift_hours=cfg.SHIFT_HOURS,
        )
        # 2025-01-10 00:00:00 America/Sao_Paulo -> 2025-01-10 03:00:00 UTC, com -3h -> 00:00 UTC
        self.assertEqual(str(out.iloc[0]), "2025-01-10 00:00:00+00:00")

    def test_time_integrity_report_detects_duplicates(self):
        df = pd.DataFrame(
            {
                "data_datetime": pd.to_datetime(
                    ["2025-01-01 00:00:00", "2025-01-01 00:00:00", "2025-01-01 00:01:00"]
                )
            }
        )
        rep = build_time_integrity_report(df, "data_datetime", "raw")
        self.assertEqual(rep["duplicate_timestamps"], 1)
        self.assertTrue(rep["is_monotonic_increasing"])


if __name__ == "__main__":
    unittest.main()
