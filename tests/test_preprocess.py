import unittest

import numpy as np
import pandas as pd

from src.cnn1d_ae.config import PipelineConfig
from src.cnn1d_ae.preprocess import (
    detect_and_mask_sentinels,
    hampel_filter,
    build_stable_gradient_mask,
    build_startup_exclusion_mask,
    normalize_train_only,
    clip_outliers,
)


def _temp_series(values, freq="30s"):
    idx = pd.date_range("2025-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


class TestDetectAndMaskSentinels(unittest.TestCase):
    def test_values_below_low_become_nan(self):
        s = _temp_series([600.0, -40.5, 650.0, 660.0])
        result, n = detect_and_mask_sentinels(s, low=-10.0, high=None)
        self.assertEqual(n, 1)
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertFalse(np.isnan(result.iloc[0]))

    def test_values_above_high_become_nan(self):
        s = _temp_series([600.0, 950.0, 650.0])
        result, n = detect_and_mask_sentinels(s, low=None, high=900.0)
        self.assertEqual(n, 1)
        self.assertTrue(np.isnan(result.iloc[1]))

    def test_both_bounds_applied_simultaneously(self):
        s = _temp_series([-50.0, 600.0, 950.0, 700.0])
        result, n = detect_and_mask_sentinels(s, low=-10.0, high=900.0)
        self.assertEqual(n, 2)
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertTrue(np.isnan(result.iloc[2]))
        self.assertFalse(np.isnan(result.iloc[1]))

    def test_none_bounds_leave_series_unchanged(self):
        s = _temp_series([1.0, 2.0, 3.0])
        result, n = detect_and_mask_sentinels(s, low=None, high=None)
        self.assertEqual(n, 0)
        pd.testing.assert_series_equal(result, s)

    def test_does_not_modify_original_series(self):
        s = _temp_series([-50.0, 600.0])
        original = s.copy()
        detect_and_mask_sentinels(s, low=-10.0, high=None)
        pd.testing.assert_series_equal(s, original)


class TestHampelFilter(unittest.TestCase):
    def test_isolated_spike_is_replaced(self):
        values = [600.0] * 20 + [1500.0] + [600.0] * 20
        s = _temp_series(values)
        result, n = hampel_filter(s, window=5, sigma=3.0)
        self.assertGreater(n, 0)
        # O spike no meio deve ser substituído por algo próximo de 600
        self.assertLess(abs(result.iloc[20] - 600.0), 10.0)

    def test_sustained_ramp_is_not_removed(self):
        # Rampa de startup: 30 pontos subindo de 32 a 670 — não deve ser filtrada
        ramp = list(np.linspace(32, 670, 30))
        values = [670.0] * 10 + ramp + [670.0] * 10
        s = _temp_series(values)
        result, n = hampel_filter(s, window=5, sigma=3.0)
        # A rampa tem gradiente sustentado; spikes removidos devem ser poucos ou zero
        ramp_region = result.iloc[10:40]
        self.assertTrue(ramp_region.min() > 30.0)  # região da rampa não zerada

    def test_flat_signal_produces_no_spikes(self):
        s = _temp_series([500.0] * 100)
        result, n = hampel_filter(s, window=5, sigma=3.0)
        self.assertEqual(n, 0)
        pd.testing.assert_series_equal(result, s)

    def test_output_has_no_nans(self):
        values = [600.0] * 10 + [np.nan] * 3 + [600.0] * 10
        s = _temp_series(values)
        s_filled = s.ffill().bfill()
        result, _ = hampel_filter(s_filled, window=5, sigma=3.0)
        self.assertFalse(result.isna().any())

    def test_output_length_unchanged(self):
        s = _temp_series(list(range(50)))
        result, _ = hampel_filter(s, window=5, sigma=3.0)
        self.assertEqual(len(result), len(s))


class TestBuildStableGradientMask(unittest.TestCase):
    def test_stable_region_flagged_true(self):
        # Primeiros 50 pontos estáveis, últimos 10 com gradiente alto
        stable = [600.0] * 50
        noisy = list(np.linspace(600, 700, 10))
        df = pd.DataFrame(
            {"T5": stable + noisy},
            index=pd.date_range("2025-01-01", periods=60, freq="30s"),
        )
        mask = build_stable_gradient_mask(df, "T5", quantile=0.95)
        # Região estável deve ter maioria True
        self.assertGreater(mask.iloc[:50].sum(), 40)

    def test_returns_boolean_series_same_length(self):
        df = pd.DataFrame(
            {"T5": [600.0, 601.0, 602.0, 650.0]},
            index=pd.date_range("2025-01-01", periods=4, freq="30s"),
        )
        mask = build_stable_gradient_mask(df, "T5", quantile=0.8)
        self.assertEqual(len(mask), 4)
        self.assertEqual(mask.dtype, bool)

    def test_quantile_0_returns_all_false_except_zero_gradient(self):
        # Com quantile=0, threshold=0 -> apenas pontos com diff=0 ficam True
        df = pd.DataFrame(
            {"T5": [600.0, 600.0, 601.0, 600.0]},
            index=pd.date_range("2025-01-01", periods=4, freq="30s"),
        )
        mask = build_stable_gradient_mask(df, "T5", quantile=0.0)
        # Ponto 0 tem diff=NaN->0, ponto 1 tem diff=0, pontos 2,3 têm diff!=0
        self.assertTrue(mask.iloc[0])
        self.assertTrue(mask.iloc[1])


class TestBuildStartupExclusionMask(unittest.TestCase):
    def _make_startup_series(self):
        # Turbina off (32°C) por 10 pontos, startup, on (670°C) por 20 pontos
        idx = pd.date_range("2025-01-01", periods=30, freq="30s")
        values = [32.0] * 10 + [670.0] * 20
        return idx, pd.Series(values, index=idx)

    def test_startup_region_excluded(self):
        idx, s = self._make_startup_series()
        mask = build_startup_exclusion_mask(idx, s, off_threshold=50.0, minutes_after_startup=5)
        # Os primeiros pontos após o startup (índice 10 em diante) devem ser True
        self.assertTrue(mask.iloc[10])
        self.assertTrue(mask.iloc[11])

    def test_pre_startup_not_excluded(self):
        idx, s = self._make_startup_series()
        mask = build_startup_exclusion_mask(idx, s, off_threshold=50.0, minutes_after_startup=5)
        # Período OFF não deve ser excluído pela startup mask
        self.assertFalse(mask.iloc[0])
        self.assertFalse(mask.iloc[5])

    def test_zero_minutes_returns_all_false(self):
        idx, s = self._make_startup_series()
        mask = build_startup_exclusion_mask(idx, s, off_threshold=50.0, minutes_after_startup=0)
        self.assertFalse(mask.any())

    def test_mask_length_equals_index_length(self):
        idx, s = self._make_startup_series()
        mask = build_startup_exclusion_mask(idx, s, off_threshold=50.0, minutes_after_startup=5)
        self.assertEqual(len(mask), len(idx))


class TestNormalizeTrainOnlyStable(unittest.TestCase):
    def _make_bimodal_df(self):
        # Simula distribuição bimodal: 200 pts OFF (~32°C) + 800 pts ON (~650°C)
        idx = pd.date_range("2025-01-01", periods=1000, freq="30s")
        values = [32.0] * 200 + [650.0] * 800
        return pd.DataFrame({"T5": values}, index=idx)

    def test_stable_only_centers_on_operating_regime(self):
        cfg = PipelineConfig(NORMALIZE_MODE="zscore", NORMALIZE_ON_STABLE_ONLY=True, TIME_STEPS=10)
        df = self._make_bimodal_df()
        # Máscara estável: apenas pontos ON (gradiente zero nos dois blocos estáveis)
        stable_mask = pd.Series(True, index=df.index)
        # Forçar somente pontos ON como estáveis
        stable_mask.iloc[:200] = False

        _, _, center, scale = normalize_train_only(cfg, df, df, stable_mask=stable_mask)
        # Com somente dados ON, center deve estar próximo de 650 (não ~554 do bimodal)
        self.assertGreater(center["T5"], 600.0)

    def test_fallback_to_full_when_stable_insufficient(self):
        cfg = PipelineConfig(NORMALIZE_MODE="zscore", NORMALIZE_ON_STABLE_ONLY=True, TIME_STEPS=60)
        df = self._make_bimodal_df()
        # Máscara com pouquíssimos pontos True (insuficiente para TIME_STEPS*2=120)
        stable_mask = pd.Series(False, index=df.index)
        stable_mask.iloc[:5] = True

        _, _, center, scale = normalize_train_only(cfg, df, df, stable_mask=stable_mask)
        # Deve cair no fallback com todos os dados
        expected_center = df["T5"].mean()
        self.assertAlmostEqual(center["T5"], expected_center, places=3)

    def test_stable_none_uses_full_df_normal(self):
        cfg = PipelineConfig(NORMALIZE_MODE="zscore", NORMALIZE_ON_STABLE_ONLY=True, TIME_STEPS=10)
        df = self._make_bimodal_df()
        _, _, center, _ = normalize_train_only(cfg, df, df, stable_mask=None)
        self.assertAlmostEqual(center["T5"], df["T5"].mean(), places=3)

    def test_normalize_off_applies_same_params_to_df_all(self):
        cfg = PipelineConfig(NORMALIZE_MODE="zscore", NORMALIZE_ON_STABLE_ONLY=False, TIME_STEPS=10)
        df_normal = self._make_bimodal_df().iloc[:600]
        df_all = self._make_bimodal_df()
        df_normal_z, df_all_z, center, scale = normalize_train_only(cfg, df_normal, df_all)
        # df_all_z deve estar normalizado com os mesmos center/scale de df_normal
        expected_all_z = (df_all["T5"] - center["T5"]) / scale["T5"]
        pd.testing.assert_series_equal(df_all_z["T5"], expected_all_z, check_names=False)


class TestClipOutliersMad(unittest.TestCase):
    def test_mad_clipping_removes_extreme_sentinel(self):
        cfg = PipelineConfig(OUTLIER_MODE="mad", OUTLIER_MAD_K=3.5)
        values = [650.0] * 98 + [-40.5, 950.0]
        idx = pd.date_range("2025-01-01", periods=100, freq="30s")
        df = pd.DataFrame({"T5": values}, index=idx)
        clipped = clip_outliers(df, cfg)
        # -40.5 e 950.0 devem ser clipados para dentro dos limites MAD
        self.assertGreater(clipped["T5"].min(), -40.5)
        self.assertLess(clipped["T5"].max(), 950.0)

    def test_mad_with_k4_more_permissive_than_k35(self):
        cfg_strict = PipelineConfig(OUTLIER_MODE="mad", OUTLIER_MAD_K=3.5)
        cfg_loose = PipelineConfig(OUTLIER_MODE="mad", OUTLIER_MAD_K=4.0)
        values = list(range(100))
        idx = pd.date_range("2025-01-01", periods=100, freq="30s")
        df = pd.DataFrame({"v": [float(v) for v in values]}, index=idx)
        strict = clip_outliers(df, cfg_strict)
        loose = clip_outliers(df, cfg_loose)
        self.assertGreaterEqual(strict["v"].max(), 0)
        self.assertLessEqual(strict["v"].max(), loose["v"].max())


class TestCommonModeRemoval(unittest.TestCase):
    """O resíduo contra a média dos irmãos só é válido se a sentinela sair ANTES
    da subtração: um termopar aberto (-40.5) desloca a média em ~140°C e a
    pontuação não aplica clipping, então o resíduo envenenado vira falso positivo."""

    GRUPO = [f"TC382_0{i}_A" for i in range(1, 7)]

    def _frame(self, n=40):
        idx = pd.date_range("2025-01-01", periods=n, freq="30s", tz="UTC")
        df = pd.DataFrame({"data_datetime": idx})
        for i, c in enumerate(self.GRUPO):
            df[c] = 700.0 + i          # irmãos separados por 1°C
        df["RUNNING_A"] = 1.0
        return df

    def _cfg(self, **kw):
        return PipelineConfig(ENABLE_COMMON_MODE_REMOVAL=True,
                              COMMON_MODE_GROUP=list(self.GRUPO),
                              SENTINEL_LOW=500.0, SENTINEL_HIGH=950.0,
                              COMMON_MODE_VALID_LOW=-30.0, COMMON_MODE_VALID_HIGH=1200.0,
                              SENTINEL_MODE="none", **kw)

    def _residuo(self, df, cfg, sensor="TC382_03_A"):
        return self._residuo_e_mascara(df, cfg, sensor)[0]

    def _residuo_e_mascara(self, df, cfg, sensor="TC382_03_A"):
        """O pipeline preenche todo NaN restante por interpolação temporal, então
        buraco não sobrevive no valor — quem marca o trecho como não-confiável, e
        o tira do treino via EXCLUDE_LONG_GAPS_FROM_TRAIN, é o long_gap_mask."""
        from src.cnn1d_ae.preprocess import build_sensor_dataframe
        out, long_gap = build_sensor_dataframe(cfg, df, df, sensor)
        return pd.to_numeric(out[sensor], errors="coerce"), long_gap

    def test_residuo_sem_sentinela_bate_a_conta(self):
        df = self._frame()
        r = self._residuo(df, self._cfg())
        # TC382_03_A = 702; irmãos = 700,701,703,704,705 -> média 702.6
        self.assertAlmostEqual(float(r.dropna().iloc[0]), 702.0 - 702.6, places=6)

    def test_irmao_em_sentinela_nao_envenena_o_residuo(self):
        df = self._frame()
        df.loc[5, "TC382_01_A"] = -40.5          # termopar aberto num irmão
        r = self._residuo(df, self._cfg())
        # sem o descarte, a média cairia ~148°C e o resíduo saltaria para ~+148
        self.assertLess(abs(float(r.iloc[5])), 5.0)

    def test_leitura_de_maquina_parada_nao_e_anulada(self):
        """~23-33 C com a turbina parada e leitura legitima. Usar SENTINEL_LOW=500
        como criterio de validade anularia 34% da serie do TC382 em vez de 2%."""
        df = self._frame()
        for i, c in enumerate(self.GRUPO):
            df.loc[9:19, c] = 28.0 + i          # turbina parada, todos frios
        _, gap = self._residuo_e_mascara(df, self._cfg(INTERPOLATE_LIMIT=3))
        self.assertFalse(bool(gap.iloc[14]))

    def test_sentinela_isolada_no_proprio_sensor_nao_envenena(self):
        # buraco curto: a interpolação (INTERPOLATE_LIMIT) preenche, e deve
        # preencher com um valor são — não com o -40.5 menos a média dos irmãos.
        df = self._frame()
        df.loc[9, "TC382_03_A"] = -40.5
        r = self._residuo(df, self._cfg())
        self.assertLess(abs(float(r.iloc[9])), 5.0)

    def test_sentinela_longa_no_proprio_sensor_marca_gap(self):
        # buraco maior que INTERPOLATE_LIMIT: o valor é preenchido pelo fallback
        # temporal, mas o trecho tem de ficar marcado para sair do treino.
        df = self._frame()
        df.loc[9:19, "TC382_03_A"] = -40.5
        r, gap = self._residuo_e_mascara(df, self._cfg(INTERPOLATE_LIMIT=3))
        self.assertTrue(bool(gap.iloc[14]))
        self.assertLess(abs(float(r.iloc[14])), 5.0)   # e o valor não fica envenenado

    def test_poucos_irmaos_validos_marcam_gap(self):
        df = self._frame()
        for c in ["TC382_01_A", "TC382_02_A", "TC382_04_A"]:
            df.loc[9:19, c] = -40.5              # sobram 2 irmãos válidos
        _, gap = self._residuo_e_mascara(
            df, self._cfg(COMMON_MODE_MIN_SIBLINGS=3, INTERPOLATE_LIMIT=3))
        self.assertTrue(bool(gap.iloc[14]))

    def test_com_dois_irmaos_exigidos_o_mesmo_trecho_e_aceito(self):
        # o limiar é COMMON_MODE_MIN_SIBLINGS, não uma regra fixa
        df = self._frame()
        for c in ["TC382_01_A", "TC382_02_A", "TC382_04_A"]:
            df.loc[9:19, c] = -40.5
        _, gap = self._residuo_e_mascara(
            df, self._cfg(COMMON_MODE_MIN_SIBLINGS=2, INTERPOLATE_LIMIT=3))
        self.assertFalse(bool(gap.iloc[14]))


if __name__ == "__main__":
    unittest.main()
