"""Testes de invariante do pré-processamento — codificam contratos que NÃO podem
regredir (foram fonte de bugs reais): normalização robusta, normalização não clipa
anomalia fora-de-faixa, e gaps longos são detectados (base da exclusão no scoring)."""
import unittest

import numpy as np
import pandas as pd

from src.cnn1d_ae.config import PipelineConfig
from src.cnn1d_ae.preprocess import normalize_train_only, build_sensor_dataframe


def _idx(n, freq="30s"):
    return pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")


class TestNormalizeInvariants(unittest.TestCase):
    def test_robust_usa_mediana_e_iqr(self):
        cfg = PipelineConfig(NORMALIZE_MODE="robust")
        s = pd.Series([10, 12, 14, 16, 18, 1000.0])  # outlier não move mediana/IQR muito
        df = pd.DataFrame({"x": s})
        _, _, center, scale = normalize_train_only(cfg, df, df)
        self.assertAlmostEqual(float(center["x"]), float(s.median()))
        self.assertAlmostEqual(float(scale["x"]), float(s.quantile(0.75) - s.quantile(0.25)))

    def test_normalizacao_NAO_clipa_anomalia_fora_de_faixa(self):
        # invariante crítico: normalizar não pode comprimir o valor anômalo — ele
        # precisa virar um z grande para o AE detectar (bug do clip que zerou recall).
        cfg = PipelineConfig(NORMALIZE_MODE="zscore")
        train = pd.DataFrame({"x": np.full(50, 100.0) + np.random.normal(0, 1, 50)})
        alld = pd.DataFrame({"x": [100.0, 100.0, 400.0]})  # 400 = anomalia fora-de-faixa
        _, df_all_z, center, scale = normalize_train_only(cfg, train, alld)
        z_anom = float(df_all_z["x"].iloc[-1])
        # 400 fica MUITO acima do normal (~0); jamais clipado para perto de 0
        self.assertGreater(z_anom, 50.0)


class TestLongGapDetection(unittest.TestCase):
    def test_long_gap_mask_marca_regiao_interpolada(self):
        # gap de 20 pontos NaN >> INTERPOLATE_LIMIT=3 → região marcada como long gap
        n = 60
        vals = np.full(n, 100.0)
        vals[20:40] = np.nan  # buraco longo
        df_raw = pd.DataFrame({"data_datetime": _idx(n), "TC1": vals})
        cfg = PipelineConfig(
            TRAIN_SOURCE="raw", TIME_COL="data_datetime", INTERPOLATE_LIMIT=3,
            SENTINEL_MODE="none", ENABLE_CONTEXT_FEATURES=False,
        )
        df_use, long_gap_mask = build_sensor_dataframe(cfg, pd.DataFrame(), df_raw, "TC1")
        lg = long_gap_mask.reindex(df_use.index).fillna(False)
        # o miolo do gap deve estar marcado (dado fabricado pela interpolação)
        self.assertTrue(bool(lg.iloc[28]), "miolo do gap longo deveria ser marcado")
        # pontos com dado real (fora do gap) não marcados
        self.assertFalse(bool(lg.iloc[5]), "ponto com dado real não deveria ser marcado")


if __name__ == "__main__":
    unittest.main()
