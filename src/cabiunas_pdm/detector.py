"""Detector FP-first do TC-330.03A — modelo campeão do backtest 2025/26.

Três sinais sobre a grade 2 min (apenas operação estável):
  S1: PCA-recon família temperaturas (EWMA 1h) > 2×p99, sustentado 30 min
  S2: PCA-recon família pressões/óleo  (idem)
  S3: |z| do spread do mancal TI_0305 − mediana(TI_0301/0303/0307) > 3 (EWMA 30 min)
Níveis: atenção = ≥1 sinal · CONFIRMADO = ≥2 sinais simultâneos.
Blackout após cada partida. Vibração fora da política (quarentena).

A definição de "operação" vem do ``OperabilityResolver``: ``NGP_A`` acima do
limiar calibrado onde o sinal existe e ``RUNNING_A == 1`` como fallback
explícito onde não existe (ver ``operability.py`` e o diagnóstico em
reports/DIAGNOSTICO_OPERABILIDADE.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cleaning, config, scoring
from .operability import OperabilityResolver

SUSTAIN = 15          # 15 × 2 min = 30 min
THR_FAM = 2.0
THR_SPREAD = 3.0
# Blackout pós-partida: 6h (revisado 2026-07-18). Testado 4h/6h/12h no backtest:
# 6h recupera a detecção CONFIRMADA do trip de óleo de 04/11/2025 (8,8h de
# antecedência) SEM aumentar o FP em held-out (~1-2 episódios/mês, igual a 12h).
# 4h detectaria também o trip de mancal, mas custa ~4/mês de FP — ver
# reports/BACKTEST_RESULTADOS.md secção 3b.
BLACKOUT = "6h"
FIT_POINTS = 20_000   # ~28 dias estáveis


def _spread_mancal(X: pd.DataFrame) -> pd.Series:
    irm = X[["954005_624_TI_0301", "954005_624_TI_0303",
             "954005_624_TI_0307"]].median(axis=1)
    return X["954005_624_TI_0305"] - irm


def _sustained(s: pd.Series, thr: float) -> pd.Series:
    return ((s > thr).astype(int)
            .rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)


class Detector:
    """Fit em operação estável do regime vigente; score de qualquer trecho 2 min."""

    def fit(self, df: pd.DataFrame, fit_end: str | None = None) -> "Detector":
        sensors = df[config.SENSOR_TAGS]
        if "stable" in df.columns:
            stable = df["stable"].astype(bool)
        else:
            stable = OperabilityResolver().resolve(df).stable
        self.operability_ = self._operability_provenance(df)
        fit = sensors[stable]
        if fit_end:
            fit = fit.loc[:fit_end]
        fit = fit.dropna().tail(FIT_POINTS)
        self.fit_start_, self.fit_end_ = fit.index.min(), fit.index.max()
        self.n_fit_points = len(fit)

        self.pca_temp = scoring.MultivariateScorer().fit(fit[config.TEMPERATURE_TAGS])
        self.pca_press = scoring.MultivariateScorer().fit(fit[config.PRESSURE_TAGS])
        b = _spread_mancal(fit)
        self.spread_med_ = float(b.median())
        self.spread_mad_ = float((b - b.median()).abs().median() * 1.4826)
        return self

    @staticmethod
    def _operability_provenance(df: pd.DataFrame) -> dict:
        """Registra em que sinal o baseline se apoiou — vai para os metadados."""
        if "operability_source" not in df.columns:
            return {"fonte": "resolvida_em_tempo_de_uso"}
        counts = df.loc[df.get("stable", pd.Series(True, index=df.index)).astype(bool),
                        "operability_source"].value_counts()
        total = max(int(counts.sum()), 1)
        return {str(k): round(float(v) / total * 100, 2) for k, v in counts.items()}

    def _mask(self, df: pd.DataFrame) -> pd.Series:
        """Máscara de pontuação: operação estável menos o blackout pós-partida.

        A operabilidade vem do ``OperabilityResolver`` (NGP_A com fallback
        explícito para RUNNING_A). Se o canônico já traz as colunas resolvidas,
        elas são reaproveitadas; caso contrário resolve-se na hora, para que
        pontuar um recorte cru dê exatamente o mesmo resultado.
        """
        resolution = None
        if {"stable", "in_operation"} <= set(df.columns):
            stable = df["stable"].astype(bool)
            in_operation = df["in_operation"].astype(bool)
        else:
            resolution = OperabilityResolver().resolve(df)
            stable, in_operation = resolution.stable, resolution.in_operation

        starts = in_operation & ~in_operation.shift(fill_value=False)
        n_black = int(pd.Timedelta(BLACKOUT) / pd.Timedelta(config.GRID))
        blackout = starts.rolling(n_black, min_periods=1).max().astype(bool)
        return stable & ~blackout

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """`df` no formato do canônico (sensores + operabilidade resolvida)."""
        sensors = df[config.SENSOR_TAGS]
        mask = self._mask(df)

        out = pd.DataFrame(index=df.index)
        out["score_temp"] = (self.pca_temp.score(sensors[config.TEMPERATURE_TAGS])
                             ["pca_recon"].ewm(halflife=pd.Timedelta("1h"),
                                               times=df.index).mean().where(mask))
        out["score_press"] = (self.pca_press.score(sensors[config.PRESSURE_TAGS])
                              ["pca_recon"].ewm(halflife=pd.Timedelta("1h"),
                                                times=df.index).mean().where(mask))
        z = (_spread_mancal(sensors) - self.spread_med_) / self.spread_mad_
        out["z_spread"] = z.abs().ewm(halflife=pd.Timedelta("30min"),
                                      times=df.index).mean().where(mask)

        s1 = _sustained(out["score_temp"], THR_FAM)
        s2 = _sustained(out["score_press"], THR_FAM)
        s3 = _sustained(out["z_spread"], THR_SPREAD)
        n = s1.astype(int) + s2.astype(int) + s3.astype(int)
        out["atencao"] = n >= 1
        out["confirmado"] = n >= 2
        return out

    @staticmethod
    def load(models_dir) -> "Detector":
        import joblib
        from pathlib import Path
        return joblib.load(Path(models_dir) / "detector_fp_first.joblib")
