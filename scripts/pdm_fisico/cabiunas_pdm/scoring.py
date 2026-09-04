"""Scores de anomalia de pré-modelagem (sem redes neurais):

- Univariado: z-score robusto (mediana/MAD do baseline) por sensor, com EWMA.
- Multivariado: erro de reconstrução PCA e distância de Mahalanobis
  (covariância Ledoit-Wolf), ambos ajustados apenas no baseline estável.

Servem para responder ANTES do treino: há estrutura de normal aprendível?
Há sinal precursor nos trips? Univariado basta ou multivariado agrega?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


# ------------------------------------------------------------- univariado
def robust_z(df: pd.DataFrame, baseline: pd.DataFrame,
             ewma_halflife: str = "30min") -> pd.DataFrame:
    """Z-score robusto por coluna relativo ao baseline, suavizado por EWMA."""
    med = baseline.median()
    mad = (baseline - med).abs().median() * 1.4826
    mad = mad.replace(0, np.nan)
    z = (df - med) / mad
    return z.ewm(halflife=pd.Timedelta(ewma_halflife), times=df.index).mean()


# ----------------------------------------------------------- multivariado
@dataclass
class MultivariateScorer:
    n_components: float = 0.95     # fração de variância retida no PCA
    scaler: RobustScaler | None = None
    pca: PCA | None = None
    lw: LedoitWolf | None = None
    cols: list[str] | None = None
    recon_p99: float | None = None
    maha_p99: float | None = None

    def fit(self, baseline: pd.DataFrame) -> "MultivariateScorer":
        X = baseline.dropna()
        self.cols = list(X.columns)
        self.scaler = RobustScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.pca = PCA(n_components=self.n_components, svd_solver="full").fit(Xs)
        self.lw = LedoitWolf().fit(Xs)
        r, m = self._raw_scores(X)
        self.recon_p99 = float(np.nanpercentile(r, 99))
        self.maha_p99 = float(np.nanpercentile(m, 99))
        return self

    def _raw_scores(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[self.cols]
        mask = X.notna().all(axis=1).to_numpy()
        recon = np.full(len(X), np.nan)
        maha = np.full(len(X), np.nan)
        if mask.any():
            Xs = self.scaler.transform(X[mask])
            proj = self.pca.inverse_transform(self.pca.transform(Xs))
            recon[mask] = np.mean((Xs - proj) ** 2, axis=1)
            maha[mask] = self.lw.mahalanobis(Xs)
        return recon, maha

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scores normalizados pelo p99 do baseline (>1 = fora do normal)."""
        recon, maha = self._raw_scores(df)
        return pd.DataFrame(
            {"pca_recon": recon / self.recon_p99, "mahalanobis": maha / self.maha_p99},
            index=df.index,
        )


# ------------------------------------------------------------- lead time
def first_sustained_exceedance(score: pd.Series, threshold: float,
                               sustain: int, trip_time: pd.Timestamp,
                               search_start: pd.Timestamp) -> pd.Timestamp | None:
    """Primeiro instante, em [search_start, trip], em que o score fica acima do
    threshold por `sustain` amostras consecutivas (debounce)."""
    w = score.loc[search_start:trip_time]
    above = (w > threshold).astype(int)
    run = above.rolling(sustain, min_periods=sustain).sum()
    hits = run[run >= sustain]
    if hits.empty:
        return None
    return hits.index[0]
