"""A escada de baselines da Fase 2: do mais simples ao mais flexível, todos sobre as MESMAS tags, a MESMA escala robusta e o MESMO treino causal.

  rz_max    z robusto por tag, escore = maior |z|                        (o univariado)
  pca_spe   PCA (95% da variância), erro de reconstrução (SPE)           (o clássico multivariado; o v2 usa uma variante por família)
  pca_t2    T2 de Hotelling no espaço dos componentes                   (o desvio DENTRO do subespaço)
  maha_mcd  distância de Mahalanobis com covariância robusta (MCD)      (multivariado robusto)
  iforest   Isolation Forest                                            (não paramétrico, árvores)
  knn       distância média aos k vizinhos mais próximos no treino      (não paramétrico, memória)

A escala por tag é mediana e 1,4826 x MAD do treino com PISO (5% da mediana de |x|, mínimo 1e-6): sem piso, sensores de estado de MAD ~0 (PI_0319) dominam todo escore.
"""
from __future__ import annotations
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


class Escala:
    def fit(self, X):
        self.med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - self.med), axis=0)
        self.sc = np.maximum(np.maximum(mad, 0.05 * np.median(np.abs(X), axis=0)), 1e-6)
        return self

    def transform(self, X):
        return (X - self.med) / self.sc


class _Base:
    def __init__(self, semente: int = 0):
        self.semente = semente

    def _ajusta_escala(self, X):
        self.escala = Escala().fit(X)
        return self.escala.transform(X)

    def score(self, X):
        out = np.full(len(X), np.nan); ok = np.isfinite(X).all(axis=1)
        if ok.any():
            out[ok] = self._score(self.escala.transform(X[ok]))
        return out


class RZMax(_Base):
    def fit(self, X):
        self._ajusta_escala(X); return self

    def _score(self, Z):
        return np.abs(Z).max(axis=1)


class PCASPE(_Base):
    def fit(self, X):
        Z = self._ajusta_escala(X); self.pca = PCA(n_components=0.95, svd_solver="full").fit(Z); return self

    def _score(self, Z):
        return ((Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2).sum(axis=1)


class PCAT2(_Base):
    def fit(self, X):
        Z = self._ajusta_escala(X); self.pca = PCA(n_components=0.95, svd_solver="full").fit(Z); return self

    def _score(self, Z):
        T = self.pca.transform(Z)
        return (T ** 2 / self.pca.explained_variance_).sum(axis=1)


class MahaMCD(_Base):
    def fit(self, X):
        Z = self._ajusta_escala(X); rng = np.random.default_rng(self.semente)
        sub = Z[rng.choice(len(Z), size=min(4000, len(Z)), replace=False)]
        self.mcd = MinCovDet(support_fraction=0.9, random_state=self.semente).fit(sub); return self

    def _score(self, Z):
        return np.sqrt(self.mcd.mahalanobis(Z))


class IForest(_Base):
    def fit(self, X):
        Z = self._ajusta_escala(X)
        self.m = IsolationForest(n_estimators=200, max_samples=512, random_state=self.semente, n_jobs=4).fit(Z); return self

    def _score(self, Z):
        return -self.m.score_samples(Z)


class KNN(_Base):
    def fit(self, X):
        Z = self._ajusta_escala(X); rng = np.random.default_rng(self.semente)
        self.pca = PCA(n_components=min(10, Z.shape[1]), svd_solver="full").fit(Z)
        ref = self.pca.transform(Z[rng.choice(len(Z), size=min(5000, len(Z)), replace=False)])
        self.nn = NearestNeighbors(n_neighbors=10, n_jobs=4).fit(ref); return self

    def _score(self, Z):
        d, _ = self.nn.kneighbors(self.pca.transform(Z))
        return d.mean(axis=1)


FABRICAS = {"rz_max": RZMax, "pca_spe": PCASPE, "pca_t2": PCAT2, "maha_mcd": MahaMCD, "iforest": IForest, "knn": KNN}
