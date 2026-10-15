"""Escores POR FAMÍLIA de tags (temperatura, pressão), no desenho do canal do v2.

Dois tipos de escore, para separar duas coisas que o EXP 04/06 misturaram (escores globais e agregação por soma/distância):
  RESÍDUO POR SENSOR  o modelo prevê/reconstrói cada tag da família; o escore é o MÁXIMO, entre as tags, do erro quadrado normalizado pelo p99 do erro da tag no treino (piso 0,1 x mediana dos p99), dividido pelo p99 do próprio escore no treino: exatamente a agregação `ScorerMax` do canal t/p do v2 (`cabiunas_inference._recon_pca`).
      pca1, pca3   PCA com 1 e 3 componentes (o bundle atual da temperatura tem k = 1; a pressão, k = 4)
      ridge        cada tag prevista por regressão ridge nas OUTRAS tags (o "AAKR/MSET" linear)
      gbm          cada tag prevista por LightGBM nas outras (não linear; sensível a regime)
  DENSIDADE CONJUNTA  um escore único de quão raro é o vetor inteiro (sem agregação por sensor)
      ocsvm, iforest, knn, lof, gmm, ecod
Todos aceitam `fit(X)` (X finito, n x d) e `score(X)` (vetor); a escala robusta (mediana/MAD com piso) é a do restante da nova frente.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
import lightgbm as lgb

from .baselines import _Base


class _Residuo(_Base):
    """Base dos escores por sensor: subclasses implementam `_ajusta(Z)` e `_erro2(Z)` (n x d)."""

    def fit(self, X):
        Z = self._ajusta_escala(X); self._ajusta(Z); e = self._erro2(Z)
        p99 = np.quantile(e, 0.99, axis=0); self.sens = np.maximum(p99, 0.1 * np.median(p99))
        self.p99s = float(np.quantile((e / self.sens).max(axis=1), 0.99)); return self

    def _score(self, Z):
        return (self._erro2(Z) / self.sens).max(axis=1) / max(self.p99s, 1e-12)


class PCAk(_Residuo):
    k = 1

    def _ajusta(self, Z):
        self.pca = PCA(n_components=self.k, svd_solver="full").fit(Z)

    def _erro2(self, Z):
        return (Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2


class PCA1(PCAk):
    k = 1


class PCA3(PCAk):
    k = 3


class RidgeOutras(_Residuo):
    def _ajusta(self, Z):
        d = Z.shape[1]; self.m = [Ridge(alpha=1.0).fit(np.delete(Z, j, axis=1), Z[:, j]) for j in range(d)]

    def _erro2(self, Z):
        return np.column_stack([(Z[:, j] - self.m[j].predict(np.delete(Z, j, axis=1))) ** 2 for j in range(Z.shape[1])])


class GBMOutras(_Residuo):
    def _ajusta(self, Z):
        d = Z.shape[1]; self.m = [lgb.LGBMRegressor(n_estimators=80, learning_rate=0.1, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, random_state=self.semente + j, verbose=-1, n_jobs=2).fit(np.delete(Z, j, axis=1), Z[:, j]) for j in range(d)]

    def _erro2(self, Z):
        return np.column_stack([(Z[:, j] - self.m[j].predict(np.delete(Z, j, axis=1))) ** 2 for j in range(Z.shape[1])])


class _Sub(_Base):
    n_sub = 4000

    def _sub(self, Z):
        return Z[np.random.default_rng(self.semente).choice(len(Z), size=min(self.n_sub, len(Z)), replace=False)]


class OCSVM(_Sub):
    def fit(self, X):
        Z = self._ajusta_escala(X); self.m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02).fit(self._sub(Z)); return self

    def _score(self, Z):
        return -self.m.decision_function(Z)


class IForestF(_Base):
    def fit(self, X):
        Z = self._ajusta_escala(X); self.m = IsolationForest(n_estimators=200, max_samples=512, random_state=self.semente, n_jobs=2).fit(Z); return self

    def _score(self, Z):
        return -self.m.score_samples(Z)


class KNNF(_Sub):
    def fit(self, X):
        Z = self._ajusta_escala(X); self.nn = NearestNeighbors(n_neighbors=10, n_jobs=2).fit(self._sub(Z)); return self

    def _score(self, Z):
        d, _ = self.nn.kneighbors(Z); return d.mean(axis=1)


class LOF(_Sub):
    def fit(self, X):
        Z = self._ajusta_escala(X); self.m = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=2).fit(self._sub(Z)); return self

    def _score(self, Z):
        return -self.m.score_samples(Z)


class GMM(_Sub):
    def fit(self, X):
        Z = self._ajusta_escala(X); self.m = GaussianMixture(n_components=6, covariance_type="full", reg_covar=1e-3, random_state=self.semente).fit(self._sub(Z)); return self

    def _score(self, Z):
        return -self.m.score_samples(Z)


class ECODlite(_Base):
    """ECOD simplificado (implementação própria, sem pyod): soma, entre as dimensões, de -log da probabilidade de cauda empírica (a pior das duas caudas)."""

    def fit(self, X):
        Z = self._ajusta_escala(X); self.s = np.sort(Z, axis=0); return self

    def _score(self, Z):
        n = len(self.s); out = np.zeros(len(Z))
        for j in range(Z.shape[1]):
            r = np.searchsorted(self.s[:, j], Z[:, j]) / n
            out += -np.log(np.clip(np.minimum(r, 1 - r) * 2, 1.0 / n, 1.0))
        return out


def deslocado(cls):
    """Escores de densidade podem ser negativos (OCSVM, LOF, GMM): subtrai a mediana do treino, para que o p99 do treino (usado em `walk_forward` para igualar os meses) seja positivo."""
    class D(cls):
        def fit(self, X):
            super().fit(X); self.o = float(np.nanmedian(super().score(X))); return self

        def score(self, X):
            return super().score(X) - self.o
    D.__name__ = cls.__name__
    return D


FAMILIA_FABRICAS = {"pca1": PCA1, "pca3": PCA3, "ridge": RidgeOutras, "gbm": GBMOutras, **{n: deslocado(c) for n, c in (("ocsvm", OCSVM), ("iforest", IForestF), ("knn", KNNF), ("lof", LOF), ("gmm", GMM), ("ecod", ECODlite))}}
ESTOCASTICOS_F = {"gbm", "ocsvm", "iforest", "knn", "lof", "gmm"}
