"""
model_if.py
Isolation Forest como autoencoder drop-in: extrai features estatísticas de cada
janela temporal e produz um score de anomalia no mesmo formato que o CNN/GRU AE
(array 1-D de floats por sequência, escala [0, 1]).

Interface pública:
    train_scores, all_scores = fit_and_score(x_train, x_all, cfg)

Os arrays retornados substituem train_mae_seq / mae_seq_all no pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from .config import PipelineConfig


# ---------------------------------------------------------------------------
# Extração de features por janela
# ---------------------------------------------------------------------------

def _extract_features(sequences: np.ndarray) -> np.ndarray:
    """Converte (N, T, 1) → (N, F) com features estatísticas de cada janela.

    Features (14 por canal):
        mean, std, min, max, range,
        p10, p25, p75, p90,
        skewness, kurtosis,
        autocorr_lag1, autocorr_lag2,
        linear_slope
    """
    x = sequences[:, :, 0]          # (N, T) — univariado
    N, T = x.shape
    feats = []

    feats.append(x.mean(axis=1))
    feats.append(x.std(axis=1) + 1e-9)
    feats.append(x.min(axis=1))
    feats.append(x.max(axis=1))
    feats.append(x.max(axis=1) - x.min(axis=1))

    feats.append(np.percentile(x, 10, axis=1))
    feats.append(np.percentile(x, 25, axis=1))
    feats.append(np.percentile(x, 75, axis=1))
    feats.append(np.percentile(x, 90, axis=1))

    feats.append(stats.skew(x, axis=1))
    feats.append(stats.kurtosis(x, axis=1))

    # Autocorrelação lag-1 e lag-2 (via numpy correlate normalizado)
    def autocorr_lag(arr, lag):
        mu = arr.mean(axis=1, keepdims=True)
        centered = arr - mu
        n = arr.shape[1] - lag
        ac = (centered[:, :n] * centered[:, lag:]).sum(axis=1)
        denom = (centered ** 2).sum(axis=1) + 1e-9
        return ac / denom

    feats.append(autocorr_lag(x, 1))
    feats.append(autocorr_lag(x, 2))

    # Tendência linear (slope normalizado pelo desvio)
    t = np.arange(T, dtype=float)
    t_c = t - t.mean()
    slopes = (x * t_c).sum(axis=1) / (t_c ** 2).sum()
    feats.append(slopes)

    return np.stack(feats, axis=1)   # (N, 14)


# ---------------------------------------------------------------------------
# Treino + scoring
# ---------------------------------------------------------------------------

def fit_and_score(
    x_train: np.ndarray,
    x_all: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Treina IF em x_train (normal), retorna scores em [0,1] para train e all.

    Score alto = mais anômalo (compatível com mae_seq semântica do CNN-AE).
    """
    n_estimators    = getattr(cfg, "IF_N_ESTIMATORS", 200)
    contamination   = getattr(cfg, "IF_CONTAMINATION", "auto")
    max_samples     = getattr(cfg, "IF_MAX_SAMPLES", "auto")
    random_state    = getattr(cfg, "RANDOM_SEED", 42)

    print(f"[IF] Extraindo features de {len(x_train)} janelas de treino...")
    feat_train = _extract_features(x_train)

    print(f"[IF] Treinando IsolationForest "
          f"(n_estimators={n_estimators}, contamination={contamination})...")
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(feat_train)

    # score_samples retorna log-density: menor = mais anômalo
    # Invertemos e normalizamos para [0,1] (alto = anômalo)
    print(f"[IF] Scoring {len(x_all)} janelas totais...")
    feat_all = _extract_features(x_all)

    raw_train = -clf.score_samples(feat_train)   # positivo, maior = mais anômalo
    raw_all   = -clf.score_samples(feat_all)

    # Normaliza pelo range de treino para escala consistente
    scaler = MinMaxScaler()
    scaler.fit(raw_train.reshape(-1, 1))
    train_scores = scaler.transform(raw_train.reshape(-1, 1)).ravel()
    all_scores   = scaler.transform(raw_all.reshape(-1, 1)).ravel()

    n_anom_train = (clf.predict(feat_train) == -1).sum()
    print(f"[IF] Treino: {n_anom_train}/{len(x_train)} janelas marcadas como outlier "
          f"({100*n_anom_train/max(len(x_train),1):.1f}%)")

    return train_scores, all_scores
