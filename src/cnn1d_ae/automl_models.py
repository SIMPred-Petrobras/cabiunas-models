from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


def build_dense_autoencoder(
    n_features: int, layer_sizes: Sequence[int], dropout: float, lr: float
) -> keras.Model:
    """Autoencoder denso (nao-sequencial): reconstroi o vetor de features de um
    unico instante de tempo. Equivalente Keras ao modelo 'dense' em PyTorch
    usado na pipeline de AutoML da Lara (ver analise_automl_lara.md)."""
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units in layer_sizes:
        x = layers.Dense(units, activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    for units in reversed(list(layer_sizes)[:-1]):
        x = layers.Dense(units, activation="relu")(x)
    outputs = layers.Dense(n_features)(x)

    model = keras.Model(inputs, outputs, name="dense_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def dense_reconstruction_error(model: keras.Model, x: np.ndarray, batch_size: int) -> np.ndarray:
    x_pred = model.predict(x, batch_size=batch_size, verbose=0)
    return np.mean(np.square(x_pred - x), axis=1)


def fit_ocsvm(x_train: np.ndarray, nu: float, gamma: Any) -> OneClassSVM:
    clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    clf.fit(x_train)
    return clf


def ocsvm_error(clf: OneClassSVM, x: np.ndarray) -> np.ndarray:
    return (-clf.decision_function(x)).astype("float32")


def fit_isolation_forest(
    x_train: np.ndarray, contamination: float, n_estimators: int, random_state: int
) -> IsolationForest:
    model = IsolationForest(
        contamination=contamination, n_estimators=n_estimators, random_state=random_state
    )
    model.fit(x_train)
    return model


def isolation_forest_error(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    """Quanto MAIOR, mais anomalo (score_samples e o inverso)."""
    return -model.score_samples(x)
