from __future__ import annotations

import numpy as np
from typing import Tuple


def make_sequences(values_2d: np.ndarray, time_steps: int, stride: int) -> np.ndarray:
    if values_2d.ndim != 2:
        raise ValueError("make_sequences espera entrada 2D: [n_pontos, n_features].")

    n = len(values_2d)
    if n < time_steps:
        raise ValueError("Quantidade de pontos menor que TIME_STEPS; nao ha sequencias suficientes.")

    last = n - time_steps
    out = []
    for i in range(0, last + 1, stride):
        out.append(values_2d[i : i + time_steps])
    return np.stack(out, axis=0)


def train_val_split(
    x: np.ndarray,
    val_frac: float,
    shuffle: bool,
    seed: int,
    split_mode: str = "temporal",
) -> Tuple[np.ndarray, np.ndarray]:
    n_total = x.shape[0]
    n_val = int(np.floor(val_frac * n_total))
    n_train = n_total - n_val

    if n_val <= 0 or n_train <= 0:
        raise ValueError("VAL_FRAC resultou em split invalido para a quantidade de sequencias.")

    mode = split_mode.lower()
    if mode == "temporal":
        return x[:n_train], x[n_train:]

    if mode == "random":
        idx = np.arange(n_total)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]
        return x[train_idx], x[val_idx]

    raise ValueError("SPLIT_MODE invalido. Use 'temporal' ou 'random'.")
