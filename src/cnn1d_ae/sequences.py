from __future__ import annotations

import numpy as np
from typing import Tuple


def make_sequences(values_2d: np.ndarray, time_steps: int, stride: int) -> np.ndarray:
    n = len(values_2d)
    last = n - time_steps
    out = []
    for i in range(0, last + 1, stride):
        out.append(values_2d[i : i + time_steps])
    return np.stack(out, axis=0)


def sequence_all_true(mask_1d: np.ndarray, time_steps: int, stride: int) -> np.ndarray:
    """Para cada janela que make_sequences() geraria a partir de um array do
    mesmo tamanho de mask_1d (mesmos time_steps/stride), retorna True se
    TODOS os pontos da janela sao True em mask_1d. Usado para filtrar a
    fatia de calibracao conformal por operational_state=="on" -- indices
    de saida alinhados 1:1 com make_sequences(mesmo array-base)."""
    n = len(mask_1d)
    last = n - time_steps
    starts = np.arange(0, last + 1, stride)
    cum = np.concatenate(([0], np.cumsum(mask_1d.astype(np.int64))))
    window_sum = cum[starts + time_steps] - cum[starts]
    return window_sum == time_steps


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


def train_val_calib_split(
    x: np.ndarray,
    val_frac: float,
    calib_frac: float,
    shuffle: bool,
    seed: int,
    split_mode: str = "temporal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Como train_val_split, mas reserva uma 3a fatia (calibracao) para o
    THRESH_MODE="conformal". Ordem temporal: train | val | calib -- calib
    fica com o trecho mais recente (mais proximo do corte OOS), val fica no
    meio (mesma posicao de sempre quando calib_frac=0, preservando o
    comportamento existente byte-a-byte)."""
    n_total = x.shape[0]
    n_calib = int(np.floor(calib_frac * n_total))
    n_val = int(np.floor(val_frac * n_total))
    n_train = n_total - n_val - n_calib

    if n_train <= 0 or n_val <= 0 or (calib_frac > 0 and n_calib <= 0):
        raise ValueError("VAL_FRAC/CALIBRATION_FRAC resultou em split invalido para a quantidade de sequencias.")

    mode = split_mode.lower()
    if mode == "temporal":
        return x[:n_train], x[n_train:n_train + n_val], x[n_train + n_val:]

    if mode == "random":
        idx = np.arange(n_total)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        calib_idx = idx[n_train + n_val:]
        return x[train_idx], x[val_idx], x[calib_idx]

    raise ValueError("SPLIT_MODE invalido. Use 'temporal' ou 'random'.")
