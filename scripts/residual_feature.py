"""Computa a feature de resíduo common-mode em produção/streaming, a partir das
6 temperaturas brutas TC382_0X_A + RUNNING_A.

Reproduz exatamente `build_residual_csv.py` (residual = TC_X - média dos outros 5
válidos), mas operando linha-a-linha sobre um dataframe qualquer (histórico ou lote de
streaming) em vez de reescrever um CSV inteiro — é o passo que falta entre o dado bruto
que chega em produção e o bundle `TC382_03_A_inference_bundle.json` treinado sobre
resíduo, que espera receber uma coluna `TC382_03_A` já em unidades de resíduo.

Uso:
    from scripts.residual_feature import add_residual_column
    df_resid = add_residual_column(df_raw, target="TC382_03_A")
    scores = score_production(model, bundle, df_resid)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TCS = ["TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A", "TC382_06_A"]
LOW, HIGH = 500.0, 950.0
MIN_PEERS = 3


def add_residual_column(
    df: pd.DataFrame,
    target: str = "TC382_03_A",
    tcs: list[str] = TCS,
    running_col: str = "RUNNING_A",
    low: float = LOW,
    high: float = HIGH,
    min_peers: int = MIN_PEERS,
) -> pd.DataFrame:
    """Retorna uma cópia de `df` com a coluna `target` substituída pelo resíduo
    `target - média dos outros sensores de `tcs` válidos` (mesma máscara/gate de
    `build_residual_csv.py`: valor em [low, high], RUNNING_A>0.5, >=min_peers pares
    válidos; caso contrário resíduo = 0.0).
    """
    peers = [c for c in tcs if c != target]
    running = pd.to_numeric(df[running_col], errors="coerce").fillna(0.0)
    on = (running > 0.5).to_numpy()

    masked = {}
    for c in [target] + peers:
        v = pd.to_numeric(df[c], errors="coerce")
        masked[c] = v.where((v >= low) & (v <= high))

    peer_mat = pd.concat([masked[c] for c in peers], axis=1)
    peer_mat.columns = peers
    peer_mean = peer_mat.mean(axis=1)
    n_peers = peer_mat.notna().sum(axis=1)

    resid = masked[target] - peer_mean
    ok = on & masked[target].notna().to_numpy() & (n_peers >= min_peers).to_numpy()

    out = df.copy()
    out[target] = np.where(ok, resid.to_numpy(), 0.0)
    return out
