#!/usr/bin/env python3
"""Referencia rolante com banda de guarda -- o detector compara a maquina com
ela mesma, recentemente, pulando as ultimas GUARDA horas.

Motivo de existir. O baseline mensal do protocolo do Francisco existe porque
retreinar autoencoder e caro; para um z robusto nao ha esse custo, e o preco de
esperar o mes virar e alto: um evento no dia 28 e comparado contra dado de ate
oito semanas antes, e a maquina deriva nesse prazo. A autopsia dos 9 eventos --
que usou os 30 dias imediatamente anteriores com 72 h de folga -- via z de +9 a
+24 onde o baseline mensal ve quase nada.

A banda de guarda e o que impede a degradacao de virar o proprio normal: sem
ela, uma deriva lenta e absorvida pela referencia em poucas horas.

Consequencia util: essa construcao e um passa-alta no estado da maquina. Um
regime novo que se instala vira a nova referencia depois de HORAS_BASE horas de
operacao (e para de alarmar); uma degradacao progressiva continua correndo na
frente da propria referencia (e segue alarmando). E exatamente a distincao que
o detector precisa fazer e que o baseline mensal nao fazia.
"""
from __future__ import annotations
import numpy as np, pandas as pd

PAS = pd.Timedelta("2min")
POR_H = 30                      # amostras de 2 min por hora


def z_rolante(X: pd.DataFrame, quente: pd.Series, falhas: pd.Series,
              horas_base=400.0, guarda_h=72.0, phi=0.0, passo_h=6.0,
              excl_dias=7.0) -> pd.DataFrame:
    idx = X.index
    hot = np.flatnonzero(quente.to_numpy())
    if hot.size == 0:
        return pd.DataFrame(np.nan, index=idx, columns=X.columns, dtype="float32")

    Xh = X.to_numpy()[hot].astype("float64")            # so operacao quente
    Xr = Xh.copy()                                      # copia usada como referencia
    th = idx[hot]
    for f in falhas:                                    # tira degradacao conhecida da referencia
        m = (th >= f - pd.Timedelta(days=excl_dias)) & (th <= f + pd.Timedelta(days=2))
        Xr[np.asarray(m)] = np.nan

    n_base = int(horas_base * POR_H)
    guarda = int(guarda_h * POR_H)
    passo = max(1, int(passo_h * POR_H))
    n, C = Xh.shape
    Zh = np.full((n, C), np.nan)

    glob_s = None
    for k in range(0, n, passo):
        fim = k - guarda
        ini = max(0, fim - n_base)
        if fim - ini < n_base // 4:
            continue
        W = Xr[ini:fim]
        med = np.nanmedian(W, axis=0)
        s = np.nanmedian(np.abs(W - med), axis=0) * 1.4826
        if phi > 0:
            # escala global so precisa da ordem de grandeza: subamostra 1/20 e
            # so recalcula a cada 20 passos (senao sao O(n) medianas por passo)
            if glob_s is None or (k // passo) % 20 == 0:
                G = Xr[:fim:20]
                gm = np.nanmedian(G, axis=0)
                glob_s = np.nanmedian(np.abs(G - gm), axis=0) * 1.4826
            s = np.maximum(s, phi * glob_s)
        s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
        j = min(n, k + passo)
        Zh[k:j] = np.abs((Xh[k:j] - med) / s)

    Z = pd.DataFrame(np.nan, index=idx, columns=X.columns, dtype="float32")
    Z.iloc[hot] = Zh.astype("float32")
    return Z


def limiar_rolante(Z: pd.DataFrame, quente: pd.Series, falhas: pd.Series,
                   alpha: float, horas_cal=700.0, guarda_h=72.0, passo_h=6.0,
                   excl_dias=7.0):
    """Mesmo esquema para o corte: percentil de Sidak tomado nas ultimas
    horas_cal de score ja produzido, pulando a guarda."""
    M = Z.shape[1]
    p = (1.0 - alpha) ** (1.0 / M)
    idx = Z.index
    hot = np.flatnonzero(quente.to_numpy())
    Zh = Z.to_numpy()[hot].astype("float64")
    Zr = Zh.copy()
    th = idx[hot]
    for f in falhas:
        m = (th >= f - pd.Timedelta(days=excl_dias)) & (th <= f + pd.Timedelta(days=2))
        Zr[np.asarray(m)] = np.nan
    n_cal = int(horas_cal * POR_H); guarda = int(guarda_h * POR_H)
    passo = max(1, int(passo_h * POR_H))
    n, C = Zh.shape
    rh = np.full(n, np.nan); ch = np.full(n, -1, dtype=int)
    for k in range(0, n, passo):
        fim = k - guarda; ini = max(0, fim - n_cal)
        if fim - ini < n_cal // 4:
            continue
        q = np.nanquantile(Zr[ini:fim], p, axis=0)
        q = np.where(np.isfinite(q) & (q > 0), q, np.nan)
        j = min(n, k + passo)
        r = Zh[k:j] / q
        todo = np.isnan(r).all(axis=1)
        rr = np.where(np.isnan(r), -np.inf, r)
        a = np.argmax(rr, axis=1)
        rh[k:j] = np.where(todo, np.nan, rr[np.arange(j - k), a])
        ch[k:j] = np.where(todo, -1, a)
    razao = pd.Series(np.nan, index=idx, dtype="float32"); razao.iloc[hot] = rh.astype("float32")
    cols = np.asarray(Z.columns)
    culp = pd.Series(pd.NA, index=idx, dtype="object")
    culp.iloc[hot] = np.where(ch >= 0, cols[np.clip(ch, 0, None)], None)
    return razao, culp
