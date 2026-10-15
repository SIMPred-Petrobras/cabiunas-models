"""Autoencoder SEQUENCIAL de janela (convolução 1-D no tempo, PyTorch), retreinado por mês sobre as últimas `n_max` janelas estáveis anteriores.

  entrada   janela de W amostras (30 min a 2 min) das d tags, escala robusta com piso; o escore de cada instante usa a janela que TERMINA nele (só passado)
  rede      Conv1d(d,32,3) -> Conv1d(32,16,3,passo 2) -> linear k=4 -> linear -> d x W
  corrupção, perda, escore: os mesmos do AE pontual (máscara de 15% por tag-amostra, ruído 0,05, Huber; escore = máximo entre as tags do erro quadrado médio na janela normalizado pelo p99 da tag no treino, piso 0,1 x mediana, dividido pelo p99 do escore no treino)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from .baselines import Escala
from ..walk import inicios_de_mes

W = 15


class Rede(nn.Module):
    def __init__(self, d: int, w: int = W, k: int = 4):
        super().__init__()
        self.d, self.w = d, w
        self.conv = nn.Sequential(nn.Conv1d(d, 32, 3, padding=1), nn.GELU(), nn.Conv1d(32, 16, 3, stride=2, padding=1), nn.GELU())
        n = 16 * ((w + 1) // 2)
        self.enc = nn.Linear(n, k); self.dec = nn.Sequential(nn.Linear(k, 64), nn.GELU(), nn.Linear(64, d * w))

    def forward(self, x):  # x: (n, w, d)
        z = self.enc(self.conv(x.transpose(1, 2)).flatten(1))
        return self.dec(z).view(-1, self.w, self.d)


def janelas(Z, ends):
    """Z (n,d) -> (len(ends), W, d) com a janela terminando em cada índice de `ends` (>= W-1)."""
    V = sliding_window_view(Z, W, axis=0)  # (n-W+1, d, W)
    return np.ascontiguousarray(V[np.asarray(ends) - (W - 1)].transpose(0, 2, 1))


class ModeloTCN:
    def __init__(self, d: int, semente: int = 0):
        self.d, self.semente = d, semente
        torch.manual_seed(semente); self.net = Rede(d)

    def fit(self, X, ends, epocas=30, lote=512, lr=2e-3):
        self.escala = Escala().fit(X[ends]); Zn = self.escala.transform(X)
        J = torch.as_tensor(janelas(Zn, ends), dtype=torch.float32)
        g = torch.Generator().manual_seed(self.semente)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4); perda = nn.HuberLoss(delta=1.0); n = len(J); self.net.train()
        for _ in range(epocas):
            perm = torch.randperm(n, generator=g)
            for a in range(0, n, lote):
                x = J[perm[a:a + lote]]; m = torch.rand(x.shape, generator=g) < 0.15
                xc = x.masked_fill(m, 0.0) + 0.05 * torch.randn(x.shape, generator=g)
                opt.zero_grad(); perda(self.net(xc), x).backward(); opt.step()
        self.net.eval(); e = self._erro2(J.numpy())
        am = e[np.random.default_rng(self.semente).choice(len(e), size=min(len(e), 50000), replace=False)]
        p99 = np.quantile(am, 0.99, axis=0); self.sens = np.maximum(p99, 0.1 * np.median(p99))
        self.p99s = float(np.quantile((am / self.sens).max(axis=1), 0.99)); return self

    @torch.no_grad()
    def _erro2(self, J):
        out = np.empty((len(J), self.d), dtype=np.float32)
        for a in range(0, len(J), 4096):
            x = torch.as_tensor(J[a:a + 4096], dtype=torch.float32); out[a:a + 4096] = ((self.net(x) - x) ** 2).mean(1).numpy()
        return out

    def score(self, X, ends):
        Zn = self.escala.transform(X); J = janelas(Zn, ends); return (self._erro2(J) / self.sens).max(axis=1) / max(self.p99s, 1e-12)


def walk_tcn(g: pd.DataFrame, tags: list[str], estavel: np.ndarray, ini, fim, semente=0, n_max=20000, ep0=30) -> pd.Series:
    X = g[tags].to_numpy("float64"); idx = g.index; n = len(g); out = np.full(n, np.nan); meses = inicios_de_mes(ini, fim)
    ok = np.isfinite(X).all(axis=1); okw = np.zeros(n, bool)  # janela completa finita terminando em i
    c = np.convolve(ok.astype(int), np.ones(W, int), "full")[:n]; okw[W - 1:] = c[W - 1:] == W
    ce = np.convolve((estavel & ok).astype(int), np.ones(W, int), "full")[:n]; estw = np.zeros(n, bool); estw[W - 1:] = ce[W - 1:] == W
    for k, m0 in enumerate(meses):
        m1 = meses[k + 1] if k + 1 < len(meses) else fim + pd.Timedelta("2min"); i0 = idx.searchsorted(m0); i1 = idx.searchsorted(m1)
        cand = np.flatnonzero(estw[:i0])
        if len(cand) < 5000:
            continue
        Xf = np.nan_to_num(X, nan=0.0); mod = ModeloTCN(len(tags), semente + 1000 * k).fit(Xf, cand[-n_max:], ep0)
        e = np.flatnonzero(okw[i0:i1]) + i0
        if len(e):
            out[e] = mod.score(Xf, e)
    return pd.Series(out, index=idx)
