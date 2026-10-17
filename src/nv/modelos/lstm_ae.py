"""Autoencoder LSTM de janela (o modelo mais citado na literatura de turbina a gás), retreinado por mês sobre as últimas `n_max` janelas estáveis anteriores.

  janela de W = 15 amostras (30 min) terminando no instante (só passado); escala robusta como nos demais; codificador LSTM(d, 32) -> estado final -> linear -> gargalo de 4 -> linear -> repetido W vezes -> decodificador LSTM(32, 32) -> linear d.
  corrupção, perda Huber e escore: os mesmos do `tcn_torch` (máximo entre as tags do erro quadrado médio na janela normalizado pelo p99 da tag no treino, piso 0,1 x mediana, dividido pelo p99 do escore no treino).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .baselines import Escala
from .tcn_torch import W, janelas
from ..walk import inicios_de_mes


class RedeLSTM(nn.Module):
    def __init__(self, d, w=W, h=32, k=4):
        super().__init__()
        self.w = w; self.enc = nn.LSTM(d, h, batch_first=True); self.z = nn.Linear(h, k); self.up = nn.Linear(k, h); self.dec = nn.LSTM(h, h, batch_first=True); self.out = nn.Linear(h, d)

    def forward(self, x):
        _, (hn, _) = self.enc(x); z = self.z(hn[-1]); u = torch.relu(self.up(z)).unsqueeze(1).repeat(1, self.w, 1); o, _ = self.dec(u); return self.out(o)


class ModeloLSTM:
    def __init__(self, d, semente=0):
        self.d, self.semente = d, semente; torch.manual_seed(semente); self.net = RedeLSTM(d)

    def fit(self, X, ends, epocas=20, lote=512, lr=3e-3):
        self.escala = Escala().fit(X[ends]); Zn = self.escala.transform(X); J = torch.as_tensor(janelas(Zn, ends), dtype=torch.float32); g = torch.Generator().manual_seed(self.semente)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4); perda = nn.HuberLoss(delta=1.0); n = len(J); self.net.train()
        for _ in range(epocas):
            perm = torch.randperm(n, generator=g)
            for a in range(0, n, lote):
                x = J[perm[a:a + lote]]; m = torch.rand(x.shape, generator=g) < 0.15; xc = x.masked_fill(m, 0.0) + 0.05 * torch.randn(x.shape, generator=g)
                opt.zero_grad(); perda(self.net(xc), x).backward(); opt.step()
        self.net.eval(); e = self._erro2(J.numpy()); am = e[np.random.default_rng(self.semente).choice(len(e), size=min(len(e), 50000), replace=False)]
        p99 = np.quantile(am, 0.99, axis=0); self.sens = np.maximum(p99, 0.1 * np.median(p99)); self.p99s = float(np.quantile((am / self.sens).max(axis=1), 0.99)); return self

    @torch.no_grad()
    def _erro2(self, J):
        out = np.empty((len(J), self.d), dtype=np.float32)
        for a in range(0, len(J), 4096):
            x = torch.as_tensor(J[a:a + 4096], dtype=torch.float32); out[a:a + 4096] = ((self.net(x) - x) ** 2).mean(1).numpy()
        return out

    def score(self, X, ends):
        return (self._erro2(janelas(self.escala.transform(X), ends)) / self.sens).max(axis=1) / max(self.p99s, 1e-12)


def walk_lstm(g, tags, estavel, ini, fim, semente=0, n_max=20000, ep0=20) -> pd.Series:
    X = g[tags].to_numpy("float64"); idx = g.index; n = len(g); out = np.full(n, np.nan); meses = inicios_de_mes(ini, fim); ok = np.isfinite(X).all(axis=1)
    c = np.convolve(ok.astype(int), np.ones(W, int), "full")[:n]; okw = np.zeros(n, bool); okw[W - 1:] = c[W - 1:] == W
    ce = np.convolve((estavel & ok).astype(int), np.ones(W, int), "full")[:n]; estw = np.zeros(n, bool); estw[W - 1:] = ce[W - 1:] == W
    for k, m0 in enumerate(meses):
        m1 = meses[k + 1] if k + 1 < len(meses) else fim + pd.Timedelta("2min"); i0 = idx.searchsorted(m0); i1 = idx.searchsorted(m1); cand = np.flatnonzero(estw[:i0])
        if len(cand) < 5000:
            continue
        Xf = np.nan_to_num(X, nan=0.0); mod = ModeloLSTM(len(tags), semente + 1000 * k).fit(Xf, cand[-n_max:], ep0); e = np.flatnonzero(okw[i0:i1]) + i0
        if len(e):
            out[e] = mod.score(Xf, e)
    return pd.Series(out, index=idx)
