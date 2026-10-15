"""Previsor de série como modelo de RESÍDUO por família (EXP 15).

Para cada instante i, uma janela de L amostras que TERMINA em i - H prevê o vetor de tags em i (H = passos de antecedência; 15 passos = 30 min).
O escore é o do canal do v2: máximo, entre as tags, do erro quadrado da previsão (na escala robusta) normalizado pelo p99 do erro da tag no treino (piso 0,1 x mediana dos p99), dividido pelo p99 do próprio escore no treino.
  persist  previsão = última amostra da janela (o controle trivial: o resíduo é a variação em H passos)
  ar       regressão ridge linear sobre a janela achatada (o autorregressivo)
  gru      GRU(32) sobre a janela, com a previsão = última amostra + correção aprendida (parte da persistência)
Treino mensal causal nas últimas `n_max` amostras-alvo cuja janela inteira (de i - H - L + 1 a i) é estável e finita; o escore vale onde a janela é finita (a vigilância do v2 já exclui as 6 h pós-partida, que cobrem H + L até 3,5 h).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge

from .baselines import Escala
from ..walk import inicios_de_mes


class _GRU(nn.Module):
    def __init__(self, d, h=32):
        super().__init__(); self.g = nn.GRU(d, h, batch_first=True); self.o = nn.Linear(h, d)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, x):
        o, _ = self.g(x); return x[:, -1, :] + self.o(o[:, -1, :])


def _janelas(Z, alvo, L, H):
    """Janelas (len(alvo), L, d) terminando em alvo - H; alvos >= H + L - 1."""
    V = sliding_window_view(Z, L, axis=0)  # (n-L+1, d, L), a janela que termina em j está em V[j-L+1]
    return np.ascontiguousarray(V[np.asarray(alvo) - H - (L - 1)].transpose(0, 2, 1))


class Previsor:
    def __init__(self, kind, d, L=15, H=15, semente=0, epocas=12):
        self.kind, self.d, self.L, self.H, self.semente, self.epocas = kind, d, L, H, semente, epocas

    def fit(self, Zn, alvo):
        J = _janelas(Zn, alvo, self.L, self.H); Y = Zn[alvo]
        if self.kind == "ar":
            self.m = Ridge(alpha=1.0).fit(J.reshape(len(J), -1), Y)
        elif self.kind == "gru":
            torch.manual_seed(self.semente); self.net = _GRU(self.d); Jt = torch.as_tensor(J, dtype=torch.float32); Yt = torch.as_tensor(Y, dtype=torch.float32)
            g = torch.Generator().manual_seed(self.semente); opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-4); perda = nn.HuberLoss(delta=1.0); self.net.train()
            for _ in range(self.epocas):
                perm = torch.randperm(len(Jt), generator=g)
                for a in range(0, len(Jt), 512):
                    b = perm[a:a + 512]; opt.zero_grad(); perda(self.net(Jt[b]), Yt[b]).backward(); opt.step()
            self.net.eval()
        e = self._erro2(Zn, alvo); self.sens = np.maximum(np.quantile(e, 0.99, axis=0), 0.1 * np.median(np.quantile(e, 0.99, axis=0)))
        self.p99s = float(np.quantile((e / self.sens).max(axis=1), 0.99)); return self

    def _prev(self, J):
        if self.kind == "persist":
            return J[:, -1, :]
        if self.kind == "ar":
            return self.m.predict(J.reshape(len(J), -1))
        with torch.no_grad():
            return np.concatenate([self.net(torch.as_tensor(J[a:a + 4096], dtype=torch.float32)).numpy() for a in range(0, len(J), 4096)])

    def _erro2(self, Zn, alvo):
        out = np.empty((len(alvo), self.d), dtype=np.float32)
        for a in range(0, len(alvo), 20000):
            al = alvo[a:a + 20000]; out[a:a + 20000] = (Zn[al] - self._prev(_janelas(Zn, al, self.L, self.H))) ** 2
        return out

    def score(self, Zn, alvo):
        return (self._erro2(Zn, alvo) / self.sens).max(axis=1) / max(self.p99s, 1e-12)


def walk_previsor(g, tags, estavel, ini, fim, kind, H, L=15, semente=0, n_max=20000, epocas=12) -> pd.Series:
    X = g[tags].to_numpy("float64"); idx = g.index; n = len(g); out = np.full(n, np.nan); meses = inicios_de_mes(ini, fim)
    fin = np.isfinite(X).all(axis=1); W = H + L
    def janela_ok(m):  # todas as amostras de i - W + 1 a i satisfazem m
        c = np.convolve(m.astype(int), np.ones(W, int), "full")[:n]; r = np.zeros(n, bool); r[W - 1:] = c[W - 1:] == W; return r
    ok_fin = janela_ok(fin); ok_est = janela_ok(fin & estavel)
    for k, m0 in enumerate(meses):
        m1 = meses[k + 1] if k + 1 < len(meses) else fim + pd.Timedelta("2min"); i0 = idx.searchsorted(m0); i1 = idx.searchsorted(m1)
        cand = np.flatnonzero(ok_est[:i0])
        if len(cand) < 5000:
            continue
        tr = cand[-n_max:]; esc = Escala().fit(X[tr]); Zn = np.nan_to_num(esc.transform(X), nan=0.0)
        mod = Previsor(kind, len(tags), L, H, semente + 1000 * k, epocas).fit(Zn, tr)
        e = np.flatnonzero(ok_fin[i0:i1]) + i0
        if len(e):
            out[e] = mod.score(Zn, e)
    return pd.Series(out, index=idx)
