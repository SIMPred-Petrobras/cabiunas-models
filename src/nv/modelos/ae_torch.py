"""Autoencoder condicional, mascarado e com perda robusta (PyTorch), retreinado por mês com warm-start sobre o histórico estável anterior.

  entrada     x (d tags, escala robusta com piso) e contexto c (0, 3 ou 3 + d colunas); o contexto entra no codificador E no decodificador (autoencoder condicional)
  corrupção   cada entrada é zerada com p = 0,15 e recebe ruído gaussiano de 0,05 (a rede aprende a RELAÇÃO entre as tags, não a identidade)
  perda       Huber (delta = 1) na reconstrução do x limpo: o baseline tem anormalidade e a perda quadrática a decoraria
  escore      máximo, entre as tags, do erro quadrado normalizado pelo p99 do erro da tag no treino (com piso de 0,1 x mediana dos p99: o mesmo desenho do v2), dividido pelo p99 do próprio escore no treino
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn

from .baselines import Escala


class AE(nn.Module):
    def __init__(self, d: int, c: int, k: int = 4, h: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d + c, h), nn.GELU(), nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, k))
        self.dec = nn.Sequential(nn.Linear(k + c, h // 2), nn.GELU(), nn.Linear(h // 2, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x, c):
        z = self.enc(torch.cat([x, c], dim=1))
        return self.dec(torch.cat([z, c], dim=1))


def _tensor(a):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)


class ModeloAE:
    """Um modelo por mês. `anterior` (outro ModeloAE) dá o warm-start dos pesos."""

    def __init__(self, d: int, c: int, semente: int = 0, k: int = 4, h: int = 64):
        self.d, self.c, self.semente = d, c, semente
        torch.manual_seed(semente)
        self.net = AE(d, c, k, h)

    def fit(self, X: np.ndarray, C: np.ndarray, epocas: int, anterior: "ModeloAE | None" = None, lote: int = 1024, lr: float = 2e-3):
        self.escala = Escala().fit(X); Z = _tensor(self.escala.transform(X)); Ct = _tensor(C)
        if anterior is not None:
            self.net.load_state_dict(copy.deepcopy(anterior.net.state_dict()))
        g = torch.Generator().manual_seed(self.semente)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4); perda = nn.HuberLoss(delta=1.0)
        n = len(Z); self.net.train()
        for _ in range(epocas):
            perm = torch.randperm(n, generator=g)
            for a in range(0, n, lote):
                b = perm[a:a + lote]; x = Z[b]; c = Ct[b]
                m = (torch.rand(x.shape, generator=g) < 0.15)
                xc = x.masked_fill(m, 0.0) + 0.05 * torch.randn(x.shape, generator=g)
                opt.zero_grad(); perda(self.net(xc, c), x).backward(); opt.step()
        self.net.eval()
        e = self._erro2(X, C); amostra = e[np.random.default_rng(self.semente).choice(len(e), size=min(len(e), 50000), replace=False)]
        p99 = np.quantile(amostra, 0.99, axis=0); self.sens = np.maximum(p99, 0.1 * np.median(p99))
        self.p99s = float(np.quantile((amostra / self.sens).max(axis=1), 0.99))
        return self

    @torch.no_grad()
    def _erro2(self, X, C):
        Z = _tensor(self.escala.transform(X)); Ct = _tensor(C); out = np.empty((len(Z), self.d), dtype=np.float32)
        for a in range(0, len(Z), 8192):
            out[a:a + 8192] = ((self.net(Z[a:a + 8192], Ct[a:a + 8192]) - Z[a:a + 8192]) ** 2).numpy()
        return out

    def score(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        s = np.full(len(X), np.nan); ok = np.isfinite(X).all(axis=1) & np.isfinite(C).all(axis=1)
        if ok.any():
            s[ok] = (self._erro2(X[ok], C[ok]) / self.sens).max(axis=1) / max(self.p99s, 1e-12)
        return s
