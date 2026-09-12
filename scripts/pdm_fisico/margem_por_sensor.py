#!/usr/bin/env python3
"""Margem consumida SENSOR A SENSOR -- qual deles carrega precursor?

O max sobre os 36 satura em sensores bimodais de utilidade (PI_0319 = gas do
motor de partida, perto de zero em operacao e disparando na partida). Testar
cada sensor isoladamente evita isso e responde a pergunta certa: existe ALGUM
sensor cuja margem de seguranca e consumida antes dos trips mais do que por acaso?

Nulo por permutacao, p corrigido por Bonferroni sobre os 36 testes.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import g, mask, idx, alvo, op

LIM = pd.read_csv("limites_alarme.csv")
RNG = np.random.default_rng(20260911)
JAN = pd.Timedelta("48h"); NS = 2000
MINM = 0.10

marg = {}
for _, r in LIM.iterrows():
    c = r["col"]
    if c not in g.columns: continue
    s = g[c].astype("float64").where(mask); tip = float(s.median())
    if not np.isfinite(tip): continue
    alto = r["HH"] if pd.notna(r["HH"]) else r["H"]
    baixo = r["LL"] if pd.notna(r["LL"]) else r["L"]
    p = []
    if pd.notna(alto) and alto - tip > MINM*max(abs(tip),1e-6): p.append((s-tip)/(alto-tip))
    if pd.notna(baixo) and tip - baixo > MINM*max(abs(tip),1e-6): p.append((tip-s)/(tip-baixo))
    if p: marg[c] = pd.concat(p, axis=1).max(axis=1)

ti = np.asarray(idx.astype("int64")); jan_us = int(JAN.value)//1000
eleg = idx[(idx >= idx[0] + JAN) & op.to_numpy() & mask.to_numpy()]
sort = RNG.choice(np.asarray(eleg.astype("int64")), size=(NS, len(alvo)))
t_obs = np.asarray([int(pd.Timestamp(t).value)//1000 for t in alvo])

def picos(P, T):
    lo = np.searchsorted(ti, T - jan_us, "left"); hi = np.searchsorted(ti, T, "right")
    return np.array([np.nanmax(P[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])

lin = []
for c, s in marg.items():
    P = s.to_numpy()
    o = float(np.nanmedian(picos(P, t_obs)))
    nul = np.array([np.nanmedian(picos(P, sort[k])) for k in range(NS)])
    p = float(np.nanmean(nul >= o))
    lin.append(dict(sensor=c, obs=o, nulo=float(np.nanmedian(nul)),
                    raz=o/np.nanmedian(nul) if np.nanmedian(nul) else np.nan, p=p))
D = pd.DataFrame(lin).sort_values("p")
D["p_bonf"] = (D.p * len(D)).clip(upper=1.0)

print(f"MARGEM CONSUMIDA ANTES DOS TRIPS -- {len(D)} sensores, Bonferroni sobre {len(D)}")
print("=" * 92)
print(f"{'sensor':>24}{'pico nos trips':>16}{'nulo':>10}{'razao':>9}{'p':>9}{'p_bonf':>10}")
print("-" * 92)
for _, r in D.head(14).iterrows():
    fl = "  ***" if r.p_bonf < 0.05 else ("  *" if r.p < 0.05 else "")
    print(f"{r.sensor[-22:]:>24}{r.obs:16.3f}{r.nulo:10.3f}{r.raz:9.2f}x{r.p:9.4f}"
          f"{r.p_bonf:10.3f}{fl}")
print("-" * 92)
sig = D[D.p_bonf < 0.05]
print(f"  sensores significativos apos Bonferroni: {len(sig)} de {len(D)}")
if len(sig):
    print("   ", ", ".join(sig.sensor))
else:
    print("    NENHUM -- a margem de seguranca nao e consumida antes dos trips de")
    print("    forma distinguivel do acaso, em nenhum dos 36 sensores.")
