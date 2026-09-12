#!/usr/bin/env python3
"""PROXIMIDADE AO SETPOINT -- informacao de natureza nova no detector.

DE ONDE VEM. Planilha `Turbinas - Limites Alertas - TAGs da variaveis (2).xlsx`,
aba TS-C-33003A, trazida pelo usuario em 11/09/2026. Tem LL/L/H/HH por tag, e
**os 36 sensores do nosso detector estao todos cobertos**.

POR QUE E DIFERENTE DE TUDO QUE JA TEMOS. Os nossos quatro canais medem desvio
da NORMALIDADE APRENDIDA -- dizem "isto esta incomum". A proximidade ao setpoint
mede desvio do LIMITE DE PROTECAO PROJETADO -- diz "isto esta se aproximando da
acao protetiva". Referencia diferente, e carrega o conhecimento de engenharia
que nao esta em lugar nenhum do detector hoje.

E resolve o problema que matou o canal de alarme: medido em 11/09, o alarme
TAH_6240305 dispara com lead MEDIANO 0,0 h -- o cruzamento E o evento. A
proximidade, ao contrario, e continua e precede o cruzamento por construcao.

Este script so DIAGNOSTICA: a proximidade sobe antes dos trips, contra o que
subiria por acaso? So constroi canal se a resposta for sim.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import g, mask, idx, alvo, op

LIM = pd.read_csv("limites_alarme.csv")
RNG = np.random.default_rng(20260911)
N = 20000
JAN = pd.Timedelta("48h")

# fracao do caminho ate o limite de PROTECAO (HH/LL quando existe, senao H/L)
frac = {}
for _, r in LIM.iterrows():
    c = r["col"]
    if c not in g.columns:
        continue
    s = g[c].astype("float64")
    alto = r["HH"] if pd.notna(r["HH"]) else r["H"]
    baixo = r["LL"] if pd.notna(r["LL"]) else r["L"]
    f = []
    if pd.notna(alto) and alto != 0:
        f.append(s / alto)
    if pd.notna(baixo) and baixo != 0:
        f.append(baixo / s.where(s.abs() > 1e-6))
    if f:
        frac[c] = pd.concat(f, axis=1).max(axis=1)
F = pd.DataFrame(frac).where(mask)
print(f"sensores com fracao calculada: {F.shape[1]}")
print(f"\n{'sensor':>24}{'mediana':>10}{'p99':>9}{'max':>9}  (1,00 = no limite de protecao)")
print("-" * 78)
for c in F.columns:
    s = F[c].dropna()
    fl = "  <<< opera perto do limite" if s.median() > 0.85 else ""
    print(f"{c[-22:]:>24}{s.median():10.3f}{s.quantile(.99):9.3f}{s.max():9.3f}{fl}")

PROX = F.max(axis=1)
print(f"\nCANAL 'proximidade' = max sobre os {F.shape[1]} sensores")
print(f"  mediana {PROX.median():.3f} | p90 {PROX.quantile(.9):.3f} | "
      f"p99 {PROX.quantile(.99):.3f} | max {PROX.max():.3f}")

print("\n\nDIAGNOSTICO -- a proximidade sobe antes dos trips?")
print("=" * 92)
eleg = idx[(idx >= idx[0] + JAN) & op.to_numpy() & mask.to_numpy()]
cand = np.asarray(eleg.astype("int64"))
sort = RNG.choice(cand, size=(N, len(alvo)))
P = PROX.to_numpy(); ti = np.asarray(idx.astype("int64")); jan_us = int(JAN.value)//1000

def pico(t_us):
    lo = np.searchsorted(ti, t_us - jan_us, "left"); hi = np.searchsorted(ti, t_us, "right")
    return np.array([np.nanmax(P[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])

obs = pico(np.asarray([int(pd.Timestamp(t).value)//1000 for t in alvo]))
print(f"{'trip':>17}{'pico de proximidade nas 48h':>30}")
print("-" * 92)
for t, v in zip(alvo, obs):
    print(f"{t:%d/%m/%Y %H:%M}{v:30.3f}")
nul = np.array([np.nanmax(pico(sort[k])) for k in range(0, N, 20)])
obs_max = float(np.nanmax(obs)); obs_med = float(np.nanmedian(obs))
nul_med = np.array([np.nanmedian(pico(sort[k])) for k in range(0, N, 20)])
print("-" * 92)
print(f"  mediana do pico nos 8 trips        : {obs_med:.3f}")
print(f"  mediana do pico em janelas sorteadas: {np.nanmedian(nul_med):.3f}"
      f"  (p10 {np.nanpercentile(nul_med,10):.3f}, p90 {np.nanpercentile(nul_med,90):.3f})")
p = float((nul_med >= obs_med).mean())
print(f"  p = {p:.4f}" + ("  *** a proximidade SOBE antes dos trips"
                          if p < 0.05 else "  <-- nao distinguivel do acaso"))
