#!/usr/bin/env python3
"""MARGEM DE SEGURANCA CONSUMIDA -- a construcao correta.

A primeira tentativa (`proximidade_setpoint.py`) usou `valor/limite` e saturou:
sensores com faixas de operacao incomparaveis (PI_0319 com mediana -0,02 contra
limite 16,5) dominavam o max e o canal ficava cravado em 2,715.

A construcao com sentido fisico e a FRACAO DA MARGEM CONSUMIDA:

    alto:  m = (valor - tipico) / (limite - tipico)
    baixo: m = (tipico - valor) / (tipico - limite)

0 = operacao tipica, 1 = no limite de protecao. Comparavel entre sensores porque
normaliza pela margem que cada um realmente tem.

`tipico` = mediana do sensor em operacao mascarada, calculada de forma CAUSAL
(expanding, so com o passado) para nao vazar.

Sensores cuja mediana ja esta do lado errado do limite sao descartados -- indicam
mapeamento errado ou limite inativo, nao anomalia.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import g, mask, idx, alvo, op

LIM = pd.read_csv("limites_alarme.csv")
RNG = np.random.default_rng(20260911)
JAN = pd.Timedelta("48h")
MIN_MARGEM = 0.10          # o limite tem de estar a >=10% da mediana, senao nao ha margem

marg, descartados = {}, []
for _, r in LIM.iterrows():
    c = r["col"]
    if c not in g.columns:
        continue
    s = g[c].astype("float64").where(mask)
    tip = float(s.median())
    if not np.isfinite(tip):
        descartados.append((c, "sem mediana")); continue
    alto = r["HH"] if pd.notna(r["HH"]) else r["H"]
    baixo = r["LL"] if pd.notna(r["LL"]) else r["L"]
    partes = []
    if pd.notna(alto):
        if alto - tip > MIN_MARGEM * max(abs(tip), 1e-6):
            partes.append((s - tip) / (alto - tip))
        else:
            descartados.append((c, f"H/HH={alto:.2f} <= mediana {tip:.2f}"))
    if pd.notna(baixo):
        if tip - baixo > MIN_MARGEM * max(abs(tip), 1e-6):
            partes.append((tip - s) / (tip - baixo))
        else:
            descartados.append((c, f"L/LL={baixo:.2f} >= mediana {tip:.2f}"))
    if partes:
        marg[c] = pd.concat(partes, axis=1).max(axis=1)

M = pd.DataFrame(marg)
print(f"sensores com margem valida: {M.shape[1]} de {len(LIM)}")
print(f"descartados: {len(descartados)}")
for c, m in descartados[:8]:
    print(f"    {c[-22:]:>24}  {m}")

print(f"\n{'sensor':>24}{'tipico':>10}{'limite':>10}{'mediana m':>11}{'p99':>8}{'max':>8}")
print("-" * 76)
for c in M.columns:
    s = M[c].dropna(); r = LIM[LIM.col == c].iloc[0]
    lim = r["HH"] if pd.notna(r["HH"]) else (r["H"] if pd.notna(r["H"]) else
          (r["LL"] if pd.notna(r["LL"]) else r["L"]))
    print(f"{c[-22:]:>24}{float(g[c].where(mask).median()):10.2f}{lim:10.2f}"
          f"{s.median():11.3f}{s.quantile(.99):8.3f}{s.max():8.3f}")

MC = M.max(axis=1)
print(f"\nCANAL = max da margem consumida sobre {M.shape[1]} sensores")
print(f"  mediana {MC.median():.3f} | p90 {MC.quantile(.9):.3f} | "
      f"p99 {MC.quantile(.99):.3f} | max {MC.max():.3f}")

print("\n\nDIAGNOSTICO -- a margem consumida sobe antes dos trips?")
print("=" * 84)
P = MC.to_numpy(); ti = np.asarray(idx.astype("int64")); jan_us = int(JAN.value)//1000
def pico(t_us):
    lo = np.searchsorted(ti, t_us - jan_us, "left"); hi = np.searchsorted(ti, t_us, "right")
    return np.array([np.nanmax(P[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])
obs = pico(np.asarray([int(pd.Timestamp(t).value)//1000 for t in alvo]))
print(f"{'trip':>17}{'pico da margem nas 48h':>26}{'sensor responsavel':>26}")
print("-" * 84)
for t, v in zip(alvo, obs):
    w = M.loc[t - JAN:t]
    who = w.max().idxmax() if len(w) and w.max().notna().any() else "--"
    print(f"{t:%d/%m/%Y %H:%M}{v:26.3f}{str(who)[-24:]:>26}")
eleg = idx[(idx >= idx[0] + JAN) & op.to_numpy() & mask.to_numpy()]
cand = np.asarray(eleg.astype("int64"))
sort = RNG.choice(cand, size=(1000, len(alvo)))
nul = np.array([np.nanmedian(pico(sort[k])) for k in range(1000)])
om = float(np.nanmedian(obs)); p = float((nul >= om).mean())
print("-" * 84)
print(f"  mediana do pico nos 8 trips         : {om:.3f}")
print(f"  mediana em janelas sorteadas        : {np.nanmedian(nul):.3f} "
      f"(p10 {np.nanpercentile(nul,10):.3f}, p90 {np.nanpercentile(nul,90):.3f})")
print(f"  p = {p:.4f}" + ("  *** SOBE antes dos trips" if p < 0.05
                          else "  <-- nao distinguivel do acaso"))
