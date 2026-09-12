#!/usr/bin/env python3
"""Revisitar o ALVO -- o gargalo do projeto sao 8 rotulos, nao o algoritmo.

O alvo e derivado por regra (`verdade.py`): queda de RUNNING_A com parada >= 2 h
coincidindo com alarme de NIVEL em [-1h, +30min]. O regex NIVEL pega TRIP,
Mt.Alta, M.Baixa -- ou seja, **so o segundo estagio de protecao**.

Tres motivos para revisitar agora:
 1. ha caso conhecido excluido: 24/11/2025, parada de 43 h com PAL_6240339
    (pressao baixa no header de oleo, PRIMEIRO estagio)
 2. 3 dos nossos 6 FP tem corroboracao da planta, um com 5 ativacoes de
    TC382_03_A (temperatura de mancal)
 3. PAL_6240339 e uma das tres unicas tags com consumo de margem significativo
    antes dos trips (3,6x, passa Bonferroni)

E agora temos a planilha de setpoints, que nao existia quando o alvo foi
definido: da para confrontar a regra com a protecao real da maquina.

Este script NAO muda o alvo. Levanta os candidatos e mostra a evidencia de cada.
"""
from __future__ import annotations
import re
import numpy as np, pandas as pd
from pos_processamento import g, mask, idx, alvo, op
from fp_alarmes import catalogo
from plota_estilo_francisco import paradas_reais_2h

NIVEL = re.compile(r"TRIP|Mt\.?\s?Alta|M\.?\s?Alta|Mt\.?\s?Bx|M\.?\s?Bx|Mt\.?\s?Baixa|M\.?\s?Baixa", re.I)
cat = catalogo(idx)
cat["nivel2"] = cat["Descrição Alarme"].astype(str).str.contains(NIVEL)
paradas = paradas_reais_2h()
LIM = pd.read_csv("limites_alarme.csv")

print(f"paradas reais (>= 2 h) no periodo: {len(paradas)}")
print(f"alvo atual: {len(alvo)} eventos\n")
print("TODAS AS PARADAS >= 12 h, COM A EVIDENCIA DE ALARME EM [-1h, +30min]")
print("=" * 116)
print(f"{'parada':>17}{'dur':>8}{'no alvo?':>10}  alarmes de NIVEL (2o estagio) | "
      f"alarmes de 1o estagio")
print("-" * 116)
cands = []
for _, r in paradas.iterrows():
    if r.dur_h < 12: continue
    a, b = r.ini - pd.Timedelta("1h"), r.ini + pd.Timedelta("30min")
    j = cat[(cat.t >= a) & (cat.t <= b)]
    n2 = sorted(set(j[j.nivel2]["Tag Alarme"].astype(str)))
    n1 = sorted(set(j[~j.nivel2]["Tag Alarme"].astype(str)))
    no_alvo = any(abs((r.ini - t).total_seconds()) < 3*3600 for t in alvo)
    if not no_alvo and (n1 or n2):
        cands.append((r.ini, r.dur_h, n1, n2))
    print(f"{r.ini:%d/%m/%Y %H:%M}{r.dur_h:7.1f}h{('SIM' if no_alvo else 'nao'):>10}  "
          f"{','.join(n2)[:34]:<34} | {','.join(n1)[:36]}")
print("-" * 116)

print(f"\n\nCANDIDATOS A EVENTO -- parada >= 12 h, FORA do alvo, COM alarme")
print("=" * 116)
if not cands:
    print("  nenhum")
for t, dur, n1, n2 in cands:
    print(f"\n  {t:%d/%m/%Y %H:%M}  parada de {dur:.1f} h")
    for tg in (n2 + n1):
        s = cat[cat["Tag Alarme"].astype(str) == tg]
        d = str(s["Descrição Alarme"].dropna().iloc[0])[:52] if s["Descrição Alarme"].notna().any() else ""
        lin = LIM[LIM.tag.astype(str) == tg]
        lim = ""
        if len(lin):
            r_ = lin.iloc[0]
            lim = "  [setpoint: " + " ".join(
                f"{k}={r_[k]:.2f}" for k in ("LL","L","H","HH") if pd.notna(r_[k])) + "]"
        estagio = "2o (conta)" if NIVEL.search(d) else "1o (NAO conta)"
        print(f"     {tg:>16} {estagio:>15}  {d}{lim}")
