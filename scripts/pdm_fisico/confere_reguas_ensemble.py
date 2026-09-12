#!/usr/bin/env python3
"""O ensemble comparou os dois sob a MESMA regua? Isola cada diferenca.

Sinal de alerta: no worktree dele, com a regua dele, da 8/8 · 2,88 FP/mes. No
ensemble_detectores.py saiu 7/8 · 2,584. Se fosse a mesma medida, bateria.

Candidatos a diferenca:
  1. MASCARA   -- a nossa exclui blackout de 6 h pos-partida, T5<=300 e t<T0;
                  a dele usa `operational_state` cru
  2. DENOMINADOR -- os nossos meses vem de mask.sum(); os dele de
                  compute_operational_period_days sobre o estado dele
  3. GRADE     -- a dele e de 30 s, a nossa de 2 min; o resample com max()
                  acende o bin inteiro de 2 min
  4. ALVO      -- os 8 eventos sao os mesmos? (a lista dele e dado, usamos a nossa)
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo, op

DIEGO = "/home/thallys/Documents/projeto-petrobras/wt-diego/canais/merged.parquet"
m = pd.read_parquet(DIEGO); m.index = m.index.tz_localize("UTC")

print("1. AS DUAS MASCARAS")
print("=" * 82)
est = m["operational_state"].astype(str)
dele_op = ~est.str.startswith("off")
comum = m.index.intersection(idx)
print(f"  janela dele : {m.index[0]:%d/%m/%Y} a {m.index[-1]:%d/%m/%Y}  ({len(m):,} @30s)")
print(f"  janela nossa: {idx[0]:%d/%m/%Y} a {idx[-1]:%d/%m/%Y}  ({len(idx):,} @2min)")
print(f"  interseccao : {len(comum):,} instantes\n")
n_op_d = float(dele_op.reindex(comum).mean())
n_op_n = float(mask.reindex(comum).mean())
print(f"  em operacao pela mascara DELE  : {100*n_op_d:5.1f}% da interseccao")
print(f"  em operacao pela mascara NOSSA : {100*n_op_n:5.1f}%")
print(f"  -> a nossa e {n_op_d/n_op_n:.2f}x mais restritiva "
      f"(exclui blackout de 6 h, T5<=300 e t<T0)")
so_dele = dele_op.reindex(comum) & ~mask.reindex(comum).fillna(False)
print(f"  instantes que SO a dele considera operacao: {int(so_dele.sum()):,} "
      f"({100*float(so_dele.mean()):.1f}%)")

print("\n2. OS DENOMINADORES")
print("=" * 82)
mes_n = float(mask.sum())*2/60.0/730.0
mes_d_grade = float(dele_op.sum())*0.5/60.0/730.0     # 30 s
print(f"  nosso  (mask, 2min)              : {mes_n:6.2f} meses de operacao vigiada")
print(f"  dele   (operational_state, 30s)  : {mes_d_grade:6.2f} meses")
print(f"  -> um mesmo numero de FP vira FP/mes {mes_d_grade/mes_n:.2f}x menor na regua dele")

print("\n3. O ALVO")
print("=" * 82)
print(f"  nossos {len(alvo)} eventos, dos quais dentro da janela dele: "
      f"{int(((alvo >= m.index[0]) & (alvo <= m.index[-1])).sum())}")
fora = alvo[(alvo < m.index[0]) | (alvo > m.index[-1])]
if len(fora):
    for t in fora:
        print(f"     FORA da janela dele: {t:%d/%m/%Y %H:%M}")
print(f"  a janela dele termina em {m.index[-1]:%d/%m/%Y} e a nossa em {idx[-1]:%d/%m/%Y}"
      f"  -> {(idx[-1]-m.index[-1]).days} dias a mais do nosso lado")

print("\n4. EFEITO DO RESAMPLE 30s -> 2min")
print("=" * 82)
from avalia import GAP_EP_H
ns = sum(m[c].astype(int) for c in ("temperatura", "vibracao", "oleo", "alarme"))
v = (ns >= 2)
g = (~v).cumsum()[v]
dur = v[v].groupby(g).transform("size") * 0.5
vf = v.copy(); vf[v] = dur >= 45.0
al_d = pd.Series(False, index=m.index); bloq = None
for a, b in AV.episodios(vf, gap_h=2.0):
    if bloq is not None and a <= bloq: continue
    al_d.loc[a:b] = True; bloq = b + pd.Timedelta(hours=48)
h30 = float(al_d.sum())*0.5/60.0
r2 = al_d.resample("2min").max().reindex(idx, fill_value=False).fillna(False).astype(bool)
h2 = float(r2.sum())*2/60.0
print(f"  horas de alarme na grade de 30 s : {h30:8.1f} h")
print(f"  apos resample para 2 min         : {h2:8.1f} h   (inflacao {h2/h30:.3f}x)")
print(f"  apos aplicar a NOSSA mascara     : {float((r2 & mask).sum())*2/60.0:8.1f} h")
