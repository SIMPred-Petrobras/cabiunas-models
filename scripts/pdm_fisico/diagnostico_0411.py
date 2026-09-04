#!/usr/bin/env python3
"""Por que 04/11/2025 e fragil -- e o que ele precisaria para deixar de ser.

O PROBLEMA. O ponto de producao detecta os 8 trips, mas sob leave-one-event-out o
numero honesto e 7/8: 04/11/2025 e detectado por apenas 24% das configuracoes
dentro do orcamento de falso positivo, contra 59-76% dos outros sete. Nao e que o
detector nao veja o evento -- e que a deteccao depende do ponto exato de operacao.
Tornar essa deteccao robusta e a unica melhoria que sobra sem mexer no alvo.

HIPOTESE A TESTAR. Duas coisas conspiram nesse evento:
  (1) OBSERVABILIDADE -- a maquina operou pouco tempo nas 48 h antes do trip, e o
      blackout de 6 h pos-religamento come parte do que sobrou;
  (2) EVIDENCIA MARGINAL -- os canais que votam cruzam o limiar por pouco.
Se (1) domina, um blackout mais curto ou dependente de canal recupera o evento.
Se (2) domina, o problema e de sinal e nao ha o que fazer com estes quatro.

Este script so MEDE. Nao propoe regra nova antes de saber qual das duas manda.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import EW, BASE, mask, idx, alvo, op, part, reset
from publica_clearml import SIN, K, KAPPA, H_CUSUM, SUSTAIN, GRID, BLACKOUT
from blackout_curto import cusum

ALVO = pd.Timestamp("2025-11-04 06:22", tz="UTC")
JAN = pd.Timedelta(hours=48)
PASSO_H = pd.Timedelta(GRID).total_seconds() / 3600.0

# reconstroi a mascara em partes, para saber o que cada filtro tirou
g = pd.read_parquet("grade2min.parquet")
quente = g["T5_AVG_A"] > 300
n_bl = int(pd.Timedelta(BLACKOUT) / pd.Timedelta(GRID))
blk = part.rolling(n_bl, min_periods=1).max().astype(bool)

jan = (idx >= ALVO - JAN) & (idx <= ALVO)
h = lambda s: float(np.asarray(s)[jan].sum()) * PASSO_H

print("=" * 88)
print(f"OBSERVABILIDADE nas 48 h antes de {ALVO:%d/%m/%Y %H:%M}")
print("=" * 88)
print(f"  janela de avaliacao ................. {48.0:6.1f} h")
print(f"  maquina rodando (RUNNING_A) ......... {h(op):6.1f} h")
print(f"  rodando E quente (T5 > 300) ......... {h(op & quente):6.1f} h")
print(f"  descontado o blackout de 6 h ........ {h(mask):6.1f} h   <- o que o detector ve")
print(f"  perdido so para o blackout .......... {h((op & quente) & blk):6.1f} h")
n_part = int(part.to_numpy()[jan].sum())
print(f"  religamentos dentro da janela ....... {n_part}")

print("\n" + "=" * 88)
print("EVIDENCIA -- quanto cada canal cruza o limiar, na janela visivel")
print("=" * 88)
print(f"{'canal':>6} {'limiar':>8} {'pico E/thr':>11} {'mediana E/thr':>14} "
      f"{'h acima':>9} {'como acende':>28}")
for c in SIN:
    thr = BASE[c] * K[c]
    E = EW[c].where(mask)
    r = (E.to_numpy()[jan] / thr)
    r = r[np.isfinite(r)]
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN) & mask
    cu = pd.Series(cusum(((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx) & mask
    d, u = deg.to_numpy()[jan], cu.to_numpy()[jan]
    if d.any() and u.any():   modo = "degrau e CUSUM"
    elif d.any():             modo = "so degrau"
    elif u.any():             modo = "so CUSUM"
    else:                     modo = "-- apagado"
    pico = np.nanmax(r) if len(r) else np.nan
    med = np.nanmedian(r) if len(r) else np.nan
    h_aceso = float((d | u).sum()) * PASSO_H   # d e u ja vem fatiados por `jan`
    print(f"{c:>6} {thr:8.2f} {pico:10.2f}x {med:13.2f}x {h_aceso:8.1f} h {modo:>28}")

print("\n" + "=" * 88)
print("MARGEM -- de quanto o limiar poderia subir antes de o canal apagar")
print("=" * 88)
for c in SIN:
    thr = BASE[c] * K[c]
    E = EW[c].where(mask)
    r = E.to_numpy()[jan] / thr
    r = r[np.isfinite(r)]
    if not len(r):
        continue
    # quanto tempo ficaria acima se o limiar fosse multiplicado por f
    linha = []
    for f in (1.0, 1.25, 1.5, 2.0):
        linha.append(f"{f:.2f}x -> {float((r >= f).sum()) * PASSO_H:5.1f} h")
    print(f"  {c:>4}   " + "   ".join(linha))

print("\n" + "=" * 88)
print("COMPARACAO -- observabilidade dos 8 alvos, para situar o 04/11")
print("=" * 88)
print(f"{'evento':>12} {'rodando':>9} {'visivel':>9} {'perdido p/ blackout':>21} {'religamentos':>13}")
for t in alvo:
    j = (idx >= t - JAN) & (idx <= t)
    hh = lambda s: float(np.asarray(s)[j].sum()) * PASSO_H
    marca = "   <<< o fragil" if abs((t - ALVO).total_seconds()) < 3600 else ""
    print(f"{t:%d/%m/%Y} {hh(op):8.1f} h {hh(mask):8.1f} h {hh((op & quente) & blk):20.1f} h "
          f"{int(part.to_numpy()[j].sum()):13d}{marca}")
