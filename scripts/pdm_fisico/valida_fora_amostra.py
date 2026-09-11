#!/usr/bin/env python3
"""VALIDACAO FORA DA AMOSTRA -- as paradas reais que nao sao trip.

Todo o ajuste (kb_lo, kb_hi, kv, frac, idade, ABS, refrat) foi feito olhando os 8
TRIPS. As PARADAS REAIS (>= 2 h) que nao terminaram em trip catalogado nunca
entraram em nenhuma escolha -- sao alvo independente.

Se o gatilho de dois niveis melhorar a antecedencia NELAS tambem, e evidencia de
que o mecanismo generaliza, e nao de que ajustamos 7 parametros a 8 rotulos.
Se nao melhorar, o ganho e sobreajuste e temos de dizer isso.

Compara: ponto publicado x ponto novo, sobre as paradas fora do conjunto de trips.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from combina_final import constroi
from plota_estilo_francisco import KB, KV, paradas_reais_2h

JAN = pd.Timedelta(hours=48)
paradas = paradas_reais_2h()

# paradas que NAO sao um dos 8 trips (nem a janela de 48 h em volta deles)
fora = []
for _, r in paradas.iterrows():
    if any(abs((r.ini - t).total_seconds()) < 48*3600 for t in alvo):
        continue
    if r.ini < idx[0] + JAN:
        continue
    fora.append(r.ini)
fora = pd.DatetimeIndex(sorted(fora))
print(f"paradas reais (>= 2 h) fora dos 8 trips e das suas janelas: {len(fora)}")
print(f"periodo: {fora[0]:%d/%m/%Y} a {fora[-1]:%d/%m/%Y}\n")

# ponto publicado
ON = partes(KB, KV); ns = sum(ON[c].astype(int) for c in SIN)
v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
al_pub = pos(v, ns, REFRAT_H, DUR_MIN, False)
al_novo = constroi(1.0, 1.7, 2.2, 96, 20)


def perfil(al, eventos):
    eps = AV.episodios(al)
    nasce, leads, depe = 0, [], 0
    for t in eventos:
        t0 = t - JAN
        ini = [a for a, _ in eps if t0 <= a <= t]
        if ini:
            nasce += 1; leads.append((t - max(ini)).total_seconds()/3600)
        elif any(a <= t and b >= t0 for a, b in eps):
            depe += 1
    banda = sum(1 for l in leads if 4.0 <= l <= 48.0)
    return nasce, depe, banda, (np.mean(leads) if leads else np.nan), leads


print("ALVO INDEPENDENTE -- as paradas reais que nunca entraram no ajuste")
print("=" * 92)
print(f"{'':<24}{'publicado':>14}{'novo (2 niveis)':>18}")
print("-" * 92)
a = perfil(al_pub, fora); b = perfil(al_novo, fora)
for rot, i, f in [("nasce na janela", 0, "{}/%d" % len(fora)),
                  ("(alarme ja de pe)", 1, "{}"),
                  ("na banda [4h,48h]", 2, "{}/%d" % len(fora)),
                  ("lead medio", 3, "{:.1f} h")]:
    print(f"{rot:<24}{f.format(a[i]):>14}{f.format(b[i]):>18}")

print("\n\nPOR PARADA")
print("=" * 92)
print(f"{'parada':>12} | {'publicado':>22} | {'novo':>22}")
print("-" * 92)
for t in fora:
    linha = []
    for al in (al_pub, al_novo):
        eps = AV.episodios(al); t0 = t - JAN
        ini = [x for x, _ in eps if t0 <= x <= t]
        if ini:
            l = (t - max(ini)).total_seconds()/3600
            linha.append(f"nasce {l:5.1f}h" + ("  [banda]" if 4 <= l <= 48 else ""))
        elif any(x <= t and y >= t0 for x, y in eps):
            linha.append("de pe")
        else:
            linha.append("--")
    m = "  <<<" if linha[0] != linha[1] else ""
    print(f"{t:%d/%m/%Y} | {linha[0]:>22} | {linha[1]:>22}{m}")
print("-" * 92)
print("  se o novo melhora aqui tambem, o ganho nao e sobreajuste aos 8 trips.")
