#!/usr/bin/env python3
"""Fecha o ponto de operacao do religamento, com margem nas duas direcoes.

frac = 0,030 nao e o argmax (0,040 e mais barato) -- e o meio da faixa ESTAVEL.
Em 0,050 o custo em horas salta de 8,8 para 32,0 h/mes: 0,040 esta encostado
nessa quebra. A faixa de 5/8 vai de 0,020 a 0,060; a de custo estavel, de 0,020
a 0,040; a intersecao e 0,020-0,040 e o meio e 0,030.
"""
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, EW, mask, idx, alvo
from publica_clearml import SIN, BASE, REFRAT_H, DUR_MIN
from corte_com_rearme import corta_rearma
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

KBv, KVv, DM, FRAC = 1.7, 2.2, 120, 0.030
paradas = paradas_reais_2h(); meses = float(mask.sum()) * 2 / 60.0 / 730.0
ON = partes(KBv, KVv); ns = sum(ON[c].astype(int) for c in SIN)
K = {"t": KBv, "p": KBv, "sp": KBv, "vb": KVv}
F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
v0 = (pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])).to_numpy()
JAN = pd.Timedelta(hours=48)


def serie(fr, rf):
    v = pd.Series(corta_rearma(v0, F, fr) if fr > 0 else v0, index=idx)
    return pos(v, ns, rf, DM, False)


def mede(al):
    eps = AV.episodios(al); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
    return m["det"], mi["det"], nfp/meses, h/meses, m["lead_med"], mi["lead_med"], len(eps)


print(f"PLATO EM REFRATARIO no frac escolhido ({FRAC})")
print("=" * 68)
print(f"{'refrat':>7} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8}")
print("-" * 68)
for rf in [24, 36, 48, 60, 72, 84, 96, 120]:
    det, di, fp, h, *_ = mede(serie(FRAC, rf))
    print(f"{rf:6d}h {det:5d}/8 {di:7d}/8 {fp:9.3f} {h:8.1f}"
          + ("  <<< quebra" if det < 8 else ("  ok" if di >= 5 else "")))

RF = 72
al_novo, al_pub = serie(FRAC, RF), serie(0.0, REFRAT_H)
print(f"\n\nO PONTO: frac={FRAC} · refrat={RF}h · dur={DM}min · k={KBv}/{KVv} · portao=sim")
print("=" * 92)
print(f"{'':<22} {'publicado':>14} {'com religamento':>18}")
print("-" * 92)
a, b = mede(al_pub), mede(al_novo)
for rot, i, f in [("deteccao (nossa)", 0, "{}/8"), ("deteccao (estrita)", 1, "{}/8"),
                  ("FP/mes (regra C)", 2, "{:.3f}"), ("h/mes", 3, "{:.1f}"),
                  ("lead medio", 4, "{:.1f} h"), ("lead (inicio real)", 5, "{:.1f} h"),
                  ("episodios", 6, "{}")]:
    print(f"{rot:<22} {f.format(a[i]):>14} {f.format(b[i]):>18}")

print(f"\n\nPOR EVENTO")
print("=" * 92)
print(f"{'evento':>12} | {'publicado':>28} | {'com religamento':>28}")
print("-" * 92)
for t in alvo:
    t0 = t - JAN; cols = []
    for al in (al_pub, al_novo):
        eps = AV.episodios(al)
        ini = [x for x, y in eps if t0 <= x <= t]
        dep = [(x, y) for x, y in eps if x <= t and y >= t0]
        cols.append(f"NASCE  lead {(t-max(ini)).total_seconds()/3600:5.1f}h" if ini
                    else (f"de pe  ep de {(dep[0][1]-dep[0][0]).total_seconds()/3600:5.0f}h"
                          if dep else "nao detecta"))
    print(f"{t:%d/%m/%Y} | {cols[0]:>28} | {cols[1]:>28}"
          + ("  <<<" if cols[0] != cols[1] else ""))
