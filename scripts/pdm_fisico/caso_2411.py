#!/usr/bin/env python3
"""O 24/11/2025 e um nono evento? E o nosso detector o pega?

Unico candidato entre 14 paradas longas fora do alvo que traz alarme de SAUDE
MECANICA (PAL_6240339, pressao baixa no header de oleo lubrificante) e nao de
utilidade/gas. E NADA nesta sessao foi ajustado nele -- se for evento real, e
teste fora da amostra honesto para tudo que construimos.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import g, EW, mask, idx, alvo, sel
from publica_clearml import SIN, BASE
from escalada_por_idade import quebra_idade
from checa_degenerado import pos_dur_esc
from plota_estilo_francisco import alarme, KB, KV
from confirma_vizinhanca import canal, P
from fp_alarmes import catalogo

T = pd.Timestamp("2025-11-24 09:22", tz="UTC")
JAN = pd.Timedelta("48h")
cat = catalogo(idx)
LIM = pd.read_csv("limites_alarme.csv")

print(f"CASO {T:%d/%m/%Y %H:%M} -- parada de 43,0 h")
print("=" * 92)
print("\n1. O SENSOR DE OLEO REALMENTE CAI?  (PI_0339, tipico 3,37 | L 2,88 | LL 1,78)")
print("-" * 92)
s = g["954005_624_PI_0339"].astype("float64")
for h in (72, 48, 24, 12, 6, 3, 1):
    w = s.loc[T - pd.Timedelta(hours=h):T].dropna()
    if len(w):
        print(f"  ultimas {h:3d}h: min {w.min():6.2f} | mediana {w.median():6.2f} | "
              f"max {w.max():6.2f}" + ("   <<< abaixo de L=2,88" if w.min() < 2.88 else ""))

print("\n2. ALARMES NA JANELA DE 48 h")
print("-" * 92)
j = cat[(cat.t >= T - JAN) & (cat.t <= T)]
for tg, n in j["Tag Alarme"].value_counts().items():
    d = str(cat[cat["Tag Alarme"] == tg]["Descrição Alarme"].dropna().iloc[0])[:46]
    ult = j[j["Tag Alarme"] == tg].t.max()
    print(f"  {str(tg):>16} x{n:<3} ultimo {(T-ult).total_seconds()/3600:5.1f}h antes  {d}")

print("\n3. O QUE OS NOSSOS 4 CANAIS FAZEM NA JANELA")
print("-" * 92)
K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
print(f"  {'canal':>6}{'pico E/limiar':>16}{'media':>10}{'horas aceso':>14}")
from pos_processamento import partes
ON = partes(KB, KV)
for c in SIN:
    r = EW[c].where(mask).loc[T-JAN:T]/(BASE[c]*K[c])
    ac = ON[c].loc[T-JAN:T]
    print(f"  {c:>6}{float(r.max()) if len(r.dropna()) else np.nan:16.2f}"
          f"{float(r.mean()) if len(r.dropna()) else np.nan:10.2f}"
          f"{float(ac.sum())*2/60:13.1f}h")

print("\n4. O DETECTOR DISPARA?")
print("-" * 92)
p = {k: (dict(P["lo"]) if k == "lo" else P[k]) for k in P}; p["frac"] = 0.0
A = {c: canal(c, p["lo"][c]) for c in SIN}
B = {c: canal(c, p["hi"] if c != "vb" else p["kvhi"]) for c in SIN}
KH = {"t": p["hi"], "p": p["hi"], "sp": p["hi"], "vb": p["kvhi"]}
F = pd.concat([EW[c].where(mask)/(BASE[c]*KH[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
vA = pd.Series(sum(A[c].astype(int) for c in SIN) >= 3, index=idx) & mask
vB = pd.Series(sum(B[c].astype(int) for c in SIN) >= 2, index=idx) & mask & (B["sp"] | B["vb"])
novo = pos_dur_esc(pd.Series(quebra_idade((vA|vB).to_numpy(), F, p["abs_"], p["idade"]),
                             index=idx), p["refrat"], F, p["abs_"], p["idade"], p["dur_esc"])
for rot, al in (("publicado", alarme()), ("ponto novo", novo)):
    eps = AV.episodios(al)
    i = [a for a, _ in eps if T - JAN <= a <= T]
    if i:
        a = max(i); b = [y for x, y in eps if x == a][0]
        l = (T-a).total_seconds()/3600
        print(f"  {rot:>12}: NASCE {l:.1f}h antes, dura {(b-a).total_seconds()/60:.0f}min"
              + ("   [banda acionavel]" if l >= 4 else "   (lead < 4h)"))
    else:
        d = [(x, y) for x, y in eps if x <= T and y >= T-JAN]
        print(f"  {rot:>12}: " + (f"alarme de pe ha {(T-d[0][0]).total_seconds()/3600:.0f}h"
                                  if d else "NAO detecta"))
print(f"\n  operacao vigiada na janela de 48h: "
      f"{100*float(mask.loc[T-JAN:T].mean()):.0f}%")
