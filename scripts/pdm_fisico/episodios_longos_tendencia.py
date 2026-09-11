#!/usr/bin/env python3
"""Os 4 episodios longos sao degradacao lenta real ou alarme travado?

Se o escore SOBE ao longo das centenas de horas, o alarme permanente e a saida
correta para um precursor lento -- e o que esta errado e a saida ser binaria, nao
o detector. Se o escore fica TRAVADO logo acima do limiar, e alarme preso e a
deteccao dos 4 eventos e coincidencia de estar aceso.

Teste: regressao do escore normalizado (E/limiar) contra o tempo dentro do
episodio, e comparacao do primeiro com o ultimo quinto.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, EW, mask, idx, alvo
from publica_clearml import SIN, BASE
from plota_estilo_francisco import alarme, KB, KV

K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
ON = partes(KB, KV)
al = alarme()
eps = AV.episodios(al)
JAN = pd.Timedelta(hours=48)

# razao E/limiar por canal: >1 e "aceso"
RAZ = {c: (EW[c].where(mask) / (BASE[c] * K[c])) for c in SIN}

print("OS EPISODIOS QUE COBREM OS 8 EVENTOS -- degradacao ou alarme travado?")
print("=" * 104)
print(f"{'evento':>11} {'inicio do ep':>17} {'dur':>8} {'lead':>8} | "
      f"{'canal':>5} {'1o quinto':>10} {'ult quinto':>11} {'inclinacao':>12} {'r':>7}")
print("-" * 104)
for t in alvo:
    t0 = t - JAN
    cand = [(a, b) for a, b in eps if a <= t and b >= t0]
    if not cand:
        continue
    a, b = cand[0]
    dur = (b - a).total_seconds() / 3600
    lead = (t - a).total_seconds() / 3600
    marca = " <<<" if lead > 48 else ""
    prim = True
    for c in SIN:
        s = RAZ[c].loc[a:b].dropna()
        if len(s) < 20 or not bool(ON[c].loc[a:b].any()):
            continue
        n5 = max(len(s) // 5, 1)
        q1, q5 = float(s.iloc[:n5].mean()), float(s.iloc[-n5:].mean())
        x = (s.index - s.index[0]).total_seconds().to_numpy() / 3600
        y = s.to_numpy()
        incl = float(np.polyfit(x, y, 1)[0]) * 24          # por dia
        r = float(np.corrcoef(x, y)[0, 1])
        rot = f"{t:%d/%m/%Y} {a:%d/%m %H:%M} {dur:7.1f}h {lead:7.1f}h" if prim else " " * 45
        print(f"{rot} | {c:>5} {q1:10.2f} {q5:11.2f} {incl:+11.3f}/d {r:+7.2f}"
              + (marca if prim else ""))
        prim = False
    if prim:
        print(f"{t:%d/%m/%Y} {a:%d/%m %H:%M} {dur:7.1f}h {lead:7.1f}h | (nenhum canal)")
print("-" * 104)
print("  1o/ult quinto = media de E/limiar; >1 e aceso. inclinacao em unidades de limiar por dia.")
