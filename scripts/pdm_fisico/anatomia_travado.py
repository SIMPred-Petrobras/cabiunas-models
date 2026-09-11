#!/usr/bin/env python3
"""O que significa "alarme travado" -- a anatomia do episodio de 670 h que
precede 26/02/2026.

Percorre o episodio em fatias e mostra, por canal, a razao E/limiar (>1 = aceso)
e quantos canais estao acesos. A pergunta que isso responde: o alarme continua
de pe porque o problema continua, ou porque o voto >= 2 fica satisfeito por
sobra, muito depois do sinal que o abriu ter desaparecido?
"""
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, EW, mask, idx, alvo
from publica_clearml import SIN, BASE
from plota_estilo_francisco import alarme, KB, KV

K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
RAZ = {c: EW[c].where(mask) / (BASE[c] * K[c]) for c in SIN}
ON = partes(KB, KV)

al = alarme()
t_ev = alvo.iloc[-1]
a, b = [(x, y) for x, y in AV.episodios(al) if x <= t_ev and y >= t_ev - pd.Timedelta("48h")][0]
print(f"episodio: {a:%d/%m/%Y %H:%M} -> {b:%d/%m/%Y %H:%M}  "
      f"({(b-a).total_seconds()/3600:.0f} h)   trip em {t_ev:%d/%m/%Y %H:%M}\n")

print("A ANATOMIA, EM FATIAS DE 2 DIAS  (razao E/limiar; >1,00 = canal aceso)")
print("=" * 92)
print(f"{'fatia':>17} {'t':>8} {'p':>8} {'sp':>8} {'vb':>8} {'acesos':>8}  quem sustenta o voto")
print("-" * 92)
cortes = pd.date_range(a, b, freq="2D")
for i, s in enumerate(cortes):
    e = cortes[i+1] if i+1 < len(cortes) else b
    vals, acesos = {}, []
    for c in SIN:
        v = float(RAZ[c].loc[s:e].mean())
        vals[c] = v
        if bool(ON[c].loc[s:e].any()):
            acesos.append(c)
    print(f"{s:%d/%m %H:%M}-{e:%d/%m} " + " ".join(f"{vals[c]:8.2f}" for c in SIN)
          + f" {len(acesos):8d}  {'+'.join(acesos)}")
print("-" * 92)

# o que abriu e o que sustenta
prim = RAZ["p"].loc[a:a + pd.Timedelta("12h")].max()
ult = {c: float(RAZ[c].loc[b - pd.Timedelta("48h"):b].mean()) for c in SIN}
print(f"\n  ABRIU  : p em {prim:.1f}x o limiar")
print(f"  TERMINA: " + ", ".join(f"{c}={ult[c]:.2f}x" for c in SIN))
print(f"\n  nas 48 h finais -- a janela que 'detecta' o trip -- o voto e fechado por:")
for c in SIN:
    if bool(ON[c].loc[b - pd.Timedelta("48h"):b].any()):
        print(f"     {c}: media {ult[c]:.3f}x o limiar")
print(f"\n  quanto tempo o alarme fica de pe SEM nenhum canal acima de 1,5x:")
forte = pd.concat([RAZ[c] for c in SIN], axis=1).max(axis=1).loc[a:b]
frac = float((forte < 1.5).mean())
print(f"     {100*frac:.0f}% do episodio ({frac*(b-a).total_seconds()/3600:.0f} h de {(b-a).total_seconds()/3600:.0f} h)")
