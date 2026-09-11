#!/usr/bin/env python3
"""O que acontece entre o episodio de 26/04 e o trip de 29/04?

A hipotese a testar: a maquina PAROU e RELIGOU nesse intervalo. Se parou, o
refratario nao devia continuar valendo -- e outra corrida da maquina, e um alarme
novo na corrida nova nao e repeticao do alarme da corrida anterior. Silenciar
por 72 h atravessando uma parada e erro conceitual, nao ajuste de parametro.
"""
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, EW, mask, idx, alvo, op
from publica_clearml import SIN, BASE
from plota_estilo_francisco import KB, KV, paradas_reais_2h

t = alvo.iloc[4]          # 29/04/2025
print(f"trip: {t:%d/%m/%Y %H:%M}\n")
paradas = paradas_reais_2h()
p = paradas[(paradas.ini >= t - pd.Timedelta("10d")) & (paradas.ini <= t)]
print("PARADAS REAIS (>= 2 h) NOS 10 DIAS ANTES DO TRIP")
print("=" * 80)
if len(p):
    for _, r in p.iterrows():
        print(f"  {r.ini:%d/%m/%Y %H:%M} -> {r.fim:%d/%m %H:%M}  ({r.dur_h:.1f} h)   "
              f"termina {(t - r.fim).total_seconds()/3600:.1f} h antes do trip")
else:
    print("  nenhuma")

print("\nESTADO OPERACIONAL, EM FATIAS DE 12 h")
print("=" * 80)
for s in pd.date_range(t - pd.Timedelta("6d"), t, freq="12h"):
    e = min(s + pd.Timedelta("12h"), t)
    o = op.loc[s:e]
    frac = float(o.mean()) if len(o) else np.nan
    barra = "#" * int(round(frac*20))
    print(f"  {s:%d/%m %H:%M}  rodando {100*frac:5.1f}%  {barra}")

# quantas transicoes desligado->ligado
ON = partes(KB, KV)
K = {"t": KB, "p": KB, "sp": KB, "vb": KV}
print("\nCANAIS ACESOS NAS 48 h FINAIS")
print("=" * 80)
for c in SIN:
    s = ON[c].loc[t - pd.Timedelta("48h"):t]
    n = int(s.fillna(False).sum())
    raz = EW[c].where(mask).loc[t-pd.Timedelta("48h"):t] / (BASE[c]*K[c])
    print(f"  {c:>3}: aceso em {n:4d} amostras ({n*2/60:5.1f} h)  pico {raz.max():6.2f}x")
ns = sum(ON[c].astype(int) for c in SIN)
voto = (ns >= 2) & mask & (ON["sp"] | ON["vb"])
vv = voto.loc[t-pd.Timedelta("48h"):t]
print(f"\n  voto>=2 com portao: {int(vv.sum())} amostras ({int(vv.sum())*2/60:.1f} h) "
      f"nas 48 h finais")
if int(vv.sum()):
    e2 = AV.episodios(vv)
    for a, b in e2:
        print(f"     {a:%d/%m %H:%M} -> {b:%d/%m %H:%M}  ({(b-a).total_seconds()/60:.0f} min)"
              f"  lead {(t-a).total_seconds()/3600:.1f} h")
    print("\n  -> existe voto valido na janela; o que o suprime e o REFRATARIO.")
