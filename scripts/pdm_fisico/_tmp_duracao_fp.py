import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN

KB, KV = 1.7, 2.2
ON = partes(KB, KV)
ns = sum(ON[c].astype(int) for c in SIN)
v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
al = pos(v, ns, REFRAT_H, DUR_MIN, False)
eps = AV.episodios(al)
JAN = pd.Timedelta(hours=48); jw = [(t-JAN,t) for t in alvo]

durs=[]
for a,b in eps:
    tp = any(a<=t1 and b>=t0 for t0,t1 in jw)
    dur=(b-a).total_seconds()/3600
    durs.append((a,b,dur,"TP" if tp else "FP"))
durs.sort(key=lambda x:-x[2])
print("episodios por duracao (desc):")
tot_fp=0
for a,b,d,t in durs:
    if t=="FP": tot_fp+=d
    print(f"  {a:%d/%m/%Y %H:%M} -> {b:%d/%m/%Y %H:%M}  {d:7.1f}h  {t}")
print(f"\ntotal horas de FP: {tot_fp:.1f}h  (2 maiores = "
      f"{sum(d for a,b,d,t in durs if t=='FP')- sum(sorted([d for a,b,d,t in durs if t=='FP'])[:-2]):.1f}h "
      f"de {tot_fp:.1f}h)")

# examina o episodio de 153.7h (14/01/2025) -- e continuo ou tem furos?
a0 = pd.Timestamp("2025-01-14 22:48", tz="UTC")
b0 = pd.Timestamp("2025-01-21 04:30", tz="UTC")
vv = v.loc[a0-pd.Timedelta(hours=1): b0+pd.Timedelta(hours=1)]
print(f"\nepisodio 14/01/2025 -- fracao do tempo com voto ativo dentro do intervalo: {vv.mean()*100:.1f}%")
gaps = vv[~vv]
print(f"amostras com voto=False dentro do intervalo: {(~vv).sum()} de {len(vv)}")
