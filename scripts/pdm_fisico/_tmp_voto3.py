import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN

KB, KV = 1.7, 2.2
ON = partes(KB, KV)
ns = sum(ON[c].astype(int) for c in SIN)

def roda(voto_min):
    v = pd.Series(ns >= voto_min, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pos(v, ns, REFRAT_H, DUR_MIN, False)
    m = AV.avalia(al, alvo, mask)
    eps = AV.episodios(al)
    JAN = pd.Timedelta(hours=48); jw = [(t-JAN,t) for t in alvo]
    fp = sum(1 for a,b in eps if not any(a<=t1 and b>=t0 for t0,t1 in jw))
    meses = m["horas_op"]/730.0
    perdidos = sorted(set(t.strftime("%d/%m/%Y") for t in alvo) - set(m["detectados"]))
    return m["det"], len(eps), fp, fp/meses, perdidos

print(f"{'voto_min':>9} {'det':>5} {'episodios':>10} {'FP':>4} {'FP/mes':>8}  perdidos")
for vm in (2,3,4):
    det, ep, fp, fpm, perd = roda(vm)
    print(f"{vm:9d} {det}/8   {ep:10d} {fp:4d} {fpm:8.3f}  {perd}")
