import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo, op
from publica_clearml import SIN, REFRAT_H, DUR_MIN

KB, KV = 1.7, 2.2
ON = partes(KB, KV)
ns = sum(ON[c].astype(int) for c in SIN)
v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
al = pos(v, ns, REFRAT_H, DUR_MIN, False)
eps = AV.episodios(al)
JAN = pd.Timedelta(hours=48); jw = [(t-JAN,t) for t in alvo]

# --- distribuicao das transicoes de RUNNING_A, igual o Francisco mediu na dele ---
op_arr = op.to_numpy(); idx_arr = idx
transicoes = []
i = 0
estado = op_arr[0]
ini = idx_arr[0]
for k in range(1, len(op_arr)):
    if op_arr[k] != estado:
        dur_min = (idx_arr[k] - ini).total_seconds()/60
        transicoes.append((ini, estado, dur_min))
        estado = op_arr[k]; ini = idx_arr[k]
transicoes.append((ini, estado, (idx_arr[-1]-ini).total_seconds()/60))
T = pd.DataFrame(transicoes, columns=["ini","estado","dur_min"])
curtas = (T.dur_min < 3).sum()
print(f"transicoes de RUNNING_A na nossa serie: {len(T)}  ( <3min: {curtas} = {100*curtas/len(T):.1f}% )")
print(f"transicoes de PARADA (estado=False) com dur<3min: {((T.estado==False)&(T.dur_min<3)).sum()} "
      f"de {(T.estado==False).sum()} paradas totais")
print()

# --- paradas reais >=2h (piso do Francisco) ---
para = (~op) & op.shift(fill_value=True)
paradas_reais = []
for t0 in idx[para.to_numpy()]:
    i0 = idx_arr.searchsorted(t0); j = i0
    while j < len(op_arr) and not op_arr[j]:
        j += 1
    dur_h = (idx_arr[min(j, len(op_arr)-1)] - t0).total_seconds()/3600
    if dur_h >= 2.0:
        paradas_reais.append((t0, dur_h))
print(f"paradas reais (>=2h): {len(paradas_reais)}")

# --- regra defendida pelo Francisco: [inicio, fim+48h] ---
resultado = []
for a, b in eps:
    tp = any(a<=t1 and b>=t0 for t0,t1 in jw)
    if tp:
        resultado.append((a,b,"TP", None)); continue
    cand = [(pt, d) for pt, d in paradas_reais if a <= pt <= b + JAN]
    resultado.append((a,b,"NEUTRO" if cand else "FP", cand[0] if cand else None))

print(f"\n{'inicio':>17} {'fim':>17} {'classe':>8}  parada associada")
for a,b,cl,cand in resultado:
    extra = f"  -> parada em {cand[0]} ({cand[1]:.1f}h)" if cand else ""
    print(f"{a:%d/%m/%Y %H:%M} {b:%d/%m/%Y %H:%M} {cl:>8}{extra}")

n_tp = sum(1 for _,_,c,_ in resultado if c=="TP")
n_fp = sum(1 for _,_,c,_ in resultado if c=="FP")
n_ne = sum(1 for _,_,c,_ in resultado if c=="NEUTRO")
m = AV.avalia(al, alvo, mask)
meses = m["horas_op"]/730.0
print(f"\nTP={n_tp}  FP={n_fp}  NEUTRO={n_ne}  (de {n_tp+n_fp+n_ne} episodios)")
print(f"FP/mes regra Francisco [inicio, fim+48h] = {n_fp/meses:.3f}")
