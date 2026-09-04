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

# constroi PARADAS REAIS: toda vez que op cai de True pra False e fica False >=2h
para = (~op) & op.shift(fill_value=True)  # instantes em que a maquina acabou de PARAR
# para cada instante de parada, mede quanto tempo fica parada
paradas_reais = []
op_arr = op.to_numpy(); idx_arr = idx
onsets = idx[para.to_numpy()]
for t0 in onsets:
    i = idx_arr.searchsorted(t0)
    j = i
    while j < len(op_arr) and not op_arr[j]:
        j += 1
    dur_h = (idx_arr[min(j, len(op_arr)-1)] - t0).total_seconds()/3600
    if dur_h >= 2.0:
        paradas_reais.append((t0, dur_h))

print(f"paradas reais (>=2h) na serie toda: {len(paradas_reais)}")

FP_JAN = pd.Timedelta(hours=48)
resultado = []
for a, b in eps:
    tp = any(a<=t1 and b>=t0 for t0,t1 in jw)
    if tp:
        resultado.append((a,b,"TP", None))
        continue
    # olha se ha parada real nas 48h seguintes ao FIM do episodio (regra do Francisco)
    cand = [(pt, d) for pt, d in paradas_reais if b <= pt <= b + FP_JAN]
    if cand:
        resultado.append((a,b,"NEUTRO", cand[0]))
    else:
        resultado.append((a,b,"FP", None))

print(f"\n{'inicio':>17} {'fim':>17} {'classe':>8}  parada associada")
for a,b,cl,cand in resultado:
    extra = f"  -> parada em {cand[0]} ({cand[1]:.1f}h)" if cand else ""
    print(f"{a:%d/%m/%Y %H:%M} {b:%d/%m/%Y %H:%M} {cl:>8}{extra}")

n_tp = sum(1 for _,_,c,_ in resultado if c=="TP")
n_fp = sum(1 for _,_,c,_ in resultado if c=="FP")
n_ne = sum(1 for _,_,c,_ in resultado if c=="NEUTRO")
m = AV.avalia(al, alvo, mask)
meses = m["horas_op"]/730.0
print(f"\nTP={n_tp}  FP={n_fp}  NEUTRO={n_ne}")
print(f"FP/mes ANTES (nosso, tudo que nao e TP conta) = {(n_fp+n_ne)/meses:.3f}")
print(f"FP/mes DEPOIS (regra Francisco, exclui NEUTRO) = {n_fp/meses:.3f}")

print("\n" + "="*90)
print("HIATO ENTRE O FIM DO EPISODIO E A PARADA REAL (so os reclassificados)")
print("="*90)
for a,b,cl,cand in resultado:
    if cl == "NEUTRO":
        pt, d = cand
        hiato = (pt - b).total_seconds()/3600
        print(f"  episodio termina {b}  ->  parada real em {pt}  ->  hiato = {hiato:.2f}h  (dur. parada {d:.1f}h)")

print("\n" + "="*90)
print("SENSIBILIDADE: e se a janela fosse mais apertada que 48h?")
print("="*90)
for jan_h in (2, 6, 12, 24, 48):
    JAN_ = pd.Timedelta(hours=jan_h)
    n_ne = 0
    for a, b in eps:
        tp = any(a<=t1 and b>=t0 for t0,t1 in jw)
        if tp: continue
        cand = [(pt, d) for pt, d in paradas_reais if b <= pt <= b + JAN_]
        if cand: n_ne += 1
    n_fp_ = 12 - n_ne
    print(f"  janela={jan_h:2d}h  NEUTRO={n_ne}  FP={n_fp_}  FP/mes={n_fp_/meses:.3f}")
