"""Elo causal: e o canal de vibracao que entrega os 2 eventos que o Francisco
declara inalcancaveis? Ablacao por canal, com (kb,kv) revarrido em cada ablacao
para nao confundir 'perdeu o sinal' com 'perdeu a calibracao'."""
import sys; sys.path.insert(0, ".")
import itertools, numpy as np, pandas as pd
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
import avalia as AV

paradas = paradas_reais_2h(); JAN = pd.Timedelta(hours=48)
jw = [(t-JAN, t) for t in alvo]
CACHE = {(kb,kv): partes(kb,kv) for kb in (1.3,1.5,1.7,2.0,2.4) for kv in (1.8,2.2,2.8)}
ALVO_F = ("04/11/2025", "09/12/2025")

def roda(usar, kb, kv):
    ON = CACHE[(kb,kv)]
    ns = sum(ON[c].astype(int) for c in usar)
    obrig = [c for c in ("sp","vb") if c in usar]
    v = pd.Series(ns >= 2, index=idx) & mask
    if obrig:
        o = ON[obrig[0]].copy()
        for c in obrig[1:]: o = o | ON[c]
        v = v & o
    al = pos(v, ns, REFRAT_H, DUR_MIN, False)
    eps = AV.episodios(al)
    if not eps: return None
    m = AV.avalia(al, alvo, mask); meses = m["horas_op"]/730.0
    cl = classifica_regra_c(eps, paradas)
    n_fp = sum(1 for a,b,c,l in cl if c=="FP")
    h_fp = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="FP")
    pega = {t.strftime("%d/%m/%Y"): any(a<=t1 and b>=t0 for a,b in eps)
            for t,(t0,t1) in zip(alvo,jw)}
    return dict(det=sum(pega.values()), fp_mes=n_fp/meses, h_mes=h_fp/meses, **pega)

print(f"{'sinais usados':>22} {'det':>5} {'FP/mes':>8} {'h/mes':>8}   04/11  09/12")
for usar in (SIN, ["t","p","sp"], ["t","p","vb"], ["t","sp","vb"], ["p","sp","vb"], ["t","p"]):
    melhor = None
    for kb,kv in CACHE:
        r = roda(usar, kb, kv)
        if r is None or r["fp_mes"] > 1.0: continue
        if melhor is None or (r["det"], -r["h_mes"]) > (melhor["det"], -melhor["h_mes"]):
            melhor = r
    nome = "+".join(usar)
    if melhor is None:
        print(f"{nome:>22}    -- nada dentro do teto de 1 FP/mes"); continue
    a = "SIM" if melhor[ALVO_F[0]] else "nao"
    b = "SIM" if melhor[ALVO_F[1]] else "nao"
    marca = "   <<< sem vibracao" if "vb" not in usar else ""
    print(f"{nome:>22} {melhor['det']:4d}/8 {melhor['fp_mes']:8.3f} {melhor['h_mes']:8.2f}   "
          f"{a:>5}  {b:>5}{marca}")
