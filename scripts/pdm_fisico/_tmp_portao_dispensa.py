"""A porta de mancal (sp|vb) e o gargalo do 04/11. Dispensa-la quando o voto e
UNANIME recupera o evento sem devolver transiente de partida?

A porta existe para proteger de religamento: um sinal sozinho dispara na partida,
dois simultaneos nao. A pergunta e se TRES ou QUATRO simultaneos ja bastam como
protecao, tornando a porta desnecessaria nesse regime.

RISCO A DECLARAR: e uma regra desenhada olhando um evento. Por isso mede-se a
fragilidade dos OITO e o custo, nao so se o 04/11 passa a ser pego."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

KBS = [1.3, 1.5, 1.7, 2.0, 2.4]; KVS = [1.8, 2.2, 2.8, 3.5]
TETO = 1.0
paradas = paradas_reais_2h(); JAN = pd.Timedelta(hours=48)
jw = [(t-JAN, t) for t in alvo]; COLS = [t.strftime("%d/%m/%Y") for t in alvo]
CACHE = {(kb,kv): partes(kb,kv) for kb in KBS for kv in KVS}

def varre(dispensa_em):
    """dispensa_em=None -> porta sempre exigida (atual). =3 -> dispensada se >=3 votam."""
    out = []
    for (kb,kv), ON in CACHE.items():
        ns = sum(ON[c].astype(int) for c in SIN)
        porta = ON["sp"] | ON["vb"]
        if dispensa_em is not None:
            porta = porta | (ns >= dispensa_em)
        v = pd.Series(ns >= 2, index=idx) & mask & porta
        al = pos(v, ns, REFRAT_H, DUR_MIN, False)
        eps = AV.episodios(al)
        if not eps: continue
        m = AV.avalia(al, alvo, mask); meses = m["horas_op"]/730.0
        cl = classifica_regra_c(eps, paradas)
        n_fp = sum(1 for a,b,c,l in cl if c=="FP")
        h = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="FP")
        pega = {t.strftime("%d/%m/%Y"): any(a<=t1 and b>=t0 for a,b in eps)
                for t,(t0,t1) in zip(alvo,jw)}
        out.append(dict(kb=kb, kv=kv, det=sum(pega.values()), fp_mes=n_fp/meses,
                        h_mes=h/meses, lead=m["lead_med"], **pega))
    return pd.DataFrame(out)

print("FRAGILIDADE POR EVENTO (fracao das configs no teto de 1 FP/mes que detectam)")
print("="*104)
print(f"{'regra da porta':>26} {'cfgs':>5} " + "".join(f"{c[:5]:>8}" for c in COLS) + f"{'media':>8}")
for nome, disp in (("exigida sempre (ATUAL)", None),
                   ("dispensada se >=3 votam", 3),
                   ("dispensada se 4 votam", 4)):
    T = varre(disp); d = T[T.fp_mes <= TETO]
    if d.empty: print(f"{nome:>26}   nada no teto"); continue
    fr = [100*d[c].mean() for c in COLS]
    print(f"{nome:>26} {len(d):5d} " + "".join(f"{x:7.0f}%" for x in fr) + f"{np.mean(fr):7.0f}%")

print("\nMELHOR PONTO DE CADA REGRA (8/8 e depois menor h/mes)")
print("="*104)
print(f"{'regra da porta':>26} {'det':>5} {'kb':>5} {'kv':>5} {'FP/mes':>8} {'h/mes':>8} {'lead':>7}")
for nome, disp in (("exigida sempre (ATUAL)", None),
                   ("dispensada se >=3 votam", 3),
                   ("dispensada se 4 votam", 4)):
    T = varre(disp); d = T[T.fp_mes <= TETO]
    if d.empty: continue
    b = d.sort_values(["det","h_mes"], ascending=[False,True]).iloc[0]
    print(f"{nome:>26} {b.det:4.0f}/8 {b.kb:5.1f} {b.kv:5.1f} {b.fp_mes:8.3f} {b.h_mes:8.2f} {b.lead:6.1f}h")
