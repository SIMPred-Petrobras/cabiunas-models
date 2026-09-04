"""EXP27 dele: 1 modelo unificado da 7/9; 2 especializados por subsistema dao 8/8.
Traduzido pro nosso detector: em vez de UM voto>=2 sobre os 4 sinais, DOIS
detectores especializados cuja uniao vira o alarme.

  MANCAL/TERMICO: {t, sp, vb}   OLEO/PRESSAO: {p, vb}

Os 2 eventos que ele perde no modelo unificado (11/04/2025 mancal, 04/11/2025 oleo)
sao justamente os mais fragis no NOSSO detector tambem -- 11/04 tem lead de 2,8h
(o menor) e 04/11 e o que o LOEO sempre derruba. Convergencia que vale medir."""
import sys; sys.path.insert(0, ".")
import pandas as pd, itertools
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
import avalia as AV

KB, KV = 1.7, 2.2
ON = partes(KB, KV)
paradas = paradas_reais_2h()
ns_tot = sum(ON[c].astype(int) for c in SIN)

def mede(v, nome):
    al = pos(v, ns_tot, REFRAT_H, DUR_MIN, False)
    eps = AV.episodios(al)
    if not eps: return None
    m = AV.avalia(al, alvo, mask); meses = m["horas_op"]/730.0
    cl = classifica_regra_c(eps, paradas)
    n_tp = sum(1 for a,b,c,l in cl if c=="TP"); n_fp = sum(1 for a,b,c,l in cl if c=="FP")
    n_ne = sum(1 for a,b,c,l in cl if c=="NEUTRO")
    h_fp = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="FP")
    return dict(nome=nome, det=m["det"], TP=n_tp, FP=n_fp, NEU=n_ne,
                fp_mes=n_fp/meses, h_mes=h_fp/meses, lead=m["lead_med"], eps=len(eps))

def voto(sinais, minimo, obrig=None):
    n = sum(ON[c].astype(int) for c in sinais)
    v = pd.Series(n >= minimo, index=idx) & mask
    if obrig:
        o = ON[obrig[0]].copy()
        for c in obrig[1:]: o = o | ON[c]
        v = v & o
    return v

res = []
# referencia: o ponto atual
res.append(mede(voto(SIN, 2, ["sp","vb"]), "ATUAL: voto>=2 dos 4, exige sp|vb"))

# a arquitetura dele: dois especializados, uniao
A = voto(["t","sp","vb"], 2, ["sp","vb"])
B = voto(["p","vb"], 2)
res.append(mede(A | B, "2 especializados: {t,sp,vb}>=2 U {p,vb}>=2"))
res.append(mede(A, "  so o de mancal/termico {t,sp,vb}>=2"))
res.append(mede(B, "  so o de oleo/pressao {p,vb}>=2"))

# variantes de particao
A2 = voto(["t","sp"], 2); B2 = voto(["p","vb"], 2)
res.append(mede(A2 | B2, "2 especializados: {t,sp}>=2 U {p,vb}>=2"))
A3 = voto(["t","sp","vb"], 2, ["sp","vb"]); B3 = voto(["p","sp","vb"], 2, ["sp","vb"])
res.append(mede(A3 | B3, "2 especializados: {t,sp,vb} U {p,sp,vb}, ambos exigem sp|vb"))

print(f"{'configuracao':>48} {'det':>4} {'TP':>3} {'FP':>3} {'NEU':>4} {'FP/mes':>8} {'h/mes':>8} {'lead':>7}")
for r in res:
    if r is None: continue
    print(f"{r['nome']:>48} {r['det']:4d} {r['TP']:3d} {r['FP']:3d} {r['NEU']:4d} "
          f"{r['fp_mes']:8.3f} {r['h_mes']:8.2f} {r['lead']:7.1f}")
