"""Confere a frase do slide 4: "o spread de mancal sozinho detecta 1 de 8, a
pressao sozinha 2 de 8". Duas leituras possiveis de "sozinho" -- mede as duas."""
import sys; sys.path.insert(0, ".")
import pandas as pd, avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

ON = partes(1.7, 2.2)
ns_tot = sum(ON[c].astype(int) for c in SIN)
paradas = paradas_reais_2h()
JAN = pd.Timedelta(hours=48); jw = [(t-JAN, t) for t in alvo]

def mede(v, nome):
    al = pos(v, ns_tot, REFRAT_H, DUR_MIN, False)
    eps = AV.episodios(al)
    if not eps:
        print(f"  {nome:>46}   nenhum episodio"); return
    m = AV.avalia(al, alvo, mask); meses = m["horas_op"]/730.0
    cl = classifica_regra_c(eps, paradas)
    n_fp = sum(1 for a,b,c,l in cl if c=="FP")
    h = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="FP")
    det = sum(any(a<=t1 and b>=t0 for a,b in eps) for t0,t1 in jw)
    print(f"  {nome:>46}   {det}/8   {n_fp/meses:6.3f} FP/mes   {h/meses:6.2f} h/mes")

print('LEITURA A -- o sinal exigido como OBRIGATORIO no voto >=2')
print("=" * 92)
for c in SIN:
    v = pd.Series(ns_tot >= 2, index=idx) & mask & ON[c]
    mede(v, f"voto>=2 exigindo {c}")

print('\nLEITURA B -- o sinal como UNICO canal (sem voto, so ele acima do limiar)')
print("=" * 92)
for c in SIN:
    mede(ON[c] & mask, f"so o canal {c}, sozinho")
