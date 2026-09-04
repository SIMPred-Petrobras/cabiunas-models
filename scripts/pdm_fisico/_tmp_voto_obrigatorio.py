"""A regra de hoje exige (sp OU vb) como sinal obrigatorio do voto>=2.
A tabela de assinaturas sugere que 'p' esta enriquecido nos nao-TP e 't' nos TP.
Testa todas as variantes de sinal obrigatorio, no mesmo ponto de operacao."""
import sys; sys.path.insert(0, ".")
import pandas as pd, itertools
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
import avalia as AV

KB, KV = 1.7, 2.2
ON = partes(KB, KV)
ns = sum(ON[c].astype(int) for c in SIN)
paradas = paradas_reais_2h()

print(f"{'obrigatorio':>16} {'voto':>5} {'det':>5} {'TP':>4} {'FP':>4} {'NEU':>4} {'FP/mes':>8} {'h/mes':>8}")
combos = []
for r in (1, 2, 3, 4):
    combos += list(itertools.combinations(SIN, r))
for voto_min in (2, 3):
    for comb in combos:
        obrig = ON[comb[0]].copy()
        for c in comb[1:]:
            obrig = obrig | ON[c]
        v = pd.Series(ns >= voto_min, index=idx) & mask & obrig
        al = pos(v, ns, REFRAT_H, DUR_MIN, False)
        eps = AV.episodios(al)
        if not eps:
            continue
        m = AV.avalia(al, alvo, mask)
        meses = m["horas_op"] / 730.0
        cl = classifica_regra_c(eps, paradas)
        n_tp = sum(1 for *_, c, _ in [(a,b,c,l) for a,b,c,l in cl] if c == "TP")
        n_fp = sum(1 for a,b,c,l in cl if c == "FP")
        n_ne = sum(1 for a,b,c,l in cl if c == "NEUTRO")
        h_fp = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c == "FP")
        marca = "  <<< ATUAL" if comb == ("sp","vb") and voto_min == 2 else ""
        print(f"{'|'.join(comb):>16} {voto_min:5d} {m['det']:5d} {n_tp:4d} {n_fp:4d} {n_ne:4d} "
              f"{n_fp/meses:8.3f} {h_fp/meses:8.2f}{marca}")
