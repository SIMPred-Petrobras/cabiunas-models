"""O percentil do PONTO DE PRODUCAO (8/8, 0,517 FP/mes, 7,15 h/mes).
Por canal e, mais util, do detector COMBINADO -- porque o que decide nao e um
limiar isolado, e a conjuncao voto>=2 + SUSTAIN + duracao + refratario."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pos_processamento import partes, pos, mask, idx, alvo, EW
from publica_clearml import SIN, BASE, K, REFRAT_H, DUR_MIN
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c
import avalia as AV

KB, KV = 1.7, 2.2
ON = partes(KB, KV)
ns = sum(ON[c].astype(int) for c in SIN)
v_voto = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
al = pos(v_voto, ns, REFRAT_H, DUR_MIN, False)
n_mask = int(mask.sum())
h_op = n_mask * 2 / 60

print(f"tempo mascarado (operacao estavel, pos-blackout): {n_mask:,} amostras = {h_op:,.0f} h\n")
print("FUNIL -- fracao do tempo de operacao em cada estagio, e o percentil equivalente")
print("=" * 88)
print(f"{'estagio':>46} {'% do tempo':>12} {'percentil':>12}")

for c in SIN:
    f = 100.0 * (ON[c] & mask).sum() / n_mask
    print(f"{'canal ' + c + ' aceso (limiar+SUSTAIN|CUSUM)':>46} {f:11.3f}% {100-f:11.3f}%")

f_voto = 100.0 * v_voto.sum() / n_mask
print(f"{'voto>=2 com sp|vb (antes do pos-proc.)':>46} {f_voto:11.3f}% {100-f_voto:11.3f}%")

f_al = 100.0 * (al & mask).sum() / n_mask
print(f"{'ALARME FINAL (refratario 48h + duracao 120min)':>46} {f_al:11.3f}% {100-f_al:11.3f}%")

eps = AV.episodios(al)
cl = classifica_regra_c(eps, paradas_reais_2h())
h_tp = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="TP")
h_fp = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="FP")
h_ne = sum((b-a).total_seconds()/3600 for a,b,c,l in cl if c=="NEUTRO")

print("\n" + "=" * 88)
print("DECOMPOSICAO DO ALARME FINAL (horas de episodio, calendario)")
print("=" * 88)
for nome, h in (("antecipou falha (TP)", h_tp), ("falso positivo (FP)", h_fp),
                ("antes de parada (NEUTRO)", h_ne)):
    print(f"  {nome:>26}: {h:8.1f} h")
print(f"  {'total':>26}: {h_tp+h_fp+h_ne:8.1f} h")

m = AV.avalia(al, alvo, mask)
meses = m["horas_op"]/730.0
print(f"\n  so o FP: {h_fp:.1f} h em {meses:.2f} meses de operacao = {h_fp/meses:.2f} h/mes")
print(f"  o detector passa {100*h_fp/h_op:.3f}% do tempo de operacao em alarme FALSO")
print(f"  -> percentil efetivo do ponto de producao (so FP): p{100-100*h_fp/h_op:.3f}")
