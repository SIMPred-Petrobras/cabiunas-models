#!/usr/bin/env python3
"""Os dois "7/8" sao do mesmo evento ou de eventos diferentes?

O time ja conhece o 7/8 do LOEO, que derruba 04/11/2025 (fragilidade a escolha
de configuracao: so 8 de 72 configs no orcamento o detectam).

O achado de 04/09/2026 e outro: a deteccao de 26/02/2026 E um alarme travado --
matar o episodio de 670 h mata a deteccao junto.

Se sao eventos diferentes e as duas objecoes sao independentes, o numero honesto
nao e 7/8: e 6/8. Este script mede, em vez de compor por argumento.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV
from decaimento_pico import corta_por_decaimento

ON = partes(KB, KV)
ns = sum(ON[c].astype(int) for c in SIN)
v0 = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
JAN = pd.Timedelta(hours=48)


def detectados(al):
    return {t.strftime("%d/%m/%Y") for t in alvo
            if bool(al.loc[t - JAN:t].fillna(False).any())}


base = detectados(pos(v0, ns, REFRAT_H, DUR_MIN, False))
print("QUAL EVENTO CAI EM CADA OBJECAO")
print("=" * 78)
print(f"  base (ponto de producao): {len(base)}/8\n")
print(f"{'corte':>8} {'det':>5}  evento(s) que caem")
print("-" * 78)
sobrevive_sempre = set(base)
for frac in [0.05, 0.10, 0.20, 0.35, 0.50]:
    d = detectados(pos(corta_por_decaimento(v0, frac), ns, REFRAT_H, DUR_MIN, False))
    caem = sorted(base - d, key=lambda s: (s[6:], s[3:5], s[:2]))
    sobrevive_sempre &= d
    print(f"{frac:8.2f} {len(d):4d}/8  {', '.join(caem) if caem else '(nenhum)'}")
print("-" * 78)

LOEO = "04/11/2025"          # memoria: loeo-e-frageil-a-desempate
TRAVADO = sorted(base - detectados(pos(corta_por_decaimento(v0, 0.05), ns,
                                       REFRAT_H, DUR_MIN, False)))
print(f"\n  cai por alarme travado (decaimento 5%) : {', '.join(TRAVADO)}")
print(f"  cai por fragilidade de config (LOEO)   : {LOEO}")
if LOEO not in TRAVADO:
    print(f"\n  -> EVENTOS DIFERENTES. As duas objecoes sao independentes:")
    print(f"     8/8 in-sample  ->  7/8 tirando o travado  ->  7/8 no LOEO (outro evento)")
    print(f"     ->  {8 - len(set(TRAVADO) | {LOEO})}/8 aplicando as duas")
else:
    print(f"\n  -> MESMO EVENTO. As duas objecoes sao a mesma, e o numero segue 7/8.")
