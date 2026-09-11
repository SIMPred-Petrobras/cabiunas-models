#!/usr/bin/env python3
"""Tentar QUEBRAR o 5/8 antes de acreditar nele.

Tres suspeitas:
 1. o orcamento fixo de 1,15 FP/mes penaliza o corte duas vezes -- ele perde
    deteccao E encarece, entao sobra um conjunto pobre (18 configs contra 72).
    Comparar a igual NUMERO de configs, nao a igual orcamento.
 2. "0% detectam 07/04" com n=18 pode ser so pobreza do conjunto. Olhar as 2.187
    sem filtro de orcamento.
 3. o corte pode estar sendo aplicado de forma diferente da de decaimento_pico.py
    (que produziu o 8/8 -> 7/8). Conferir o ponto publicado nos dois caminhos.
"""
import numpy as np, pandas as pd
from loeo_semtravado import loeo, DIAS

A = pd.read_csv("busca_conjunta.csv")
B = pd.read_csv("busca_conjunta_semtravado.csv")
for d in (A, B):
    d["set"] = d["quais"].fillna("").apply(lambda s: {x for x in s.split(",") if x})

print("SUSPEITA 2 -- as 2.187 configuracoes, SEM filtro de orcamento")
print("=" * 78)
print(f"{'evento':>12} {'sem corte':>11} {'com corte':>11}")
for d in DIAS:
    print(f"{d:>12} {A['set'].apply(lambda s: d in s).mean():10.0%} "
          f"{B['set'].apply(lambda s: d in s).mean():10.0%}")
print(f"\n  configs 8/8 : sem corte {int((A.det==8).sum())}   com corte {int((B.det==8).sum())}")
print(f"  menor FP a 8/8: sem corte {A[A.det==8].fp.min():.3f}   "
      f"com corte {B[B.det==8].fp.min():.3f}")

print("\nSUSPEITA 1 -- LOEO a igual NUMERO de configs, nao a igual orcamento")
print("=" * 78)
n_alvo = int((A.fp <= 1.15).sum())
orc_b = float(B.fp.sort_values().iloc[n_alvo - 1])
print(f"  o orcamento de 1,15 deixa {n_alvo} configs sem corte.")
print(f"  para deixar {n_alvo} COM corte, o orcamento tem de ser {orc_b:.3f} FP/mes.\n")
for rot, csv, orc in [("sem corte, orc 1,15", "busca_conjunta.csv", 1.15),
                      ("com corte, orc 1,15", "busca_conjunta_semtravado.csv", 1.15),
                      (f"com corte, orc {orc_b:.2f}", "busca_conjunta_semtravado.csv", orc_b)]:
    res, frag, n = loeo(csv, orc)
    ok, perd = res["menos_horas_fp"]
    print(f"  {rot:<24} [{n:3d} configs]  LOEO {ok}/8   perdidos: "
          f"{', '.join(perd) if perd else '--'}")

print("\nSUSPEITA 1b -- e se o orcamento for generoso o bastante para nao morder?")
print("=" * 78)
for orc in [1.15, 1.5, 2.0, 3.0, 5.0, 99.0]:
    res, _, n = loeo("busca_conjunta_semtravado.csv", orc)
    ok, perd = res["menos_horas_fp"]
    resA, _, nA = loeo("busca_conjunta.csv", orc)
    okA, _ = resA["menos_horas_fp"]
    print(f"  orc {orc:5.2f} FP/mes | com corte: {ok}/8 [{n:4d} cfg] | "
          f"sem corte: {okA}/8 [{nA:4d} cfg]")
