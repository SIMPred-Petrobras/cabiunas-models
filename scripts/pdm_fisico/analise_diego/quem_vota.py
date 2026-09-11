#!/usr/bin/env python3
"""QUAIS CANAIS VOTAM EM CADA UMA DAS 8 DETECCOES DELE.

E a tabela que falta no relatorio de 04/09/2026. A Tabela 4 caracteriza os 96 FP
por combinacao de canais, mas nao ha equivalente para as DETECCOES.

Motivo para medir: o portao ">= 2 canais MODELADOS" (temperatura, vibracao,
oleo) da 0/8 -- nenhum trip sobrevive. Se dois modelos dele quase nunca acendem
juntos, entao TODA deteccao depende do canal 4 (proximidade a alarme de processo
catalogado) ser um dos dois votos. Isso muda o que o detector dele e.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from testa_nossa_decisao import carrega, mede, MODELADOS
from src.cnn1d_ae.scoring import apply_refractory, apply_min_duration_filter

idx, canais, op, ft = carrega()
ns = sum(canais[k].astype(int) for k in canais)
v = (ns >= 2)
vf = apply_min_duration_filter(pd.DataFrame({"is_anom_point": v.astype(int)}, index=idx),
                               45.0)["is_anom_point"].astype(bool)
dec = apply_refractory(vf, refractory_minutes=48*60.0)

print("QUANTO TEMPO CADA CANAL FICA ACESO (sobre operacao)")
print("=" * 76)
rod = (op.astype(str) != "off_longo") & (op.astype(str) != "off")
tot = int(rod.sum())
for k in list(canais):
    n = int((canais[k] & rod).sum())
    print(f"  {k:<12} {100*n/tot:6.2f}% do tempo em operacao")
nmod = sum(canais[k].astype(int) for k in MODELADOS)
print(f"\n  >= 2 canais MODELADOS simultaneos: "
      f"{100*float(((nmod >= 2) & rod).sum())/tot:.3f}% do tempo")
print(f"  >= 2 canais QUAISQUER simultaneos: {100*float((v & rod).sum())/tot:.3f}%")

print("\n\nQUEM VOTA EM CADA DETECCAO  (janela de 48 h antes do trip)")
print("=" * 96)
print(f"{'trip':>17} | {'canais acesos na janela':>34} | {'alarme e essencial?':>20}")
print("-" * 96)
JAN = pd.Timedelta("48h")
essenciais = 0
for t in ft:
    t0 = t - JAN
    # so conta onde a decisao final esta ativa dentro da janela
    janela = dec.loc[t0:t]
    ativo = janela[janela.fillna(False)]
    if not len(ativo):
        print(f"{t:%d/%m/%Y %H:%M} | {'(sem alerta na janela)':>34} | {'--':>20}")
        continue
    i0, i1 = ativo.index[0], ativo.index[-1]
    acesos = [k for k in canais if bool(canais[k].loc[i0:i1].any())]
    # sem o canal de alarme, o voto>=2 ainda fecha em algum instante da janela?
    sem = sum(canais[k].astype(int) for k in MODELADOS).loc[i0:i1]
    fecha_sem = bool((sem >= 2).any())
    essenciais += (not fecha_sem)
    print(f"{t:%d/%m/%Y %H:%M} | {'+'.join(acesos):>34} | "
          f"{('SIM -- sem ele nao fecha' if not fecha_sem else 'nao'):>20}")
print("-" * 96)
print(f"  deteccoes em que o canal de alarme e ESSENCIAL: {essenciais} de {len(ft)}")
