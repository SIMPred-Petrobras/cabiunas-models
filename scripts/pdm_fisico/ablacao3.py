#!/usr/bin/env python3
"""Comparacao final, a custo operacional igualado.

Regra fixada antes de olhar o resultado: o braco base roda no seu limiar nativo
(k=1) e custa X falsos positivos por mes no TREINO. Cada outro braco e avaliado
no k cujo FP de treino e o mais proximo de X. Assim nenhum braco compra
deteccao gastando mais alarme -- que foi o vicio da primeira rodada.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao2 import alerta_k, BRACOS

t = pd.read_csv("ablacao2.csv")
ALVO = float(t[(t.braco == "base") & (t.k == 1.0)]["tr_fp"].iloc[0])
print(f"custo do detector base no seu limiar nativo: {ALVO:.2f} FP/mes no treino\n")

df = canonico()
falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
idx = df.index
tr = pd.Series(idx < CORTE, index=idx); te = ~tr
ev_tr, ev_te = falhas[falhas < CORTE], falhas[falhas >= CORTE]
mask = mascara_pontuacao(df)

escolha = {}
for b in BRACOS:
    s = t[t.braco == b].copy()
    s["d"] = (s.tr_fp - ALVO).abs()
    escolha[b] = float(s.sort_values("d").iloc[0]["k"])
print("k escolhido por braco (pelo FP de treino, nunca pelo teste):", escolha, "\n")

print(f"{'braco':13s} {'k':>4} | {'treino':>6} {'FP':>5} {'h/mes':>6} {'p':>6} "
      f"| {'TESTE':>6} {'FP':>5} {'h/mes':>6} {'p':>6}")
print("-" * 86)
det_por_evento = {}
for b in BRACOS:
    out = roda(b, df, falhas)
    al = alerta_k(out, mask, b, escolha[b])
    linha = []
    for tag, m, ev in [("tr", tr, ev_tr), ("te", te, ev_te)]:
        am, qm = al[m], (mask & m)[m]
        x = A.avalia(am, ev, qm); x.update(A.permuta(am, qm, x["det"], len(ev)))
        linha.append(x)
    a, c = linha
    print(f"{b:13s} {escolha[b]:>4} | {a['det']}/{a['n_ev']:<4} {a['fp_mes']:5.2f} {a['h_fp_mes']:6.1f} "
          f"{a['p']:6.3f} | {c['det']}/{c['n_ev']:<4} {c['fp_mes']:5.2f} {c['h_fp_mes']:6.1f} {c['p']:6.3f}")
    leads = {}
    for tt in falhas:
        w = al[(idx >= tt - pd.Timedelta(hours=48)) & (idx < tt)]; w = w[w]
        leads[tt.strftime("%Y-%m-%d")] = (tt - w.index[0]).total_seconds()/3600 if len(w) else np.nan
    det_por_evento[b] = leads

print("\nantecedencia por evento, em horas (vazio = nao visto):")
E = pd.DataFrame(det_por_evento)
E.index = [f"{i} {'TESTE' if pd.Timestamp(i, tz='UTC') >= CORTE else '     '}" for i in E.index]
print(E.to_string(float_format=lambda v: f"{v:.1f}", na_rep="  --"))
E.to_csv("ablacao_leads.csv")
