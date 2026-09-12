#!/usr/bin/env python3
"""Quais tags do catalogo REALMENTE antecedem trip? (as 47, nao as 5 dele)

As 5 tags do canal 4 do Diego sao as mais MOVIMENTADAS do catalogo -- 40,9% de
todas as ativacoes -- e dao enriquecimento 1,45x (p = 0,051). A pergunta natural:
existe subconjunto melhor?

ATENCAO AO TESTE MULTIPLO: sao 47 candidatas e 8 eventos. Escolher as melhores
olhando os 8 eventos e ajuste puro. Por isso o nulo aqui e por PERMUTACAO e o
p reportado e CORRIGIDO por Bonferroni; e qualquer canal construido depois disso
precisa de LOEO antes de valer alguma coisa.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import mask, idx, alvo, op
from fp_alarmes import catalogo

RNG = np.random.default_rng(20260911)
N = 20000
JAN = pd.Timedelta("48h")
cat = catalogo(idx)
TAGS_DELE = {"PI_6240319_AL", "PAL_6240315", "PDAL_6240302", "TC382_05_A", "PAH_6240319"}

eleg = idx[(idx >= idx[0] + JAN) & op.to_numpy()]
cand = np.asarray(eleg.astype("int64"))
jan_us = int(JAN.value) // 1000
sort = RNG.choice(cand, size=(N, len(alvo)))
dias = (idx[-1] - idx[0]).days

lin = []
for tg, s in cat.groupby("Tag Alarme"):
    tv = np.sort(np.asarray(pd.DatetimeIndex(s["t"]).astype("int64")))
    if len(tv) < 3:
        continue
    obs = sum(1 for t in alvo
              if np.searchsorted(tv, int(pd.Timestamp(t).value)//1000, "right")
                 - np.searchsorted(tv, int((pd.Timestamp(t)-JAN).value)//1000, "left") > 0)
    hi = np.searchsorted(tv, sort, "right"); lo = np.searchsorted(tv, sort - jan_us, "left")
    d = (hi - lo > 0).sum(axis=1)
    esp = float(d.mean()); p = float((d >= obs).mean())
    lin.append(dict(tag=str(tg), n=len(tv), mes=len(tv)/(dias/30.44), obs=obs,
                    esp=esp, enr=(obs/esp if esp > 0 else np.inf), p=p,
                    dele=str(tg) in TAGS_DELE,
                    desc=str(s["Descrição Alarme"].dropna().iloc[0])[:40]
                    if s["Descrição Alarme"].notna().any() else ""))
D = pd.DataFrame(lin).sort_values(["obs", "enr"], ascending=False)
m = len(D)
D["p_bonf"] = (D.p * m).clip(upper=1.0)

print(f"{m} tags com >=3 ativacoes, {len(alvo)} eventos. Bonferroni sobre {m} testes.")
print("=" * 112)
print(f"{'tag':>16}{'/mes':>7}{'obs':>6}{'esperado':>10}{'enriq':>8}{'p':>8}{'p_bonf':>9}"
      f"{'dele':>6}  descricao")
print("-" * 112)
for _, r in D.head(16).iterrows():
    flag = "  ***" if r.p_bonf < 0.05 else ("  *" if r.p < 0.05 else "")
    print(f"{r.tag:>16}{r.mes:7.1f}{r.obs:5d}/8{r.esp:9.2f}{r.enr:8.2f}x{r.p:8.4f}"
          f"{r.p_bonf:9.3f}{('SIM' if r.dele else ''):>6}  {r.desc}{flag}")
print("-" * 112)
sig = D[D.p_bonf < 0.05]
print(f"  tags significativas apos Bonferroni: {len(sig)} de {m}")
if len(sig):
    print("   ", ", ".join(sig.tag))
else:
    print("    NENHUMA -- nao ha subconjunto do catalogo que anteceda trip de forma")
    print("    distinguivel do acaso com 8 eventos.")
nao_dele = D[~D.dele].head(3)
print(f"\n  melhores FORA das 5 dele: "
      + ", ".join(f"{r.tag} ({r.obs}/8, {r.enr:.2f}x, p={r.p:.3f})"
                  for _, r in nao_dele.iterrows()))
