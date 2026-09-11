#!/usr/bin/env python3
"""Tabela de features por episodio -- a base para o discriminante.

IDEIA. Nunca usamos os rotulos para DISCRIMINAR, so para calibrar limiar. No
ponto novo temos 8 TP contra ~9 FP -- quase balanceado, que e um problema de
aprendizado muito melhor que a deteccao crua. Com ~17 amostras so se justifica
uma regra de 1 ou 2 features, validada por LOEO.

Tambem responde a (3): os NEUTRO (seguidos de parada real >= 2 h) se parecem com
TP ou com FP? Se com TP, servem como rotulo fraco e dobram o conjunto positivo.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, mask, idx, alvo, op
from publica_clearml import SIN, BASE
from combina_final import constroi, canais
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

LO, HI, KV, IDA, AB = 1.0, 1.7, 2.2, 96, 20
al = constroi(LO, HI, KV, IDA, AB)
paradas = paradas_reais_2h()
cls = classifica_regra_c(AV.episodios(al), paradas)

K = {"t": HI, "p": HI, "sp": HI, "vb": KV}
RAZ = {c: (EW[c].where(mask) / (BASE[c]*K[c])) for c in SIN}
FOR = pd.concat([RAZ[c] for c in SIN], axis=1).max(axis=1)
ONb = canais(HI, KV)

# instantes de religamento, para "tempo desde a partida"
partidas = idx[(op.astype(bool) & ~op.astype(bool).shift(fill_value=False)).to_numpy()]

lin = []
for a, b, k, _ in cls:
    dur = (b - a).total_seconds()/3600
    f = FOR.loc[a:b]
    d = dict(ini=a, classe=k, dur_h=dur,
             pico=float(f.max()), mediana_f=float(f.median()),
             n_canais=sum(1 for c in SIN if bool(ONb[c].loc[a:b].any())))
    for c in SIN:
        d[f"pico_{c}"] = float(RAZ[c].loc[a:b].max())
        d[f"on_{c}"] = int(bool(ONb[c].loc[a:b].any()))
    # taxa de subida nas primeiras 3 h
    s = f.loc[a:a + pd.Timedelta("3h")].dropna()
    if len(s) > 5:
        x = (s.index - s.index[0]).total_seconds().to_numpy()/3600
        d["subida"] = float(np.polyfit(x, s.to_numpy(), 1)[0])
    else:
        d["subida"] = np.nan
    ant = partidas[partidas <= a]
    d["h_pos_partida"] = ((a - ant[-1]).total_seconds()/3600) if len(ant) else np.nan
    d["carga_med"] = float(op.loc[a:b].mean())
    lin.append(d)

D = pd.DataFrame(lin)
D.to_csv("features_episodios.csv", index=False)
print(f"{len(D)} episodios: " + ", ".join(f"{k}={int(v)}" for k, v in D.classe.value_counts().items()))

print("\nFEATURES POR EPISODIO")
print("=" * 118)
cols = ["dur_h", "pico", "mediana_f", "n_canais", "subida", "h_pos_partida", "carga_med"]
print(f"{'inicio':>17} {'classe':>7} " + "".join(f"{c:>13}" for c in cols) + "  canais")
print("-" * 118)
for _, r in D.sort_values("classe").iterrows():
    can = "+".join(c for c in SIN if r[f"on_{c}"])
    print(f"{r.ini:%d/%m/%Y %H:%M} {r.classe:>7} " +
          "".join(f"{r[c]:13.2f}" if np.isfinite(r[c]) else f"{'--':>13}" for c in cols)
          + f"  {can}")

print("\n\nSEPARACAO POR FEATURE  (TP contra FP; NEUTRO mostrado a parte)")
print("=" * 100)
print(f"{'feature':>16} {'TP mediana':>12} {'FP mediana':>12} {'NEUTRO':>10} "
      f"{'AUC(TP vs FP)':>15} {'p (MWU)':>10}")
print("-" * 100)
from scipy.stats import mannwhitneyu
tp = D[D.classe == "TP"]; fp = D[D.classe == "FP"]; nt = D[D.classe == "NEUTRO"]
for c in cols + [f"pico_{x}" for x in SIN]:
    a1, a2 = tp[c].dropna(), fp[c].dropna()
    if len(a1) < 3 or len(a2) < 3:
        continue
    u, p = mannwhitneyu(a1, a2, alternative="two-sided")
    auc = u / (len(a1)*len(a2))
    flag = "  <<<" if p < 0.10 else ""
    print(f"{c:>16} {a1.median():12.2f} {a2.median():12.2f} "
          f"{(nt[c].median() if len(nt) else np.nan):10.2f} {auc:15.2f} {p:10.3f}{flag}")
