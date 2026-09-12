#!/usr/bin/env python3
"""TESTE 1 -- `sp_vib`: spread de vibracao INTRA-CLASSE.

SUBSTITUI o `sp` atual (spread de TEMPERATURA de mancal: TI_0305 contra a mediana
de TI_0301/0303/0307), que fica aceso 41,8% do tempo e cobre 2/8 na banda
acionavel -- razao 0,55x, ABAIXO do acaso.

POR QUE VIBRACAO E NAO TEMPERATURA:
 1. e onde o sinal esta: nos 4 eventos faltantes a vibracao dos mancais 3/4/5 tem
    z de 4,6 a 14,6; a vibracao crua cobre 7/8 na banda, o sp de temperatura 2/8
 2. imunidade a carga: medido, a vibracao vai de 1,11 a 1,25 entre quintis de
    |dcarga|, praticamente plana; a pressao vai de 1,58 a 22,90 e a temperatura
    de 2,30 a 3,40. O spread na vibracao herda essa imunidade
 3. sao DUAS CLASSES fisicas, segundo a planilha de limites: TV_351/352/353 com
    HH=125 um e TV_354/355 com HH=64 um. O `vb` atual tira o max do z sobre as
    dez como se fossem iguais, e perde o contraste DENTRO de cada classe
 4. imune a deriva da referencia: medi que a referencia rolante sobe junto com a
    degradacao (TV_354Y +107% em um ano). Um spread ENTRE sondas nao se move se
    todas derivam juntas, e acusa se uma deriva sozinha

Valida com nulo por permutacao E LOEO antes de propor como canal.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import g, partes, mask, idx, alvo, op
from plota_estilo_francisco import KB, KV

RNG = np.random.default_rng(20260911)
TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
CLASSE = {"A": ["TV_351X_A","TV_351Y_A","TV_352X_A","TV_352Y_A","TV_353X_A","TV_353Y_A"],
          "B": ["TV_354X_A","TV_354Y_A","TV_355X_A","TV_355Y_A"]}
n = lambda h: int(pd.Timedelta(hours=h)/pd.Timedelta("2min"))


def constroi(ref_h):
    """max sobre as sondas do |z| do desvio de cada uma contra a mediana da sua classe."""
    zs = []
    for cl, tags in CLASSE.items():
        X = g[tags].astype("float64").where(mask)
        med_cl = X.median(axis=1)
        for c in tags:
            d = X[c] - med_cl                      # desvio da sonda contra a classe
            r = d.rolling(n(ref_h), min_periods=n(ref_h)//4)
            mad = (d - r.median()).abs().rolling(n(ref_h), min_periods=n(ref_h)//4).median()*1.4826
            zs.append(((d - r.median())/mad.replace(0, np.nan)).abs())
    return pd.concat(zs, axis=1).max(axis=1)


ti = np.asarray(idx.astype("int64"))
lo_us = int(pd.Timedelta(hours=TMAX).value)//1000
hi_us = int(pd.Timedelta(hours=TMIN).value)//1000
eleg = idx[(idx >= idx[0] + JAN) & op.to_numpy() & mask.to_numpy()]
SORT = RNG.choice(np.asarray(eleg.astype("int64")), size=(5000, len(alvo)))
T_OBS = np.asarray([int(pd.Timestamp(t).value)//1000 for t in alvo])

def cobre(A, T):
    a = np.searchsorted(ti, T - lo_us, "left"); b = np.searchsorted(ti, T - hi_us, "right")
    return np.array([bool(y > x and A[x:y].any()) for x, y in zip(a, b)])

ON = partes(KB, KV)
print("REFERENCIA: os canais de hoje")
for c in ("sp", "vb"):
    A = ON[c].to_numpy()
    o = int(cobre(A, T_OBS).sum())
    nul = np.array([cobre(A, SORT[k]).sum() for k in range(1500)])
    print(f"  {c:>8}: duty {100*float(ON[c].sum())/float(mask.sum()):5.1f}% | banda {o}/8 "
          f"| nulo {nul.mean():.2f} | {o/max(nul.mean(),1e-9):5.2f}x")

for ref_h in (400, 1200, 2400):
    S = constroi(ref_h)
    print(f"\n\nsp_vib com referencia de {ref_h} h")
    print("=" * 92)
    print(f"  mediana {float(S.median()):.2f} | p90 {float(S.quantile(.9)):.2f} "
          f"| p99 {float(S.quantile(.99)):.2f}")
    print(f"  {'limiar':>8}{'duty':>8}{'banda':>8}{'nulo':>8}{'razao':>8}{'p':>9}   LOEO")
    print("  " + "-" * 74)
    for k in (3, 4, 5, 6, 8, 10):
        A = ((S >= k) & mask).fillna(False).to_numpy()
        cob = cobre(A, T_OBS); o = int(cob.sum())
        nul = np.array([cobre(A, SORT[j]).sum() for j in range(3000)])
        p = float((nul >= o).mean())
        duty = 100*float((A & mask.to_numpy()).sum())/float(mask.sum())
        print(f"  {k:8d}{duty:7.2f}%{o:7d}/8{nul.mean():8.2f}"
              f"{o/max(nul.mean(),1e-9):7.2f}x{p:9.4f}", end="")
        print("  ***" if p < 0.05 else "")
    # LOEO do limiar: escolhe nos 7, testa no 8o
    acertos = 0
    GR = [3, 4, 5, 6, 8, 10]
    for i, t in enumerate(alvo):
        melhor_k, melhor = None, -1
        for k in GR:
            A = ((S >= k) & mask).fillna(False).to_numpy()
            cob = cobre(A, T_OBS)
            treino = int(np.delete(cob, i).sum())
            duty = float((A & mask.to_numpy()).sum())/float(mask.sum())
            score = treino - 10*duty              # cobertura penalizada por duty
            if score > melhor: melhor, melhor_k = score, k
        A = ((S >= melhor_k) & mask).fillna(False).to_numpy()
        acertos += bool(cobre(A, T_OBS)[i])
    print(f"  LOEO (limiar escolhido nos outros 7): {acertos}/8")
