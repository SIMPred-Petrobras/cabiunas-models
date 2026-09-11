#!/usr/bin/env python3
"""LIMIAR LIVRE POR CANAL -- a peca que vale copiar do Diego.

DIAGNOSTICO (06/09/2026). Os nossos canais t, p e sp COMPARTILHAM o mesmo `kb`:
`K = {"t": kb, "p": kb, "sp": kb, "vb": kv}`. Sao 2 parametros de limiar para 4
canais. Ele tem tres independentes, e escolhidos em percentis muito diferentes
(temperatura p97,5 · vibracao p95,0 · oleo p99,9).

E a diferenca que produz o perfil de antecedencia dele -- 3,8 a 43,2 h, apertado
e usavel -- contra o nosso, espalhado de 1,3 a 194,9 h. Canais em percentis
efetivos diferentes disparam em momentos descoordenados: o vb em ~p69 acende
cedo na deriva e fica de pe, t e p acendem tarde na excursao rapida.

NAO e a autocalibracao por percentil que ja foi refutada -- aquela impunha UM
percentil global a todos os canais. Aqui cada canal tem multiplicador proprio,
escolhido para a BANDA ACIONAVEL.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, pos, mask, idx, alvo
from publica_clearml import SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, DUR_MIN
from blackout_curto import cusum
from corte_com_rearme import corta_rearma
from escalada_por_idade import quebra_idade
from checa_degenerado import pos_dur_esc
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

reset = (~mask).to_numpy()
paradas = paradas_reais_2h(); meses = float(mask.sum())*2/60.0/730.0
TMIN, TMAX = 4.0, 48.0
FRAC, IDADE, ABS, REFRAT, DUR_ESC = 0.03, 96, 20, 72, 60

# canal isolado, por (nome, k) -- 4 x len(grade) calculos, nao 4^len
CAN = {}
def canal(c, k):
    if (c, k) in CAN:
        return CAN[(c, k)]
    thr = BASE[c]*k
    E = EW[c].where(mask)
    deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
    cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                         reset) > H_CUSUM, index=idx)
    CAN[(c, k)] = (deg | cu) & mask
    return CAN[(c, k)]


def avalia_tudo(al):
    eps = AV.episodios(al); banda, leads, ini = 0, [], 0
    for t in alvo:
        c = [a for a, _ in eps
             if t - pd.Timedelta(hours=TMAX) <= a <= t - pd.Timedelta(hours=TMIN)]
        if c: banda += 1; leads.append((t - max(c)).total_seconds()/3600)
        if any(t - pd.Timedelta(hours=TMAX) <= a <= t for a, _ in eps): ini += 1
    m = AV.avalia(al, alvo, mask); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    return banda, ini, m["det"], nfp/meses, h/meses, (np.mean(leads) if leads else np.nan)


GR = {"t":  [1.1, 1.4, 1.7, 2.0, 2.4],
      "p":  [1.1, 1.4, 1.7, 2.0, 2.4],
      "sp": [1.1, 1.4, 1.7, 2.0, 2.4],
      "vb": [1.5, 1.8, 2.2, 2.8, 3.4]}
print(f"varrendo {np.prod([len(v) for v in GR.values()])} combinacoes de limiar por canal ...",
      flush=True)

res = []
for kt, kp, ks, kv in itertools.product(GR["t"], GR["p"], GR["sp"], GR["vb"]):
    K = {"t": kt, "p": kp, "sp": ks, "vb": kv}
    ON = {c: canal(c, K[c]) for c in SIN}
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    F = pd.concat([EW[c].where(mask)/(BASE[c]*K[c]) for c in SIN], axis=1).max(axis=1).to_numpy()
    vv = quebra_idade(corta_rearma(v.to_numpy(), F, FRAC), F, ABS, IDADE)
    al = pos_dur_esc(pd.Series(vv, index=idx), REFRAT, F, ABS, IDADE, DUR_ESC)
    b, i, d, fp, h, lm = avalia_tudo(al)
    res.append(dict(kt=kt, kp=kp, ksp=ks, kvb=kv, banda=b, ini=i, det=d,
                    fp=round(fp, 3), h=round(h, 1), lead=round(lm, 1) if np.isfinite(lm) else np.nan))
D = pd.DataFrame(res); D.to_csv("limiar_por_canal.csv", index=False)

print(f"\n{len(D)} combinacoes\n")
print("MELHOR POR NIVEL DE BANDA ACIONAVEL, mantendo det = 8/8")
print("=" * 104)
print(f"{'banda':>7} {'n':>5} {'ini':>5} {'FP/mes':>9} {'h/mes':>8} {'lead':>8}   k_t/k_p/k_sp/k_vb")
print("-" * 104)
ok = D[D.det == 8]
for k in sorted(ok.banda.unique(), reverse=True):
    s = ok[ok.banda == k].sort_values(["fp", "h"])
    b = s.iloc[0]
    print(f"{int(k):5d}/8 {len(s):5d} {int(b.ini):3d}/8 {b.fp:9.3f} {b.h:8.1f} {b.lead:7.1f}h"
          f"   {b.kt}/{b.kp}/{b.ksp}/{b.kvb}")
print("-" * 104)
atual = D[(D.kt == 1.7) & (D.kp == 1.7) & (D.ksp == 1.7) & (D.kvb == 2.2)]
if len(atual):
    a = atual.iloc[0]
    print(f"  o nosso ponto de hoje (k igual em t/p/sp): banda {int(a.banda)}/8, "
          f"inicio {int(a.ini)}/8, det {int(a.det)}/8, {a.fp:.3f} FP/mes, {a.h:.1f} h/mes, "
          f"lead {a.lead:.1f} h")
