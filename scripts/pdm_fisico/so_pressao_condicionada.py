#!/usr/bin/env python3
"""Condicionar SO a pressao -- a refinacao dirigida pelo diagnostico.

Condicionar os dois canais (avalia_residuo_carga.py) foi refutado: banda vai de
3/8 a 4/8 mas h/mes salta de 7,1 para 80,3. O culpado e o canal `t`: a carga
explica 85% da variancia dele, entao o residuo perde faixa dinamica e o alarme
fica longo. E a contaminacao de `t` era leve de saida -- 1,48x contra 14,49x de `p`.

Aqui: `t` original, `p` condicionado, `sp` e `vb` intactos.
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo
from publica_clearml import SIN, BASE, HL, SUSTAIN, KAPPA, H_CUSUM, REFRAT_H, DUR_MIN
from blackout_curto import cusum
from avalia_residuo_carga import sinais, pos, mede, E0, E1, q

# hibrido: t do original, p do condicionado
EH = {"t": E0["t"], "p": E1["p"], "sp": E0["sp"], "vb": E0["vb"]}
K0 = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}

# reescala so o p, para o mesmo percentil de operacao do original
s0 = E0["p"].where(mask).dropna(); s1 = E1["p"].where(mask).dropna()
pc = float((s0 <= BASE["p"]*K0["p"]).mean())
BP = float(s1.quantile(pc)) / K0["p"]
BH = dict(BASE); BH["p"] = BP
print(f"escala de p: limiar orig {BASE['p']*K0['p']:.3f} = p{100*pc:.2f}"
      f"  ->  condicionado {s1.quantile(pc):.3f}  (BASE {BASE['p']:.2f} -> {BP:.3f})")

reset = (~mask).to_numpy()
def canais(EW, K, B):
    out = {}
    for c in SIN:
        thr = B[c]*K[c]; E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        cu = pd.Series(cusum(((E/thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy(),
                             reset) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    return out

print("\ncontaminacao do canal p (razao q5/q1)")
for rot, E, B in (("original    ", E0, BASE), ("condicionado", EH, BH)):
    s = E["p"].where(mask)/(B["p"]*K0["p"])
    a = float(s.reindex(q[q == 'q1'].index).mean()); b = float(s.reindex(q[q == 'q5'].index).mean())
    print(f"  {rot}: {b/a:6.2f}x")

print("\nVARREDURA -- t original + p condicionado")
print("=" * 96)
print(f"{'kb':>5}{'kv':>5} | {'banda':>7}{'inicio':>8}{'det':>7}{'FP/mes':>10}{'h/mes':>9}{'lead':>9}")
print("-" * 96)
melhor = None
for kb, kv in itertools.product([1.0, 1.3, 1.5, 1.7, 2.0, 2.4], [1.8, 2.2, 2.8]):
    ON = canais(EH, {"t": kb, "p": kb, "sp": kb, "vb": kv}, BH)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    b, i, d, fp, h, lm = mede(pos(v, REFRAT_H, DUR_MIN))
    if d == 8 and (melhor is None or (b, -fp, -h) > melhor[0]):
        melhor = ((b, -fp, -h), kb, kv, b, i, d, fp, h, lm)
    if d >= 7:
        print(f"{kb:5.1f}{kv:5.1f} | {b:5d}/8{i:7d}/8{d:6d}/8{fp:10.3f}{h:9.1f}"
              f"{(lm if np.isfinite(lm) else 0):8.1f}h")
print("-" * 96)
if melhor:
    _, kb, kv, b, i, d, fp, h, lm = melhor
    print(f"  MELHOR com det=8/8: k={kb}/{kv} -> banda {b}/8, inicio {i}/8, "
          f"{fp:.3f} FP/mes, {h:.1f} h/mes, lead {lm:.1f} h")
else:
    print("  nenhuma configuracao com det = 8/8")
print("  publicado         : banda 3/8, inicio 4/8, det 8/8, 0,517 FP/mes,  7,1 h/mes, lead 12,5 h")
print("  os dois canais    : banda 4/8, inicio 4/8, det 8/8, 1,033 FP/mes, 80,3 h/mes, lead 10,7 h")
