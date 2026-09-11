#!/usr/bin/env python3
"""IDEIA 1 -- CUSUM com esquecimento (leaky / FIR-CUSUM).

DIAGNOSTICO QUE MOTIVA. O nosso CUSUM e S_i = max(0, S_{i-1} + x_i), sem
decaimento: so zera quando a mascara quebra. Por isso a excursao de p a 35x em
29/01/2026 mantem o alarme de pe por TRES SEMANAS -- 86% do episodio de 670 h
nao tem nenhum canal acima de 1,5x. Foi o que medimos em anatomia_travado.py.

O religamento (corta_rearma) conserta o SINTOMA -- corta o episodio depois de
ele existir. A leitura correta e que o acumulador nao devia ter memoria
infinita. CUSUM com esquecimento, S_i = max(0, L*S_{i-1} + x_i), tem memoria
efetiva de 1/(1-L) amostras: a evidencia velha evapora sozinha, o episodio
fecha por conta propria, e um sinal novo perto do trip abre episodio novo --
que e exatamente o que a regua de inicio exige.

E mais principiado que o patch: 1 parametro, ataca a causa, e a memoria efetiva
tem interpretacao fisica direta (quantas horas de evidencia o detector carrega).

HIPOTESE: det_ini sobe sem o custo do religamento, porque nao fragmenta episodio
-- so impede que ele se estenda por semanas.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import EW, pos, mask, idx, alvo
from publica_clearml import (SIN, BASE, SUSTAIN, KAPPA, H_CUSUM, CARGA,
                             REFRAT_H, DUR_MIN)
from regra_inicio_varredura import avalia_inicio
from plota_estilo_francisco import KB, KV, paradas_reais_2h, classifica_regra_c

reset = (~mask).to_numpy()
PASSO_H = (idx[1] - idx[0]).total_seconds() / 3600


def cusum_leak(x, reset, L):
    """S_i = max(0, L*S_{i-1} + x_i). L = 1 recupera o CUSUM atual."""
    S = np.empty(len(x)); a = 0.0
    for i in range(len(x)):
        a = a * CARGA if reset[i] else max(0.0, a * L + x[i])
        S[i] = a
    return S


def partes_leak(kb, kv, L):
    K = {"t": kb, "p": kb, "sp": kb, "vb": kv}
    out = {}
    for c in SIN:
        thr = BASE[c] * K[c]
        E = EW[c].where(mask)
        deg = ((E > thr).astype(int).rolling(SUSTAIN, min_periods=SUSTAIN).sum() >= SUSTAIN)
        x = ((E / thr).clip(upper=20) - KAPPA).fillna(0.0).to_numpy()
        cu = pd.Series(cusum_leak(x, reset, L) > H_CUSUM, index=idx)
        out[c] = (deg | cu) & mask
    return out


paradas = paradas_reais_2h(); meses = float(mask.sum()) * 2 / 60.0 / 730.0
JAN = pd.Timedelta(hours=48)

print("IDEIA 1 -- CUSUM COM ESQUECIMENTO")
print("=" * 104)
print(f"{'L':>9} {'meia-vida':>11} {'det':>6} {'det_ini':>8} {'FP/mes':>9} {'h/mes':>8} "
      f"{'eps':>5} {'lead_ini':>9}  quem nasce na janela")
print("-" * 104)
for L in [1.0, 0.9999, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.98]:
    ON = partes_leak(KB, KV, L)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    al = pos(v, ns, REFRAT_H, DUR_MIN, False)
    m = AV.avalia(al, alvo, mask); mi = avalia_inicio(al)
    eps = AV.episodios(al); cls = classifica_regra_c(eps, paradas)
    nfp = sum(1 for _, _, k, _ in cls if k == "FP")
    h = sum((b-a).total_seconds()/3600 for a, b, k, _ in cls if k == "FP")
    nasce = sorted({t.strftime("%d/%m") for t in alvo
                    if any(t - JAN <= x <= t for x, _ in eps)})
    mh = (np.log(0.5)/np.log(L)*PASSO_H) if L < 1 else np.inf
    print(f"{L:9.4f} {(f'{mh:8.1f} h' if np.isfinite(mh) else '     inf'):>11} "
          f"{m['det']:5d}/8 {mi['det']:7d}/8 {nfp/meses:9.3f} {h/meses:8.1f} {len(eps):5d} "
          f"{(mi['lead_med'] if mi['det'] else 0):8.1f}h  {','.join(nasce)}")
print("-" * 104)
print("  referencia: publicado 8/8 · 4/8 ini · 0,517 · 7,1  |  religamento 8/8 · 5/8 ini · 0,775 · 11,3")
