#!/usr/bin/env python3
"""Por que o `vb`, aceso 52% do tempo, perde anomalias de vibracao com z de 8 a 14?

A pericia mostrou que 27/02 (TV_354X z=14,6) e 26/02 (TV_353Y z=8,6) tem
vibracao claramente anomala na janela acionavel, e o nosso canal nao acende.

Hipotese: a REFERENCIA ROLANTE de 400 h absorve a subida. Se a anomalia se
desenvolve ao longo de semanas, a referencia acompanha e o z nao sobe -- o
detector se adapta a propria degradacao.

Compara a referencia rolante de 400 h com referencias fixas e mais longas.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import g, mask, idx, alvo, op
from cabiunas_pdm import config as C

TV = C.VIBRATION_TAGS
TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)
X = g[TV].astype("float64").where(mask)
n = lambda h: int(pd.Timedelta(hours=h)/pd.Timedelta("2min"))

print("O QUE A REFERENCIA VE NOS DOIS EVENTOS DE VIBRACAO")
print("=" * 96)
for t in (alvo.iloc[0], alvo.iloc[7]):
    print(f"\n### {t:%d/%m/%Y}")
    a, b = t - JAN, t - pd.Timedelta(hours=TMIN)
    print(f"  {'sonda':>12}{'valor na janela':>17}{'ref 400h':>11}{'ref 2000h':>11}"
          f"{'z 400h':>9}{'z 2000h':>10}")
    print("  " + "-" * 70)
    for c in TV:
        s = X[c]
        w = s.loc[a:b].dropna()
        if not len(w): continue
        r400 = s.loc[a - pd.Timedelta("400h"):a].dropna()
        r2000 = s.loc[a - pd.Timedelta("2000h"):a].dropna()
        if len(r400) < 500 or len(r2000) < 2000: continue
        def z(r):
            m = float(r.median()); d = float((r - m).abs().median()*1.4826)
            return (float(w.max()) - m)/d if d > 1e-9 else np.nan
        z4, z20 = z(r400), z(r2000)
        fl = "  <<<" if abs(z20) > abs(z4)*1.5 else ""
        print(f"  {c:>12}{float(w.max()):17.2f}{float(r400.median()):11.2f}"
              f"{float(r2000.median()):11.2f}{z4:9.2f}{z20:10.2f}{fl}")

print("\n\nA REFERENCIA ESTA SUBINDO JUNTO?  (mediana rolante ao longo do tempo)")
print("=" * 96)
for c in ("TV_353Y_A", "TV_354X_A", "TV_354Y_A"):
    s = X[c]
    r = s.rolling(n(400), min_periods=500).median()
    q = [float(r.loc[:d].dropna().iloc[-1]) if len(r.loc[:d].dropna()) else np.nan
         for d in ("2025-03-01", "2025-07-01", "2025-11-01", "2026-03-01")]
    dv = (q[-1]-q[0])/q[0]*100 if q[0] else np.nan
    print(f"  {c:>12}  mar/25 {q[0]:6.2f} | jul/25 {q[1]:6.2f} | nov/25 {q[2]:6.2f} | "
          f"mar/26 {q[3]:6.2f}   deriva {dv:+6.1f}%")
print("\n  -> se a referencia sobe junto, o z nao sobe e o detector se adapta a degradacao")
