#!/usr/bin/env python3
"""Os 3 sensores significativos sao PRECURSOR ou CASCATA?

TAH_6240305 parecia forte (31,9x) e era cascata -- lead mediano 0,0 h. Mesma
checagem aqui, e e a que decide: se a margem so e consumida na ultima hora, a
informacao chega junto com a falha e nao serve.

Mede QUANDO a margem cruza limiares de 25%, 50% e 75% do consumo, antes de cada
trip.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import g, mask, idx, alvo

LIM = pd.read_csv("limites_alarme.csv")
ALVOS = ["954005_624_TI_0305", "954005_624_PI_0339", "954005_624_PI_0340"]
JAN = pd.Timedelta("72h")
MINM = 0.10

def margem(c):
    r = LIM[LIM.col == c].iloc[0]
    s = g[c].astype("float64").where(mask); tip = float(s.median())
    alto = r["HH"] if pd.notna(r["HH"]) else r["H"]
    baixo = r["LL"] if pd.notna(r["LL"]) else r["L"]
    p = []
    if pd.notna(alto) and alto - tip > MINM*max(abs(tip),1e-6): p.append((s-tip)/(alto-tip))
    if pd.notna(baixo) and tip - baixo > MINM*max(abs(tip),1e-6): p.append((tip-s)/(tip-baixo))
    return pd.concat(p, axis=1).max(axis=1), tip, (alto if pd.notna(alto) else baixo)

for c in ALVOS:
    M, tip, lim = margem(c)
    d = LIM[LIM.col == c].iloc[0]["desc"]
    print(f"\n{c}   (tipico {tip:.2f} -> limite {lim:.2f})")
    print(f"  {str(d)[:70]}")
    print("=" * 92)
    print(f"{'trip':>17}{'pico':>9}{'lead p/ 25%':>13}{'lead p/ 50%':>13}"
          f"{'lead p/ 75%':>13}{'veredito':>16}")
    print("-" * 92)
    leads50 = []
    for t in alvo:
        w = M.loc[t - JAN:t].dropna()
        if not len(w):
            print(f"{t:%d/%m/%Y %H:%M}{'--':>9}"); continue
        pk = float(w.max())
        L = {}
        for q in (0.25, 0.50, 0.75):
            cr = w[w >= q]
            L[q] = (t - cr.index[0]).total_seconds()/3600 if len(cr) else np.nan
        if np.isfinite(L[0.50]): leads50.append(L[0.50])
        vd = ("CASCATA" if np.isfinite(L[0.50]) and L[0.50] < 2
              else ("precursor" if np.isfinite(L[0.50]) else "nao cruza 50%"))
        f = lambda v: f"{v:12.1f}h" if np.isfinite(v) else f"{'--':>13}"
        print(f"{t:%d/%m/%Y %H:%M}{pk:9.3f}{f(L[0.25])}{f(L[0.50])}{f(L[0.75])}{vd:>16}")
    print("-" * 92)
    if leads50:
        print(f"  lead ate consumir 50% da margem: mediana {np.median(leads50):.1f} h, "
              f"min {min(leads50):.1f} h, max {max(leads50):.1f} h  ({len(leads50)}/8 cruzam)")
    else:
        print("  nunca cruza 50% da margem nas 72 h antes de nenhum trip")
