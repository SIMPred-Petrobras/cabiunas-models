#!/usr/bin/env python3
"""Pericia dos eventos que NAO pegamos na banda acionavel.

Para cada evento faltante, varre os 36 sensores procurando QUALQUER movimento
anomalo em [t-48h, t-4h] -- z robusto contra a linha de base das 400 h
anteriores, e margem consumida ate o setpoint. Responde: falta sinal, ou falta
so a regra que o pega?
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import g, partes, mask, idx, alvo, sel
from publica_clearml import SIN, REFRAT_H, DUR_MIN
from plota_estilo_francisco import KB, KV
from precursor_ou_cascata import margem

LIM = pd.read_csv("limites_alarme.csv")
TMIN, TMAX = 4.0, 48.0
JAN = pd.Timedelta(hours=TMAX)

# ponto atual: 4 canais + margem TI_0305 >= 20%
ON = partes(KB, KV)
MG = (margem("954005_624_TI_0305")[0] >= 0.20) & mask
ns = sum(ON[c].astype(int) for c in SIN) + MG.astype(int)
v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
al = pd.Series(False, index=idx); bloq = None
for a, b in AV.episodios(v):
    if bloq is not None and a <= bloq: continue
    al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=REFRAT_H)
fin = pd.Series(False, index=idx)
for a, b in AV.episodios(al):
    if (b-a).total_seconds()/60 + 2 >= DUR_MIN: fin.loc[a:b] = True
fin = fin & sel
eps = AV.episodios(fin)

pega, falta = [], []
for t in alvo:
    c = [a for a, _ in eps if t - JAN <= a <= t - pd.Timedelta(hours=TMIN)]
    (pega if c else falta).append(t)
print(f"ponto atual (4 canais + margem>=20%): banda {len(pega)}/8")
print(f"  pega : {', '.join(x.strftime('%d/%m/%Y') for x in pega)}")
print(f"  FALTA: {', '.join(x.strftime('%d/%m/%Y') for x in falta)}")

# z robusto de cada sensor contra as 400 h anteriores a janela
COLS = [c for c in LIM.col if c in g.columns]
print(f"\n\nVARREDURA DOS {len(COLS)} SENSORES NA JANELA [t-48h, t-4h] DE CADA FALTANTE")
print("=" * 100)
for t in falta:
    a, b = t - JAN, t - pd.Timedelta(hours=TMIN)
    base0 = a - pd.Timedelta("400h")
    print(f"\n### {t:%d/%m/%Y %H:%M}")
    op_frac = float(mask.loc[a:b].mean())
    print(f"  maquina em operacao vigiada em {100*op_frac:.0f}% da janela")
    if op_frac < 0.05:
        print("  -> janela quase toda mascarada; nao ha o que observar"); continue
    lin = []
    for c in COLS:
        s = g[c].astype("float64").where(mask)
        bs = s.loc[base0:a].dropna()
        w = s.loc[a:b].dropna()
        if len(bs) < 500 or len(w) < 50: continue
        med = float(bs.median()); mad = float((bs - med).abs().median()*1.4826)
        if mad < 1e-9: continue
        z = ((w - med)/mad).abs()
        r = LIM[LIM.col == c].iloc[0]
        tip = float(s.median())
        alto = r["HH"] if pd.notna(r["HH"]) else r["H"]
        mg = float(((w - tip)/(alto - tip)).max()) if pd.notna(alto) and alto > tip else np.nan
        lin.append((c, float(z.max()), float(z.quantile(.95)), mg))
    D = pd.DataFrame(lin, columns=["sensor", "z_max", "z_p95", "margem"]).sort_values(
        "z_p95", ascending=False)
    print(f"  {'sensor':>24}{'z max':>9}{'z p95':>9}{'margem':>9}")
    print("  " + "-" * 53)
    for _, r in D.head(6).iterrows():
        fl = "  <<<" if r.z_p95 > 3 else ""
        mg = f"{r.margem:9.2f}" if np.isfinite(r.margem) else f"{'--':>9}"
        print(f"  {r.sensor[-22:]:>24}{r.z_max:9.2f}{r.z_p95:9.2f}{mg}{fl}")
    n3 = int((D.z_p95 > 3).sum())
    print(f"  -> {n3} sensores com z_p95 > 3 na janela"
          + ("   (ha sinal, falta regra)" if n3 else "   (NAO HA SINAL nesta janela)"))
