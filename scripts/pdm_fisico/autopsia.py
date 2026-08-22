#!/usr/bin/env python3
"""Autopsia dos 9 eventos: quais grandezas se afastam do normal antes do trip.

z robusto de cada canal calculado contra os 30 dias de operacao quente que
antecedem a janela de 72 h -- ou seja, o proprio equipamento como referencia,
sem usar nada do futuro.
"""
import numpy as np, pandas as pd

g = pd.read_parquet("grade2min.parquet").drop(columns=["HSX_6240001A"])
ev = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"]

MANC = ["954005_624_TI_0301", "954005_624_TI_0303", "954005_624_TI_0305", "954005_624_TI_0307"]
TC = [f"TC382_0{i}_A" for i in range(1, 7)]
TV = [c for c in g.columns if c.startswith("TV_")]
OLEO = "954005_624_TI_0325"

quente = (g.RUNNING_A > 0.99) & (g["T5_AVG_A"] > 300)

def canais(df):
    """Grandezas fisicas derivadas, todas com sinal 'maior = pior'."""
    c = pd.DataFrame(index=df.index)
    m = df[MANC]
    c["manc_max"] = m.max(axis=1)
    c["manc_spread"] = m.max(axis=1) - m.median(axis=1)
    c["manc_dT_oleo"] = m.max(axis=1) - df[OLEO]           # geracao de calor no mancal
    c["oleo_T"] = df[OLEO]
    c["T5_spread"] = df[TC].max(axis=1) - df[TC].min(axis=1)   # pattern factor
    c["T5_avg"] = df["T5_AVG_A"]
    c["selagem_abs"] = df["954005_624_PDIT_0305"].abs()
    c["vib_max"] = df[TV].max(axis=1)
    c["vib_351"] = df[["TV_351X_A", "TV_351Y_A"]].max(axis=1)
    c["p_oleo_0308"] = -df["954005_624_PI_0308"]           # queda de pressao = ruim
    c["p_0307"] = -df["954005_624_PI_0307"]
    c["dp_0301"] = df["954005_624_PDI_0301"]
    c["dp_0302"] = df["954005_624_PDI_0302"]
    c["dp_0317"] = df["954005_624_PDI_0317"]
    c["dp_0338"] = df["954005_624_PDI_0338"]
    c["gas_0315"] = -df["954005_624_PI_0315"]
    return c

C = canais(g).where(quente)
print(f"canais: {list(C.columns)}\n")

HOR = [48, 24, 12, 6, 2]
linhas = []
for t in ev:
    ref = C[(C.index >= t - pd.Timedelta(days=33)) & (C.index < t - pd.Timedelta(hours=72))].dropna(how="all")
    if len(ref) < 500:
        print(f"[pula] {t.date()}: so {len(ref)} janelas de referencia")
        continue
    med = ref.median(); mad = (ref - med).abs().median() * 1.4826
    mad = mad.replace(0, np.nan)
    for h in HOR:
        w = C[(C.index >= t - pd.Timedelta(hours=h)) & (C.index < t)].dropna(how="all")
        if w.empty:
            continue
        z = ((w - med) / mad).median()
        linhas.append(dict(evento=t.strftime("%Y-%m-%d"), h=h, n=len(w), **z.round(1).to_dict()))

d = pd.DataFrame(linhas)
d.to_csv("autopsia.csv", index=False)
cols = [c for c in d.columns if c not in ("evento", "h", "n")]
for e, sub in d.groupby("evento"):
    print(f"=== {e}   (z robusto mediano na janela, referencia = 30 d antes)")
    print(sub.set_index("h")[cols].to_string())
    print()
